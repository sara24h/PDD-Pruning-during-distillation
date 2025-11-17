import torch
import argparse
from thop import profile
import torchvision
import torchvision.transforms as transforms

from resnet_kd import resnet20
from trainer.trainer import validate
from utils.claculate_latency import compute_latency_ms_pytorch
from vgg_kd import cvgg11_bn, cvgg11_bn_small

parser = argparse.ArgumentParser(description='Calculating flops and params')

parser.add_argument(
    '--input_image_size',
    type=int,
    default=32,
    help='The input_image_size')
parser.add_argument("--gpu", default=0, type=int, help="Which GPU to use for training")
parser.add_argument("--arch", default='resnet20_small', type=str, help="arch")
parser.add_argument("--pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("--num_classes", default=10, type=int, help="number of class")
parser.add_argument("--finetune", action="store_true", help="finetune pre-trained model")
parser.add_argument("--set", help="name of dataset", type=str, default='cifar10')
parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument("--print-freq", default=100, type=int, metavar="N", help="print frequency (default: 10)")
args = parser.parse_args()

# Set GPU device
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    print(f"Using GPU {args.gpu}")
else:
    print("CUDA not available, using CPU")


def get_dataset(args):
    """Load CIFAR-10 dataset with proper transforms"""
    if args.set == 'cifar10':
        # Training transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Test transforms
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True,
            download=True, 
            transform=transform_train
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=128,
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )

        testset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False,
            download=True, 
            transform=transform_test
        )
        val_loader = torch.utils.data.DataLoader(
            testset, 
            batch_size=128,
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    else:
        raise ValueError(f"Unsupported dataset: {args.set}")


# ================================================================================
# Model Creation and Loading
# ================================================================================

print("="*80)
print("Creating Model...")
print("="*80)

if args.arch == 'cvgg11_bn_small':
    print("Loading VGG11 Small model...")
    model = cvgg11_bn_small(
        finding_masks=False, 
        num_classes=args.num_classes, 
        batch_norm=True
    ).cuda()
    model.classifier[1] = torch.nn.Linear(398, 512).cuda()
    
    # Load trained model weights
    checkpoint_path = 'pretrained_model/cvgg11_bn_small/cifar100/T_cvgg16_bn_S_cvgg11_bn_small_cifar100.pt'
    if args.pretrained:
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=f'cuda:{args.gpu}')
        model.load_state_dict(ckpt)
        print("✓ Checkpoint loaded successfully!")

elif args.arch == 'resnet20_small':
    print("Loading ResNet20 Small model...")
    
    # Configuration from mask file (active neurons per layer)
    in_cfg = [3, 16, 14, 13, 28, 21, 24, 47, 69, 50]
    out_cfg = [16, 14, 13, 28, 21, 24, 47, 69, 50, 49]
    
    print(f"Input channels config: {in_cfg}")
    print(f"Output channels config: {out_cfg}")
    
    # Create model with dynamic masks
    model = resnet20(
        finding_masks=True,  # Enable mask parameters
        in_cfg=in_cfg, 
        out_cfg=out_cfg, 
        num_classes=args.num_classes
    ).cuda()
    
    # Load trained model weights (NOT just masks)
    model_checkpoint_path = '/kaggle/working/pretrained_model/resnet56/cifar10/cifar10_resnet20.pt'
    mask_checkpoint_path = '/kaggle/working/pretrained_model/resnet56/cifar10/cifar10_T_resnet56_S_resnet20_mask.pt'
    
    print(f"\nLoading model weights from: {model_checkpoint_path}")
    
    try:
        # Try to load the full model checkpoint first
        model_ckpt = torch.load(model_checkpoint_path, map_location=f'cuda:{args.gpu}')
        model.load_state_dict(model_ckpt, strict=False)
        print("✓ Model weights loaded successfully!")
    except FileNotFoundError:
        print(f"⚠ Model checkpoint not found at {model_checkpoint_path}")
        print(f"   Attempting to load mask only from {mask_checkpoint_path}")
        
        # If model checkpoint doesn't exist, load mask only (will give poor accuracy)
        mask_ckpt = torch.load(mask_checkpoint_path, map_location=f'cuda:{args.gpu}')
        
        # Extract only mask parameters
        mask_dict = {}
        for k, v in mask_ckpt.items():
            if "mask" in k:
                mask_dict[k] = v
        
        model.load_state_dict(mask_dict, strict=False)
        print("⚠ Only masks loaded - model not trained yet!")
        print("   Run training first to get good accuracy!")
    
    print(f"✓ Model created and loaded")

else:
    raise ValueError(f"Unsupported architecture: {args.arch}")


# ================================================================================
# Model Evaluation
# ================================================================================

model.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()

print("\n" + "="*80)
print("Loading Dataset...")
print("="*80)
train_loader, val_loader = get_dataset(args)
print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

if args.evaluate:
    print("\n" + "="*80)
    print("Evaluating Model on Validation Set...")
    print("="*80)
    
    if args.set in ['cifar10', 'cifar100']:
        acc1, acc5 = validate(val_loader, model, criterion, args)
        print(f'\n{"="*80}')
        print(f'Final Accuracy:')
        print(f'  Top-1: {acc1:.2f}%')
        print(f'  Top-5: {acc5:.2f}%')
        print(f'{"="*80}\n')
    else:
        print("No validation implemented for this dataset")


# ================================================================================
# Calculate FLOPs, Params, and Latency
# ================================================================================

print("="*80)
print("Calculating Model Statistics...")
print("="*80)

input_image_size = args.input_image_size
print(f'Input image size: {input_image_size}x{input_image_size}')

input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()

# Calculate FLOPs and Parameters
flops, params = profile(model, inputs=(input_image,))

# Calculate Latency
latency = compute_latency_ms_pytorch(model, input_image, iterations=None)

# Display Results
print("\n" + "="*80)
print("Model Statistics:")
print("="*80)
print(f'Parameters: {params:,.0f} ({params/1e6:.2f}M)')
print(f'FLOPs: {flops:,.0f} ({flops/1e6:.2f}M)')
print(f'Latency: {latency:.2f} ms')
print("="*80)
