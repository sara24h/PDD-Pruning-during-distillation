import torch
import argparse
from thop import profile
import torchvision
import torchvision.transforms as transforms
from resnet_kd import resnet20
from trainer.trainer import validate
from utils.claculate_latency import compute_latency_ms_pytorch
from vgg_kd import cvgg11_bn, cvgg11_bn_small
import os

parser = argparse.ArgumentParser(description='Evaluating pruned model')
parser.add_argument(
    '--input_image_size',
    type=int,
    default=32,
    help='The input_image_size')
parser.add_argument("--gpu", default=0, type=int, help="Which GPU to use for training")
parser.add_argument("--arch", default='resnet56', type=str, help="teacher architecture")
parser.add_argument("--arch_s", default='resnet20', type=str, help="student architecture")
parser.add_argument("--num_classes", default=10, type=int, help="number of class")
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
    elif args.set == 'cifar100':
        # Training transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
       
        # Test transforms
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
       
        trainset = torchvision.datasets.CIFAR100(
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
        testset = torchvision.datasets.CIFAR100(
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

def load_mask(args):
    """Load mask information from file"""
    mask_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_T_{args.arch}_S_{args.arch_s}_mask.pt'
    
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found at {mask_path}")
    
    mask_data = torch.load(mask_path)
    layer_num = mask_data['layer_num']
    mask_list = mask_data['mask']
    
    print(f"Loaded mask from {mask_path}")
    print(f"Active neurons per layer: {layer_num}")
    
    return layer_num, mask_list

def create_pruned_model(args):
    """Create a pruned model based on the extracted masks"""
    # Load mask information
    layer_num, mask_list = load_mask(args)
    
    if args.arch_s == 'cvgg11_bn':
        # Calculate new channel configurations
        in_cfg = [3]  # Input channels
        out_cfg = layer_num  # Output channels for each layer
        
        print(f"Creating pruned VGG11 with {len(out_cfg)} layers")
        print(f"Output channels config: {out_cfg}")
        
        # Create pruned model
        model = cvgg11_bn(
            finding_masks=False,  # No masks needed for final model
            num_classes=args.num_classes,
            batch_norm=True
        ).cuda()
        
    elif args.arch_s == 'resnet20':
        # Calculate new channel configurations
        in_cfg = [3]  # Input channels
        out_cfg = layer_num  # Output channels for each layer
        
        print(f"Creating pruned ResNet20 with {len(out_cfg)} layers")
        print(f"Input channels config: {in_cfg}")
        print(f"Output channels config: {out_cfg}")
        
        # Create pruned model
        model = resnet20(
            finding_masks=False,  # No masks needed for final model
            in_cfg=in_cfg,
            out_cfg=out_cfg,
            num_classes=args.num_classes,
            option='B'
        ).cuda()
    
    else:
        raise ValueError(f"Unsupported student architecture: {args.arch_s}")
    
    return model

# ================================================================================
# Model Creation and Loading
# ================================================================================
print("="*80)
print("Creating Pruned Model...")
print("="*80)

# Try to load fine-tuned pruned model first
finetuned_model_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_T_{args.arch}_S_{args.arch_s}_pruned_finetuned.pt'
pruned_model_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_T_{args.arch}_S_{args.arch_s}_pruned.pt'

if os.path.exists(finetuned_model_path):
    print(f"Loading fine-tuned pruned model from: {finetuned_model_path}")
    model = create_pruned_model(args)
    model.load_state_dict(torch.load(finetuned_model_path))
    print("✓ Fine-tuned pruned model loaded successfully!")
elif os.path.exists(pruned_model_path):
    print(f"Loading pruned model from: {pruned_model_path}")
    model = create_pruned_model(args)
    model.load_state_dict(torch.load(pruned_model_path))
    print("✓ Pruned model loaded successfully!")
else:
    print("No pruned model found. Please run train_auto_prune.py first.")
    exit()

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
    print("Evaluating Pruned Model on Validation Set...")
    print("="*80)
   
    if args.set in ['cifar10', 'cifar100']:
        acc1, acc5 = validate(val_loader, model, criterion, args)
        print(f'\n{"="*80}')
        print(f'Pruned Model Accuracy:')
        print(f' Top-1: {acc1:.2f}%')
        print(f' Top-5: {acc5:.2f}%')
        print(f'{"="*80}\n')
    else:
        print("No validation implemented for this dataset")

# ================================================================================
# Calculate FLOPs, Params, and Latency
# ================================================================================
print("="*80)
print("Calculating Pruned Model Statistics...")
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
print("Pruned Model Statistics:")
print("="*80)
print(f'Parameters: {params:,.0f} ({params/1e6:.2f}M)')
print(f'FLOPs: {flops:,.0f} ({flops/1e6:.2f}M)')
print(f'Latency: {latency:.2f} ms')
print("="*80)

# Calculate compression ratio
original_model_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_{args.arch_s}.pt'
if os.path.exists(original_model_path):
    # Load original model for comparison
    if args.arch_s == 'cvgg11_bn':
        original_model = cvgg11_bn(finding_masks=False, num_classes=args.num_classes, batch_norm=True).cuda()
    elif args.arch_s == 'resnet20':
        original_model = resnet20(finding_masks=False, num_classes=args.num_classes, option='B').cuda()
    
    original_model.load_state_dict(torch.load(original_model_path))
    original_model.eval()
    
    # Calculate original model stats
    original_flops, original_params = profile(original_model, inputs=(input_image,))
    
    # Calculate compression ratios
    params_ratio = (1 - params/original_params) * 100
    flops_ratio = (1 - flops/original_flops) * 100
    
    print("\n" + "="*80)
    print("Compression Ratios:")
    print("="*80)
    print(f'Original Parameters: {original_params:,.0f} ({original_params/1e6:.2f}M)')
    print(f'Pruned Parameters: {params:,.0f} ({params/1e6:.2f}M)')
    print(f'Parameter Reduction: {params_ratio:.2f}%')
    print(f'Original FLOPs: {original_flops:,.0f} ({original_flops/1e6:.2f}M)')
    print(f'Pruned FLOPs: {flops:,.0f} ({flops/1e6:.2f}M)')
    print(f'FLOPs Reduction: {flops_ratio:.2f}%')
    print("="*80)
