import torch
import argparse
from thop import profile
import torchvision
import torchvision.transforms as transforms  # Import transforms

from resnet_kd import resnet20
from trainer.trainer import validate
from utils.claculate_latency import compute_latency_ms_pytorch
# from utils.get_dataset import get_dataset  # Remove
# from utils.get_model import get_model  # Remove
from vgg_kd import cvgg11_bn, cvgg11_bn_small

parser = argparse.ArgumentParser(description='Calculating flops and params')

parser.add_argument(
    '--input_image_size',
    type=int,
    default=32,
    help='The input_image_size')
parser.add_argument("--gpu", default=None, type=int, help="Which GPU to use for training")
parser.add_argument("--arch", default=None, type=str, help="arch")
parser.add_argument("--pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("--num_classes", default=10, type=int, help="number of class")
parser.add_argument("--finetune", action="store_true", help="finetune pre-trained model")
parser.add_argument("--set", help="name of dataset", type=str, default='cifar10')
parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument("--print-freq", default=100, type=int, metavar="N", help="print frequency (default: 10)")
args = parser.parse_args()
torch.cuda.set_device(args.gpu)


def get_dataset(args):
    if args.set == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((args.input_image_size, args.input_image_size)),  # Resize added
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='/public/MountData/dataset/cifar10', train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='/public/MountData/dataset/cifar10', train=False,
                                               download=True, transform=transform)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                 shuffle=False, num_workers=2)
        return train_loader, val_loader  # Return both loaders
    else:
        raise ValueError("Unsupported dataset: {}".format(args.set))


# model = get_model(args).cuda()

# python calculating_flops.py --gpu 3 --arch cvgg11_bn_small --pretrained --evaluate

if args.arch == 'cvgg11_bn_small':
    # mask = torch.load('/public/ly/xianyu/pretrained_model/cvgg16_bn/cifar100/cifar100_T_cvgg16_bn_S_cvgg11_bn_mask.pt')  # 要手动调整
    # print(mask['layer_num'])

    model = cvgg11_bn_small(finding_masks=False, num_classes=args.num_classes, batch_norm=True).cuda()
    model.classifier[1] = torch.nn.Linear(398, 512).cuda()
    # ckpt = torch.load('/public/ly/xianyu/pretrained_model/cvgg11_bn_small/cifar100/T_cvgg16_bn_S_cvgg11_bn_small_cifar100.pt')  # 要手动调整
    # model.load_state_dict(ckpt)


if args.arch == 'resnet20_small':
    in_cfg = [3, 16, 14, 13, 28, 21, 24, 47, 69, 50]  # 要手动調整
    out_cfg = [16, 14, 13, 28, 21, 24, 47, 69, 50, 49]

    model = resnet20(finding_masks=True, in_cfg=in_cfg, out_cfg=out_cfg, num_classes=args.num_classes).cuda() # Changed to True
    ckpt = torch.load('/kaggle/working/pretrained_model/resnet56/cifar10/cifar10_T_resnet56_S_resnet20_mask.pt',map_location='cuda:%d' % args.gpu)  # 要手动調整
    
    # creates a new ordered dictionary that only contains the mask parameters from the loaded checkpoint
    new_ckpt = {}
    for k, v in ckpt.items():
        if "mask" in k:
            new_ckpt[k] = v
    model.load_state_dict(new_ckpt, strict=False)  #只加载部分参数，设置为 False



model.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()
train_loader, val_loader = get_dataset(args) # Use the get_dataset function


if args.evaluate:
    if args.set in ['cifar10', 'cifar100']:
        acc1, acc5 = validate(val_loader, model, criterion, args) # Use val_loader here
    else:
        print("No validation implemented for this dataset") #Removed: acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)

    print('Acc is {}'.format(acc1))


input_image_size = args.input_image_size
print('image size is {}'.format(input_image_size))
input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
flops, params = profile(model, inputs=(input_image,))
latency = compute_latency_ms_pytorch(model, input_image, iterations=None)

print('Params: %.2f' % (params))
print('Flops: %.2f' % (flops))
print('Latency: %.2f' % (latency))
