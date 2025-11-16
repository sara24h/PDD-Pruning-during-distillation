import torch
import argparse
from thop import profile
import torchvision

from resnet_kd import resnet20
from trainer.trainer import validate
from utils.claculate_latency import compute_latency_ms_pytorch

from data.Data import CIFAR10, CIFAR100
#from utils.get_model import get_model
from vgg_kd import cvgg11_bn, cvgg11_bn_small

def get_dataset(args):
    """تابعی برای بارگذاری دیتاست"""
    if args.set == 'cifar10':
        return CIFAR10(batch_size=128)  # یا هر batch_size مناسب دیگر
    elif args.set == 'cifar100':
        return CIFAR100(batch_size=128)  # یا هر batch_size مناسب دیگر
    else:
        raise ValueError(f"Dataset {args.set} not supported")

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

# python calculating_flops.py --gpu 3 --arch cvgg11_bn_small --pretrained --evaluate

if args.arch == 'cvgg11_bn_small':
    mask = torch.load('/public/ly/xianyu/pretrained_model/cvgg16_bn/cifar100/cifar100_T_cvgg16_bn_S_cvgg11_bn_mask.pt')  # 要手动调整
    print(mask['layer_num'])

    model = cvgg11_bn_small(finding_masks=False, num_classes=args.num_classes, batch_norm=True).cuda()
    model.classifier[1] = torch.nn.Linear(398, 512).cuda()
    ckpt = torch.load('/public/ly/xianyu/pretrained_model/cvgg11_bn_small/cifar100/T_cvgg16_bn_S_cvgg11_bn_small_cifar100.pt')  # 要手动调整
    model.load_state_dict(ckpt)


if args.arch == 'resnet20_small':
    # in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]  # 第一层不减
    # out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
    in_cfg = [3, 16, 14, 13, 28, 21, 24, 47, 69, 50]  # 要手动调整
    out_cfg = [16, 14, 13, 28, 21, 24, 47, 69, 50, 49]

    model = resnet20(finding_masks=False, in_cfg=in_cfg, out_cfg=out_cfg, num_classes=args.num_classes).cuda()
    mask = torch.load('/kaggle/working/pretrained_model/resnet56/cifar10/cifar10_T_resnet56_S_resnet20_mask.pt')
    ckpt = torch.load('/kaggle/working/pretrained_model/resnet56/cifar10/cifar10_resnet20.pt',map_location='cuda:%d' % args.gpu)  # 要手动调整
    model.load_state_dict(ckpt)

model.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()
data = get_dataset(args)

if args.evaluate:
    if args.set in ['cifar10', 'cifar100']:
        acc1, acc5 = validate(data.val_loader, model, criterion, args)
    else:
        # تابع validate_ImageNet را باید تعریف یا import کنید
        # acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)
        acc1, acc5 = validate(data.val_loader, model, criterion, args)  # جایگزین موقت

    print('Acc is {}'.format(acc1))


# calculate model size
input_image_size = args.input_image_size
print('image size is {}'.format(input_image_size))
input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
flops, params = profile(model, inputs=(input_image,))
latency = compute_latency_ms_pytorch(model, input_image, iterations=None)

print('Params: %.2f' % (params))
print('Flops: %.2f' % (flops))

print('Latency: %.2f' % (latency))

