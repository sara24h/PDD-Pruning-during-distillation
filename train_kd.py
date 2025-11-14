import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import datetime

from args import args
from data.Data import CIFAR10, CIFAR100

# --- Models ---
# Teacher: استاندارد (بدون Mask، سازگار با chenyaofo)
from model.resnet_standard import resnet56 as resnet56_teacher

# Student: سفارشی (با Mask، in_cfg، out_cfg)
from model.samll_resnet import resnet20 as resnet20_student

# Other models (if needed)
from model.VGG_cifar import cvgg16_bn, cvgg19_bn
from vgg_kd import cvgg11_bn

from trainer.trainer import validate, train_KD
from utils.utils import set_random_seed, set_gpu, Logger, get_logger

# -----------------------------
# Helper function (if not defined elsewhere)
def ApproxSign(mask):
    out_forward = torch.sign(mask)
    mask1 = mask < -1
    mask2 = mask < 0
    mask3 = mask < 1
    out1 = (-1) * mask1.type(torch.float32) + (mask * mask + 2 * mask) * (1 - mask1.type(torch.float32))
    out2 = out1 * mask2.type(torch.float32) + (-mask * mask + 2 * mask) * (1 - mask2.type(torch.float32))
    out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
    out = out_forward.detach() - out3.detach() + out3
    out = (out + 1) / 2
    return out

# -----------------------------
def main():
    print(args)
    sys.stdout = Logger('print process.log', sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)

def main_worker(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = os.path.join('pretrained_model', args.arch, args.set)
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(os.path.join(save_dir, f'logger{now}.log'))

    for key, val in vars(args).items():
        logger.info(f"{key}: {val}")

    # === Student Model (custom with Mask) ===
    if args.arch_s == 'resnet20':
        in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]
        out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
        model_s = resnet20_student(finding_masks=True, in_cfg=in_cfg, out_cfg=out_cfg, num_classes=args.num_classes)
    elif args.arch_s == 'cvgg11_bn':
        model_s = cvgg11_bn(finding_masks=True, num_classes=args.num_classes, batch_norm=True)
    else:
        raise ValueError(f"Unsupported student arch: {args.arch_s}")

    # === Teacher Model (standard, pretrained-compatible) ===
    if args.arch == 'resnet56':
        model = resnet56_teacher(num_classes=args.num_classes)
        if args.pretrained and args.set == 'cifar10':
            state_dict = torch.hub.load_state_dict_from_url(
                'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt',
                map_location=f'cuda:{args.gpu}'
            )
            model.load_state_dict(state_dict)
        elif args.pretrained and args.set == 'cifar100':
            raise NotImplementedError("Pretrained resnet56 for CIFAR100 not available from chenyaofo.")
    else:
        raise ValueError(f"Unsupported teacher arch: {args.arch}")

    model_s = set_gpu(args, model_s)
    model = set_gpu(args, model)

    # Freeze teacher
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Data
    data = CIFAR10() if args.set == 'cifar10' else CIFAR100()

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()
    divergence_loss = F.kl_div

    # Validate teacher
    acc1, acc5 = validate(data.val_loader, model, criterion, args)
    print(f"Teacher Acc@1: {acc1:.2f}%")

    # Optimizer for student
    optimizer = torch.optim.SGD(
        model_s.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    # Training loop
    best_acc1 = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_acc1, train_acc5 = train_KD(
            data.train_loader, model, model_s, divergence_loss, criterion, optimizer, epoch, args
        )
        acc1, acc5 = validate(data.val_loader, model_s, criterion, args)
        scheduler.step()

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            model_s.hook_masks()
            masks = model_s.get_masks()
            mask_list = []
            layer_num = []
            for key in masks.keys():
                msk = ApproxSign(masks[key].mask).squeeze()
                total = torch.sum(msk)
                layer_num.append(int(total.cpu().detach().numpy()))
                mask_list.append(msk)
            model_s.remove_hooks()

            logger.info(f"Best Acc@1: {acc1}")
            torch.save({
                'layer_num': layer_num,
                'mask': mask_list
            }, os.path.join(save_dir, f"{args.set}_T_{args.arch}_S_{args.arch_s}_mask.pt"))

            torch.save(model_s.state_dict(), os.path.join(save_dir, f"{args.set}_{args.arch_s}.pt"))

    logger.info(f"Final Best Student Acc@1: {best_acc1:.2f}%")

if __name__ == "__main__":
    main()
