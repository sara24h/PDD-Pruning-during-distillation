import os
import sys
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
import torch.nn.functional as F

from args import args
import datetime
from data.Data import CIFAR10, CIFAR100
from model.VGG_cifar import cvgg16_bn, cvgg19_bn
from model.samll_resnet import resnet56, resnet110
from resnet_kd import resnet20
from trainer.trainer import validate, train, train_KD
from utils.utils import set_random_seed, set_gpu, Logger, get_logger, get_lr
from vgg_kd import cvgg11_bn


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
    logger.info(f"arch: {args.arch}")
    logger.info(f"dataset: {args.set}")
    logger.info(f"batch_size: {args.batch_size}")
    logger.info(f"weight_decay: {args.weight_decay}")
    logger.info(f"lr: {args.lr}")
    logger.info(f"epochs: {args.epochs}")
    logger.info(f"lr_decay_step: {args.lr_decay_step}")
    logger.info(f"num_classes: {args.num_classes}")

    # Initialize student model
    if args.arch_s == 'cvgg11_bn':
        model_s = cvgg11_bn(finding_masks=True, num_classes=args.num_classes, batch_norm=True)
    elif args.arch_s == 'resnet20':
        in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]
        out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
        model_s = resnet20(finding_masks=True, in_cfg=in_cfg, out_cfg=out_cfg, num_classes=args.num_classes)
    else:
        raise ValueError(f"Unsupported student arch: {args.arch_s}")

    # Initialize teacher model
    if args.arch == 'cvgg16_bn':
        model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            path = f'/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/{args.set}/scores.pt'
            ckpt = torch.load(path, map_location=f'cuda:{args.gpu}')
            model.load_state_dict(ckpt)

    elif args.arch == 'cvgg19_bn':
        model = cvgg19_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            path = f'/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/{args.set}/scores.pt'
            ckpt = torch.load(path, map_location=f'cuda:{args.gpu}')
            model.load_state_dict(ckpt)

    elif args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                state_dict = torch.hub.load_state_dict_from_url(
                    'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt',
                    map_location=f'cuda:{args.gpu}'
                )
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
            elif args.set == 'cifar100':
                path = '/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56/cifar100/scores.pt'
                ckpt = torch.load(path, map_location=f'cuda:{args.gpu}')
                model.load_state_dict(ckpt)

    elif args.arch == 'resnet110':
        model = resnet110(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110.th', map_location=f'cuda:{args.gpu}')
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
                model.load_state_dict(ckpt)
            elif args.set == 'cifar100':
                path = '/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110/cifar100/scores.pt'
                ckpt = torch.load(path, map_location=f'cuda:{args.gpu}')
                model.load_state_dict(ckpt)
    else:
        raise ValueError(f"Unsupported teacher arch: {args.arch}")

    model_s = set_gpu(args, model_s)
    model = set_gpu(args, model)

    # Freeze teacher model
    for param in model.parameters():
        param.requires_grad = False

    # Logging gradients (optional)
    for name, param in model.named_parameters():
        print(f"Teacher Model Param: {name}, Requires Grad: {param.requires_grad}")
    print('-' * 100)
    for name, param in model_s.named_parameters():
        print(f"Student Model Param: {name}, Requires Grad: {param.requires_grad}")

    model.eval()

    divergence_loss = F.kl_div
    criterion = nn.CrossEntropyLoss().cuda()

    # Load appropriate dataset
    if args.set == 'cifar10':
        data = CIFAR10()
    elif args.set == 'cifar100':
        data = CIFAR100()
    else:
        raise ValueError(f"Unsupported dataset: {args.set}")

    # Validate teacher accuracy
    acc1, acc5 = validate(data.val_loader, model, criterion, args)
    print(f"Teacher model top-1 accuracy: {acc1:.2f}%")

    # Student optimizer & scheduler
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
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0
    args.start_epoch = args.start_epoch or 0

    for epoch in range(args.start_epoch, args.epochs):
        train_acc1, train_acc5 = train_KD(
            data.train_loader, model, model_s, divergence_loss, criterion, optimizer, epoch, args
        )
        acc1, acc5 = validate(data.val_loader, model_s, criterion, args)
        scheduler.step()

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
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

                logger.info(f"Best Val Acc@1: {acc1}")
                logger.info(f"Layer sparsity (non-zero counts): {layer_num}")

                torch.save({
                    'layer_num': layer_num,
                    'mask': mask_list
                }, os.path.join(save_dir, f"{args.set}_T_{args.arch}_S_{args.arch_s}_mask.pt"))

                torch.save(model_s.state_dict(), os.path.join(save_dir, f"{args.set}_{args.arch_s}.pt"))

    logger.info(f"Final Best Student Acc@1: {best_acc1:.2f}%")


if __name__ == "__main__":
    # Example: python train_kd.py --gpu 3 --arch cvgg16_bn --set cifar10 --lr 0.01 --batch_size 256 --weight_decay 0.005 --epochs 50 --lr_decay_step 20,40 --num_classes 10 --pretrained --arch_s cvgg11_bn
    main()
