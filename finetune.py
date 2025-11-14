import datetime
import os
import sys
import argparse

import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms

from args import args
from data.Data import CIFAR10, CIFAR100
from resnet_kd import resnet20
from trainer.trainer import validate, train
from utils.utils import set_gpu, get_logger, Logger, set_random_seed
from vgg_kd import cvgg11_bn, cvgg11_bn_small

def main():
    print(args)
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file = f'print_process_{now}.log'
    sys.stdout = Logger(log_file, sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)


def main_worker(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = f'pretrained_model/{args.arch_s}/{args.set}'
    os.makedirs(save_dir, exist_ok=True)
    
    logger = get_logger(f'{save_dir}/logger_{now}.log')
    logger.info(f"Architecture: {args.arch}")
    logger.info(f"Student Architecture: {args.arch_s}")
    logger.info(f"Dataset: {args.set}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"LR decay steps: {args.lr_decay_step}")
    logger.info(f"Number of classes: {args.num_classes}")

    # بارگذاری داده‌ها
    if args.set == 'cifar10':
        data = CIFAR10()
    elif args.set == 'cifar100':
        data = CIFAR100()
    else:
        raise ValueError(f"Unsupported dataset: {args.set}")

    # تنظیم پیکربندی‌های معماری
    if args.arch_s == 'cvgg11_bn_small':
        # مسیر فایل mask را به مسیر واقعی در محیط شما تنظیم کنید
        mask_path = f'{save_dir}/mask.pt'
        if not os.path.exists(mask_path):
            logger.error(f"Mask file not found at {mask_path}")
            raise FileNotFoundError(f"Mask file not found at {mask_path}")
        
        mask = torch.load(mask_path)
        logger.info(f"Mask layer numbers: {mask['layer_num']}")

        model = cvgg11_bn(num_classes=args.num_classes, batch_norm=True)
        model_s = cvgg11_bn_small(num_classes=args.num_classes, batch_norm=True)
        
        # مسیر چک‌پوینت مدل اصلی را تنظیم کنید
        ckpt_path = f'{save_dir}/teacher_model.pt'
        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint file not found at {ckpt_path}")
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
        
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)
        model_s.classifier[1] = nn.Linear(mask['layer_num'][-1], 512)

    elif args.arch_s == 'resnet20_small':
        # پیکربندی‌های لایه‌ها برای ResNet20
        in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]
        out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
        
        # این مقادیر را باید با مقادیر واقعی از mask خود جایگزین کنید
        in_cfg_s = [3, 16, 11, 14, 24, 25, 30, 61, 64, 53]
        out_cfg_s = [16, 11, 14, 24, 25, 30, 61, 64, 53, 44]

        # ساخت مدل‌ها بدون پارامتر finding_masks
        model = resnet20(in_cfg=in_cfg, out_cfg=out_cfg, num_classes=args.num_classes)
        
        # مسیر چک‌پوینت مدل اصلی را تنظیم کنید
        ckpt_path = f'{save_dir}/teacher_model.pt'
        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint file not found at {ckpt_path}")
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location=f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(ckpt)
        
        # ساخت مدل دانش‌آموز بدون پارامتر finding_masks
        model_s = resnet20(in_cfg=in_cfg_s, out_cfg=out_cfg_s, num_classes=args.num_classes)
    else:
        raise ValueError(f"Unsupported student architecture: {args.arch_s}")

    # تنظیم GPU
    model_s = set_gpu(args, model_s)
    model = set_gpu(args, model)
    logger.info(f"Student model architecture:\n{model_s}")

    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    
    # اعتبارسنجی مدل معلم
    acc1, acc5 = validate(data.val_loader, model, criterion, args)
    logger.info(f"Teacher model validation accuracy: {acc1:.2f}%")

    # بارگذاری وزن‌ها از مدل معلم به مدل دانش‌آموز
    if args.arch_s == 'cvgg11_bn_small':
        load_vgg_model(model_s, model.state_dict())
    elif args.arch_s == 'resnet20_small':
        load_resnet_model(model_s, model.state_dict(), layer=20)

    # تنظیم بهینه‌ساز
    optimizer = optim.SGD(model_s.parameters(), lr=args.lr, 
                          momentum=args.momentum, weight_decay=args.weight_decay)
    
    # تنظیم کاهش نرخ یادگیری
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    best_acc1 = 0.0
    best_acc5 = 0.0

    # شروع آموزش
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # آموزش
        train_acc1, train_acc5 = train(data.train_loader, model_s, criterion, optimizer, epoch, args)
        logger.info(f"Train accuracy: {train_acc1:.2f}%, {train_acc5:.2f}%")
        
        # اعتبارسنجی
        acc1, acc5 = validate(data.val_loader, model_s, criterion, args)
        logger.info(f"Validation accuracy: {acc1:.2f}%, {acc5:.2f}%")
        
        # به‌روزرسانی نرخ یادگیری
        scheduler.step()
        
        # ذخیره بهترین مدل
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_acc5 = acc5
            logger.info(f"New best accuracy: {best_acc1:.2f}%")
            
            # ذخیره مدل بهترین
            model_path = f'{save_dir}/best_model_T_{args.arch}_S_{args.arch_s}_{args.set}.pt'
            torch.save(model_s.state_dict(), model_path)
            logger.info(f"Best model saved to {model_path}")

        # ذخیره مدل در هر دوره
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            checkpoint_path = f'{save_dir}/checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_s.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc1': best_acc1,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    # ذخیره مدل نهایی
    final_model_path = f'{save_dir}/final_model_T_{args.arch}_S_{args.arch_s}_{args.set}.pt'
    torch.save(model_s.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    logger.info(f"Best validation accuracy: {best_acc1:.2f}%")


def load_vgg_model(model, oristate_dict):
    state_dict = model.state_dict()
    last_select_index = None

    # مسیر فایل mask را تنظیم کنید
    mask_path = f'pretrained_model/{args.arch_s}/{args.set}/mask.pt'
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found at {mask_path}")
    
    mask = torch.load(mask_path)
    logger.info(f"Mask layer numbers: {mask['layer_num']}")
    
    cnt = -1
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            cnt += 1
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:
                select_index = torch.argsort(mask['mask'][cnt])[orifilter_num - currentfilter_num:]
                select_index.sort()

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
            else:
                state_dict[name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)


def load_resnet_model(model, oristate_dict, layer):
    cfg = {
        20: [3, 3, 3],
    }

    state_dict = model.state_dict()
    current_cfg = cfg[layer]
    last_select_index = None
    all_conv_weight = []

    # مسیر فایل mask را تنظیم کنید
    mask_path = f'pretrained_model/{args.arch_s}/{args.set}/mask.pt'
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found at {mask_path}")
    
    mask = torch.load(mask_path)
    logger.info(f"Mask layer numbers: {mask['layer_num']}")
    
    cnt = -1
    for layer_idx, num in enumerate(current_cfg):
        layer_name = f'layer{layer_idx + 1}.'
        for k in range(num):
            cnt += 1
            for l in range(2):
                conv_name = f'{layer_name}{k}.conv{l + 1}'
                conv_weight_name = f'{conv_name}.weight'
                all_conv_weight.append(conv_weight_name)
                
                if conv_weight_name not in oristate_dict or conv_weight_name not in state_dict:
                    continue
                    
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    select_index = torch.argsort(mask['mask'][cnt])[orifilter_num - currentfilter_num:]
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight and conv_name in oristate_dict:
                state_dict[conv_name] = oristate_dict[conv_name]

    model.load_state_dict(state_dict)


if __name__ == "__main__":
    # مثال استفاده:
    # python finetune.py --gpu 0 --arch_s resnet20_small --set cifar10 --lr 0.01 --batch_size 128 --weight_decay 0.005 --epochs 10 --lr_decay_step 5,8 --num_classes 10 --arch resnet56
    parser = argparse.ArgumentParser(description='PDD Fine-tuning')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--arch', type=str, default='resnet56', help='Teacher architecture')
    parser.add_argument('--arch_s', type=str, default='resnet20_small', help='Student architecture')
    parser.add_argument('--set', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr_decay_step', type=str, default='5,8', help='LR decay steps')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every n epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch')
    parser.add_argument('--log_root', type=str, default='log', help='Log root directory')
    args = parser.parse_args()
    
    main()
