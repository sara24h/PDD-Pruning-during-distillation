import os
import sys
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
#from Hrank_resnet import resnet_110
from args import args
import datetime
from data.Data import CIFAR10, CIFAR100
from model.VGG_cifar import cvgg16_bn, cvgg19_bn
from model.samll_resnet import resnet56, resnet110
from resnet_kd import resnet20
from trainer.trainer import validate, train, train_KD
from utils.utils import set_random_seed, set_gpu, Logger, get_logger, get_lr
from vgg_kd import cvgg11_bn
import torch.nn.functional as F

def main():
    print(args)
    sys.stdout = Logger('print process.log', sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)


def main_worker(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('pretrained_model/' + args.arch + '/' + args.set):
        os.makedirs('pretrained_model/' + args.arch + '/' + args.set, exist_ok=True)
    logger = get_logger('pretrained_model/' + args.arch + '/' + args.set + '/logger' + now + '.log')
    logger.info(args.arch)
    logger.info(args.set)
    logger.info(args.batch_size)
    logger.info(args.weight_decay)
    logger.info(args.lr)
    logger.info(args.epochs)
    logger.info(args.lr_decay_step)
    logger.info(args.num_classes)

    # ایجاد مدل Student
    if args.arch_s == 'cvgg11_bn':
        model_s = cvgg11_bn(finding_masks=True, num_classes=args.num_classes, batch_norm=True)
    elif args.arch_s == 'resnet20':
        in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]
        out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
        model_s = resnet20(finding_masks=True, in_cfg=in_cfg, out_cfg=out_cfg, num_classes=args.num_classes)

    # ایجاد و بارگذاری مدل Teacher
    if args.arch == 'cvgg16_bn':
        model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_bn':
        model = cvgg19_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                # دانلود از اینترنت به جای فایل لوکال
                print("Downloading ResNet-56 CIFAR-10 checkpoint from internet...")
                checkpoint_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt'
                try:
                    save = torch.hub.load_state_dict_from_url(
                        checkpoint_url, 
                        map_location='cuda:%d' % args.gpu,
                        progress=True,
                        check_hash=True
                    )
                    # بررسی فرمت state_dict
                    if isinstance(save, dict) and 'state_dict' in save:
                        ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
                    else:
                        ckpt = {k.replace('module.', ''): v for k, v in save.items()}
                    print("Checkpoint downloaded successfully!")
                except Exception as e:
                    print(f"Error downloading checkpoint: {e}")
                    print("Falling back to local file...")
                    save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
                    ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                # برای CIFAR-100 می‌توانید از این URL استفاده کنید
                print("Downloading ResNet-56 CIFAR-100 checkpoint from internet...")
                checkpoint_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt'
                try:
                    save = torch.hub.load_state_dict_from_url(
                        checkpoint_url, 
                        map_location='cuda:%d' % args.gpu,
                        progress=True,
                        check_hash=True
                    )
                    if isinstance(save, dict) and 'state_dict' in save:
                        ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
                    else:
                        ckpt = {k.replace('module.', ''): v for k, v in save.items()}
                    print("Checkpoint downloaded successfully!")
                except Exception as e:
                    print(f"Error downloading checkpoint: {e}")
                    print("Falling back to local file...")
                    ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet110':
        model = resnet110(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                # دانلود ResNet-110 از اینترنت
                print("Downloading ResNet-110 CIFAR-10 checkpoint from internet...")
                checkpoint_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet110-1d1ed7c2.pt'
                try:
                    save = torch.hub.load_state_dict_from_url(
                        checkpoint_url, 
                        map_location='cuda:%d' % args.gpu,
                        progress=True,
                        check_hash=True
                    )
                    if isinstance(save, dict) and 'state_dict' in save:
                        ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
                    else:
                        ckpt = {k.replace('module.', ''): v for k, v in save.items()}
                    print("Checkpoint downloaded successfully!")
                except Exception as e:
                    print(f"Error downloading checkpoint: {e}")
                    print("Falling back to local file...")
                    save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110.th', map_location='cuda:%d' % args.gpu)
                    ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                print("Downloading ResNet-110 CIFAR-100 checkpoint from internet...")
                checkpoint_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet110-c8a8dd84.pt'
                try:
                    save = torch.hub.load_state_dict_from_url(
                        checkpoint_url, 
                        map_location='cuda:%d' % args.gpu,
                        progress=True,
                        check_hash=True
                    )
                    if isinstance(save, dict) and 'state_dict' in save:
                        ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
                    else:
                        ckpt = {k.replace('module.', ''): v for k, v in save.items()}
                    print("Checkpoint downloaded successfully!")
                except Exception as e:
                    print(f"Error downloading checkpoint: {e}")
                    print("Falling back to local file...")
                    ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)

    # بارگذاری وزن‌های Teacher
    model.load_state_dict(ckpt, strict=False)
    model_s = set_gpu(args, model_s)
    model = set_gpu(args, model)

    # فریز کردن پارامترهای Teacher
    for param in model.parameters():
        param.requires_grad = False

    # چاپ وضعیت پارامترها (فقط برای debug)
    print("Teacher Model Parameters:")
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")

    print('-'*100)
    print("Student Model Parameters:")
    for name, param in model_s.named_parameters():
        print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
    
    model.eval()

    # تعریف loss ها
    divergence_loss = F.kl_div
    criterion = nn.CrossEntropyLoss().cuda()
    
    # ✅ انتخاب دیتاست صحیح بر اساس args.set
    if args.set == 'cifar10':
        data = CIFAR10()
        print("=" * 80)
        print("Using CIFAR-10 dataset (10 classes)")
        print("=" * 80)
    elif args.set == 'cifar100':
        data = CIFAR100()
        print("=" * 80)
        print("Using CIFAR-100 dataset (100 classes)")
        print("=" * 80)
    else:
        raise ValueError(f"Unknown dataset: {args.set}. Must be 'cifar10' or 'cifar100'")

    # اعتبارسنجی دقت Teacher
    acc1, acc5 = validate(data.val_loader, model, criterion, args)
    print(f"Teacher model accuracy: Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")

    # تنظیم optimizer
    optimizer = torch.optim.SGD(
        model_s.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    
    # تنظیم learning rate scheduler
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=lr_decay_step, 
        gamma=0.1
    )

    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    args.start_epoch = args.start_epoch or 0
    mask_list = []
    layer_num = []
    
    # شروع آموزش
    print("=" * 80)
    print("Starting Knowledge Distillation Training...")
    print("=" * 80)
    
    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        print(f"{'='*80}")
        
        train_acc1, train_acc5 = train_KD(
            data.train_loader, 
            model, 
            model_s, 
            divergence_loss, 
            criterion, 
            optimizer, 
            epoch, 
            args
        )
        
        acc1, acc5 = validate(data.val_loader, model_s, criterion, args)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train - Top-1: {train_acc1:.2f}%, Top-5: {train_acc5:.2f}%")
        print(f"  Val   - Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")
        
        scheduler.step()

        # ذخیره بهترین مدل
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                mask_list = []
                layer_num = []
                model_s.hook_masks()
                masks = model_s.get_masks()
                
                for key in masks.keys():
                    msk = ApproxSign(masks[key].mask).squeeze()
                    total = torch.sum(msk)
                    layer_num.append(int(total.cpu().detach().numpy()))
                    mask_list.append(msk)

                model_s.remove_hooks()
                
                logger.info(f"New Best Accuracy: {acc1:.2f}%")
                logger.info(f"Layer neurons: {layer_num}")
                
                print(f"\n{'*'*80}")
                print(f"✓ New Best Model! Accuracy: {acc1:.2f}%")
                print(f"  Active neurons per layer: {layer_num}")
                print(f"{'*'*80}\n")

                to = {'layer_num': layer_num, 'mask': mask_list}
                mask_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_T_{args.arch}_S_{args.arch_s}_mask.pt'
                model_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_{args.arch_s}.pt'
                
                torch.save(to, mask_path)
                torch.save(model_s.state_dict(), model_path)
                
                print(f"Saved mask to: {mask_path}")
                print(f"Saved model to: {model_path}")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation accuracy: {best_acc1:.2f}%")
    print("=" * 80)


def ApproxSign(mask):
    """تابع محاسبه علامت تقریبی برای ماسک‌ها"""
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

if __name__ == "__main__":
    # مثال استفاده:
    # CIFAR-10:
    # python train_kd.py --gpu 0 --arch resnet56 --set cifar10 --lr 0.01 --batch_size 128 --weight_decay 0.005 --epochs 160 --lr_decay_step 80,120 --num_classes 10 --pretrained --arch_s resnet20
    
    # CIFAR-100:
    # python train_kd.py --gpu 0 --arch resnet56 --set cifar100 --lr 0.01 --batch_size 128 --weight_decay 0.005 --epochs 160 --lr_decay_step 80,120 --num_classes 100 --pretrained --arch_s resnet20
    
    main()
