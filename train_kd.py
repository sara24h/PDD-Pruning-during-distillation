import os
import sys
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from args import args
import datetime
from data.Data import CIFAR10, CIFAR100
from model.VGG_cifar import cvgg16_bn, cvgg19_bn
# ✅ برای Teacher از resnet_kd استفاده کنید (همان معماری checkpoint)
from resnet_kd import resnet20, resnet56, resnet110
from trainer.trainer import validate, train, train_KD
from utils.utils import set_random_seed, set_gpu, Logger, get_logger, get_lr
from vgg_kd import cvgg11_bn
import torch.nn.functional as F


def load_teacher_checkpoint(args):

    ckpt = None
    
    if args.arch == 'cvgg16_bn':
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar10/scores.pt', 
                                map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar100/scores.pt', 
                                map_location='cuda:%d' % args.gpu)
                
    elif args.arch == 'cvgg19_bn':
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar10/scores.pt', 
                                map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar100/scores.pt', 
                                map_location='cuda:%d' % args.gpu)
                
    elif args.arch == 'resnet56':
        if args.pretrained:
            if args.set == 'cifar10':
                print("=" * 80)
                print("Downloading ResNet-56 CIFAR-10 checkpoint from internet...")
                print("=" * 80)
                checkpoint_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt'
                try:
                    ckpt = torch.hub.load_state_dict_from_url(
                        checkpoint_url, 
                        map_location='cuda:%d' % args.gpu,
                        progress=True,
                        check_hash=True
                    )
                    print("✓ Checkpoint downloaded successfully!")
                    # نیازی به تبدیل کلید ندارد - fc همان linear است
                    
                except Exception as e:
                    print(f"✗ Error downloading checkpoint: {e}")
                    raise
                        
            elif args.set == 'cifar100':
                print("=" * 80)
                print("Downloading ResNet-56 CIFAR-100 checkpoint from internet...")
                print("=" * 80)
                checkpoint_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt'
                try:
                    ckpt = torch.hub.load_state_dict_from_url(
                        checkpoint_url, 
                        map_location='cuda:%d' % args.gpu,
                        progress=True,
                        check_hash=True
                    )
                    print("✓ Checkpoint downloaded successfully!")
                    
                except Exception as e:
                    print(f"✗ Error downloading checkpoint: {e}")
                    raise
                    
    elif args.arch == 'resnet110':
        if args.pretrained:
            if args.set == 'cifar10':
                print("=" * 80)
                print("Downloading ResNet-110 CIFAR-10 checkpoint from internet...")
                print("=" * 80)
                checkpoint_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet110-1d1ed7c2.pt'
                try:
                    ckpt = torch.hub.load_state_dict_from_url(
                        checkpoint_url, 
                        map_location='cuda:%d' % args.gpu,
                        progress=True,
                        check_hash=True
                    )
                    print("✓ Checkpoint downloaded successfully!")
                    
                except Exception as e:
                    print(f"✗ Error downloading checkpoint: {e}")
                    raise
                    
            elif args.set == 'cifar100':
                print("=" * 80)
                print("Downloading ResNet-110 CIFAR-100 checkpoint from internet...")
                print("=" * 80)
                checkpoint_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet110-c8a8dd84.pt'
                try:
                    ckpt = torch.hub.load_state_dict_from_url(
                        checkpoint_url, 
                        map_location='cuda:%d' % args.gpu,
                        progress=True,
                        check_hash=True
                    )
                    print("✓ Checkpoint downloaded successfully!")
                    
                except Exception as e:
                    print(f"✗ Error downloading checkpoint: {e}")
                    raise
    
    return ckpt


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
    print("\n" + "=" * 80)
    print("Creating Student Model...")
    print("=" * 80)
    if args.arch_s == 'cvgg11_bn':
        model_s = cvgg11_bn(finding_masks=True, num_classes=args.num_classes, batch_norm=True)
    elif args.arch_s == 'resnet20':
        in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]
        out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
        # استفاده از option='B' برای سازگاری با checkpoint
        model_s = resnet20(finding_masks=True, in_cfg=in_cfg, out_cfg=out_cfg, 
                          num_classes=args.num_classes, option='B')
    print(f"✓ Student model created: {args.arch_s}")

    # ✅ ایجاد مدل Teacher با همان معماری checkpoint (option='B')
    print("\n" + "=" * 80)
    print("Creating Teacher Model...")
    print("=" * 80)
    if args.arch == 'cvgg16_bn':
        model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
    elif args.arch == 'cvgg19_bn':
        model = cvgg19_bn(num_classes=args.num_classes, batch_norm=True)
    elif args.arch == 'resnet56':
        # ✅ استفاده از option='B' برای مطابقت با checkpoint
        model = resnet56(num_classes=args.num_classes, option='B', finding_masks=False)
    elif args.arch == 'resnet110':
        # ✅ استفاده از option='B' برای مطابقت با checkpoint
        model = resnet110(num_classes=args.num_classes, option='B', finding_masks=False)
    print(f"✓ Teacher model created: {args.arch}")

    # بارگذاری checkpoint
    if args.pretrained:
        print("\n" + "=" * 80)
        print("Loading Teacher Checkpoint...")
        print("=" * 80)
        ckpt = load_teacher_checkpoint(args)
        
        if ckpt is not None:
            # ✅ تطبیق نام کلیدها (fc -> linear, downsample -> shortcut)
            new_ckpt = {}
            for key, value in ckpt.items():
                new_key = key
                # تبدیل fc به linear
                if key.startswith('fc.'):
                    new_key = key.replace('fc.', 'linear.')
                    print(f"  Renamed: {key} -> {new_key}")
                # تبدیل downsample به shortcut
                elif 'downsample' in key:
                    new_key = key.replace('downsample', 'shortcut')
                    print(f"  Renamed: {key} -> {new_key}")
                
                new_ckpt[new_key] = value
            
            # بارگذاری با strict=True (حالا باید کار کند)
            try:
                model.load_state_dict(new_ckpt, strict=True)
                print("✓ Checkpoint loaded successfully!")
            except RuntimeError as e:
                print(f"✗ Error loading checkpoint: {e}")
                print("\nAttempting non-strict loading...")
                missing_keys, unexpected_keys = model.load_state_dict(new_ckpt, strict=False)
                
                if missing_keys:
                    print(f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}")
                if unexpected_keys:
                    print(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}")
                
                # بررسی اینکه آیا فقط num_batches_tracked هستند
                important_missing = [k for k in missing_keys if 'num_batches_tracked' not in k]
                if not important_missing:
                    print("✓ Only batch norm tracking keys missing (this is OK)")
                else:
                    print(f"⚠ Warning: Important keys missing: {important_missing[:3]}")

    # انتقال مدل‌ها به GPU
    model_s = set_gpu(args, model_s)
    model = set_gpu(args, model)

    # فریز کردن پارامترهای Teacher
    print("\n" + "=" * 80)
    print("Freezing Teacher Model Parameters...")
    print("=" * 80)
    for param in model.parameters():
        param.requires_grad = False
    print("✓ All teacher parameters frozen")
    
    model.eval()

    # تعریف loss ها
    divergence_loss = F.kl_div
    criterion = nn.CrossEntropyLoss().cuda()
    
    # انتخاب دیتاست
    print("\n" + "=" * 80)
    print("Loading Dataset...")
    print("=" * 80)
    if args.set == 'cifar10':
        data = CIFAR10()
        print(f"✓ Using CIFAR-10 dataset ({args.num_classes} classes)")
    elif args.set == 'cifar100':
        data = CIFAR100()
        print(f"✓ Using CIFAR-100 dataset ({args.num_classes} classes)")
    else:
        raise ValueError(f"Unknown dataset: {args.set}. Must be 'cifar10' or 'cifar100'")

    # اعتبارسنجی دقت Teacher
    print("\n" + "=" * 80)
    print("Validating Teacher Model...")
    print("=" * 80)
    acc1, acc5 = validate(data.val_loader, model, criterion, args)
    print(f"Teacher Accuracy - Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")

    # بررسی دقت Teacher
    if acc1 < 80:  # ResNet-56 باید حدود 93% دقت داشته باشد
        print("\n" + "!" * 80)
        print("⚠ WARNING: Teacher model has lower than expected accuracy!")
        print(f"   Expected: ~93% for ResNet-56 on CIFAR-10")
        print(f"   Got: {acc1:.2f}%")
        print("This may indicate:")
        print("  1. Checkpoint loading issue")
        print("  2. Wrong dataset")
        print("  3. Model architecture mismatch")
        print("!" * 80)


    # تنظیم optimizer
    print("\n" + "=" * 80)
    print("Setting up Optimizer and Scheduler...")
    print("=" * 80)
    optimizer = torch.optim.SGD(
        model_s.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=lr_decay_step, 
        gamma=0.1
    )
    print(f"✓ Optimizer: SGD (lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay})")
    print(f"✓ Scheduler: MultiStepLR (milestones={lr_decay_step}, gamma=0.1)")

    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    args.start_epoch = args.start_epoch or 0
    mask_list = []
    layer_num = []
    
    # شروع آموزش
    print("\n" + "=" * 80)
    print("Starting Knowledge Distillation Training...")
    print(f"Total Epochs: {args.epochs}")
    print("=" * 80)
    
    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{args.epochs}] - LR: {get_lr(optimizer):.6f}")
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
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train - Top-1: {train_acc1:.2f}%, Top-5: {train_acc5:.2f}%")
        print(f"  Val   - Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")
        print(f"  Best  - Top-1: {best_acc1:.2f}%, Top-5: {best_acc5:.2f}%")
        
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
                print(f"🎉 New Best Model! Accuracy: {acc1:.2f}%")
                print(f"   Active neurons per layer: {layer_num}")
                print(f"{'*'*80}\n")

                to = {'layer_num': layer_num, 'mask': mask_list}
                mask_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_T_{args.arch}_S_{args.arch_s}_mask.pt'
                model_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_{args.arch_s}.pt'
                
                torch.save(to, mask_path)
                torch.save(model_s.state_dict(), model_path)
                
                print(f"✓ Saved mask to: {mask_path}")
                print(f"✓ Saved model to: {model_path}")

    print("\n" + "=" * 80)
    print("🎊 Training Completed Successfully!")
    print(f"Best Validation Accuracy: {best_acc1:.2f}%")
    print(f"Best Training Accuracy: {best_train_acc1:.2f}%")
    print("=" * 80)


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


if __name__ == "__main__":
    # مثال استفاده:
    # CIFAR-10:
    # python train_kd_fixed.py --gpu 0 --arch resnet56 --set cifar10 --lr 0.01 --batch_size 128 --weight_decay 0.005 --epochs 160 --lr_decay_step 80,120 --num_classes 10 --pretrained --arch_s resnet20
    
    # CIFAR-100:
    # python train_kd_fixed.py --gpu 0 --arch resnet56 --set cifar100 --lr 0.01 --batch_size 128 --weight_decay 0.005 --epochs 160 --lr_decay_step 80,120 --num_classes 100 --pretrained --arch_s resnet20
    
    main()
