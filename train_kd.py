import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from args import args
from data.Data import CIFAR10, CIFAR100
from resnet_kd_auto_prune import resnet20_auto, resnet56_auto
from trainer.trainer import validate, train_KD
from utils.utils import set_random_seed, set_gpu, Logger, get_logger


def ApproxSign(mask):
  
    out_forward = torch.sign(mask)
    mask1 = mask < -1
    mask2 = mask < 0
    mask3 = mask < 1
    out1 = (-1) * mask1.type(torch.float32) + (mask * mask + 2 * mask) * (1 - mask1.type(torch.float32))
    out2 = out1 * mask2.type(torch.float32) + (-mask * mask + 2 * mask) * (1 - mask2.type(torch.float32))
    out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
    out = out_forward.detach() - out3.detach() + out3
    out = (out + 1) / 2  # نرمال‌سازی به [0, 1]
    return out


def load_teacher_checkpoint(args):

    ckpt = None
    
    if args.arch == 'resnet56':
        if args.pretrained:
            if args.set == 'cifar10':
                print("="*80)
                print("Downloading ResNet-56 CIFAR-10 checkpoint...")
                print("="*80)
                checkpoint_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt'
                try:
                    ckpt = torch.hub.load_state_dict_from_url(
                        checkpoint_url, 
                        map_location=f'cuda:{args.gpu}',
                        progress=True,
                        check_hash=True
                    )
                    print("✓ Checkpoint downloaded successfully!")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    raise
                        
            elif args.set == 'cifar100':
                checkpoint_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt'
                ckpt = torch.hub.load_state_dict_from_url(
                    checkpoint_url, 
                    map_location=f'cuda:{args.gpu}',
                    progress=True
                )
    
    return ckpt


def main():
    print(args)
    sys.stdout = Logger('print_process.log', sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)


def main_worker(args):

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = f'pretrained_model/{args.arch}/{args.set}'
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(f'{log_dir}/logger_{now}.log')
    
    logger.info("="*80)
    logger.info("Automatic Pruning Configuration:")
    logger.info(f"  Teacher: {args.arch}")
    logger.info(f"  Student: {args.arch_s}")
    logger.info(f"  Dataset: {args.set}")
    logger.info(f"  Pruning: AUTOMATIC (no manual channel config needed)")
    logger.info(f"  Pruning Threshold: {args.pruning_threshold if hasattr(args, 'pruning_threshold') else 0.0}")
    logger.info("="*80)

    # ========================================================================
    # مرحله 1: ساخت مدل دانشجو با Pruning خودکار
    # ========================================================================
    print("\n" + "="*80)
    print("Creating Student Model with Automatic Pruning...")
    print("="*80)
    
    if args.arch_s == 'resnet20':
        # ✅ نیازی به in_cfg و out_cfg ندارید!
        model_s = resnet20_auto(
            num_classes=args.num_classes,
            option='B',
            use_pruning=True  # فعال کردن هرس خودکار
        )
        print("✓ Student model created: ResNet-20 (with automatic pruning)")
    else:
        raise ValueError(f"Unsupported student: {args.arch_s}")
    
    # ========================================================================
    # مرحله 2: ساخت مدل معلم
    # ========================================================================
    print("\n" + "="*80)
    print("Creating Teacher Model...")
    print("="*80)
    
    if args.arch == 'resnet56':
        model = resnet56_auto(
            num_classes=args.num_classes,
            option='B',
            use_pruning=False  # معلم نیازی به pruning ندارد
        )
        print("✓ Teacher model created: ResNet-56")
    else:
        raise ValueError(f"Unsupported teacher: {args.arch}")
    
    # ========================================================================
    # مرحله 3: بارگذاری checkpoint معلم
    # ========================================================================
    if args.pretrained:
        print("\n" + "="*80)
        print("Loading Teacher Checkpoint...")
        print("="*80)
        ckpt = load_teacher_checkpoint(args)
        
        if ckpt is not None:
            # تصحیح نام کلیدها
            new_ckpt = {}
            for key, value in ckpt.items():
                new_key = key
                if key.startswith('fc.'):
                    new_key = key.replace('fc.', 'linear.')
                elif 'downsample' in key:
                    new_key = key.replace('downsample', 'shortcut')
                new_ckpt[new_key] = value
            
            model.load_state_dict(new_ckpt, strict=False)
            print("✓ Teacher checkpoint loaded!")
    
    # انتقال به GPU
    model_s = set_gpu(args, model_s)
    model = set_gpu(args, model)
    
    # ========================================================================
    # مرحله 4: فریز کردن پارامترهای معلم
    # ========================================================================
    print("\n" + "="*80)
    print("Freezing Teacher Parameters...")
    print("="*80)
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("✓ Teacher frozen")
    
    # ========================================================================
    # مرحله 5: تعریف توابع Loss
    # ========================================================================
    criterion = nn.CrossEntropyLoss().cuda()
    divergence_loss = F.kl_div
    
    # ========================================================================
    # مرحله 6: بارگذاری دیتاست
    # ========================================================================
    print("\n" + "="*80)
    print("Loading Dataset...")
    print("="*80)
    
    if args.set == 'cifar10':
        data = CIFAR10()
    elif args.set == 'cifar100':
        data = CIFAR100()
    else:
        raise ValueError(f"Unknown dataset: {args.set}")
    print(f"✓ Dataset loaded: {args.set.upper()}")
    
    # ========================================================================
    # مرحله 7: اعتبارسنجی معلم
    # ========================================================================
    print("\n" + "="*80)
    print("Validating Teacher...")
    print("="*80)
    
    acc1, acc5 = validate(data.val_loader, model, criterion, args)
    print(f"Teacher Accuracy: Top-1={acc1:.2f}%, Top-5={acc5:.2f}%")
    logger.info(f"Teacher: Top-1={acc1:.2f}%, Top-5={acc5:.2f}%")
    
    # ========================================================================
    # مرحله 8: تنظیم Optimizer و Scheduler
    # ========================================================================
    print("\n" + "="*80)
    print("Setup Optimizer & Scheduler...")
    print("="*80)
    
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
    print(f"✓ SGD: lr={args.lr}, momentum={args.momentum}, wd={args.weight_decay}")
    print(f"✓ Scheduler: milestones={lr_decay_step}")
    
    # ========================================================================
    # مرحله 9: شروع آموزش
    # ========================================================================
    best_acc1 = 0.0
    best_acc5 = 0.0
    pruning_threshold = getattr(args, 'pruning_threshold', 0.0)
    
    print("\n" + "="*80)
    print(f"Starting Training with Automatic Pruning...")
    print(f"Total Epochs: {args.epochs}")
    print(f"Pruning Threshold: {pruning_threshold}")
    print("="*80)
    
    for epoch in range(args.start_epoch, args.epochs):
        print("\n" + "="*80)
        print(f"Epoch [{epoch+1}/{args.epochs}] - LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("="*80)
        
        # آموزش با distillation
        train_acc1, train_acc5 = train_KD(
            data.train_loader, 
            model,      # Teacher
            model_s,    # Student
            divergence_loss, 
            criterion, 
            optimizer, 
            epoch, 
            args
        )
        
        # اعتبارسنجی
        acc1, acc5 = validate(data.val_loader, model_s, criterion, args)
        
        # به‌روزرسانی learning rate
        scheduler.step()
        
        # به‌روزرسانی بهترین نتایج
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        
        # چاپ خلاصه
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train: Top-1={train_acc1:.2f}%, Top-5={train_acc5:.2f}%")
        print(f"  Val:   Top-1={acc1:.2f}%, Top-5={acc5:.2f}%")
        print(f"  Best:  Top-1={best_acc1:.2f}%, Top-5={best_acc5:.2f}%")
        logger.info(f"Epoch {epoch+1}: Train={train_acc1:.2f}%, Val={acc1:.2f}%")
        
        # ====================================================================
        # نمایش آمار Pruning هر 10 epoch
        # ====================================================================
        if (epoch + 1) % 10 == 0 or is_best:
            print("\n" + "-"*80)
            print("Current Pruning Statistics:")
            print("-"*80)
            model_s.print_pruning_stats(threshold=pruning_threshold)
        
        # ====================================================================
        # ذخیره بهترین مدل
        # ====================================================================
        if is_best:
            print(f"\n{'*'*80}")
            print(f"🎉 New Best Model! Accuracy: {acc1:.2f}%")
            
            # استخراج معماری هرس‌شده
            arch_config = model_s.extract_pruned_architecture(threshold=pruning_threshold)
            print(f"\nPruned Architecture (threshold={pruning_threshold}):")
            print(f"  in_cfg:  {arch_config['in_cfg']}")
            print(f"  out_cfg: {arch_config['out_cfg']}")
            print(f"{'*'*80}\n")
            
            # ذخیره checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model_s.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
                'arch_config': arch_config,
                'threshold': pruning_threshold
            }
            
            save_path = f'{log_dir}/best_model.pth'
            torch.save(checkpoint, save_path)
            print(f"✓ Best model saved to: {save_path}")
            
            logger.info(f"New best: Epoch={epoch+1}, Acc={acc1:.2f}%")
            logger.info(f"Architecture: in_cfg={arch_config['in_cfg']}")
            logger.info(f"Architecture: out_cfg={arch_config['out_cfg']}")
    
    # ========================================================================
    # مرحله 10: اعمال هرس نهایی
    # ========================================================================
    print("\n" + "="*80)
    print("Applying Final Pruning...")
    print("="*80)
    
    model_s.apply_pruning(threshold=pruning_threshold)
    
    # اعتبارسنجی نهایی
    final_acc1, final_acc5 = validate(data.val_loader, model_s, criterion, args)
    
    print("\n" + "="*80)
    print("🎊 Training Completed!")
    print(f"Best Validation Accuracy: {best_acc1:.2f}%")
    print(f"Final Accuracy (after pruning): {final_acc1:.2f}%")
    print("="*80)
    
    logger.info("="*80)
    logger.info("Training Completed!")
    logger.info(f"Best Accuracy: {best_acc1:.2f}%")
    logger.info(f"Final Accuracy: {final_acc1:.2f}%")
    logger.info("="*80)
    
    # ذخیره مدل نهایی
    final_checkpoint = {
        'state_dict': model_s.state_dict(),
        'accuracy': final_acc1,
        'arch_config': model_s.extract_pruned_architecture(threshold=pruning_threshold),
        'threshold': pruning_threshold
    }
    
    final_path = f'{log_dir}/final_pruned_model.pth'
    torch.save(final_checkpoint, final_path)
    print(f"\n✓ Final pruned model saved to: {final_path}")


if __name__ == "__main__":
   
    main()
