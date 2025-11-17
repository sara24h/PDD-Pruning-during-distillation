import os
import sys
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
import argparse  # <<<< اضافه شد
import datetime  # <<<< اضافه شد
from data.Data import CIFAR10, CIFAR100
from model.VGG_cifar import cvgg16_bn
from resnet_kd import resnet20, resnet56, resnet110
from trainer.trainer import validate, train_KD
from utils.utils import set_random_seed, set_gpu, Logger, get_logger
from vgg_kd import cvgg11_bn
import torch.nn.functional as F


# ================================================================================
# بخش اضافه شده برای تعریف آرگومان‌های خط فرمان
# ================================================================================
parser = argparse.ArgumentParser(description='PDD: Pruning During Knowledge Distillation')
parser.add_argument("--gpu", default=0, type=int, help="Which GPU to use")
parser.add_argument("--arch", default='resnet56', type=str, help="Teacher architecture")
parser.add_argument("--arch_s", default='resnet20', type=str, help="Student architecture")
parser.add_argument("--set", default='cifar10', type=str, help="Dataset name (cifar10, cifar100)")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay")
parser.add_argument("--epochs", default=50, type=int, help="Number of total epochs to run")
parser.add_argument("--lr_decay_step", default='20,40', type=str, help="Epochs to decay learning rate")
parser.add_argument("--num_classes", default=10, type=int, help="Number of classes in the dataset")
parser.add_argument("--pretrained", action="store_true", help="Use pre-trained teacher model")
parser.add_argument("--start_epoch", default=0, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--save_every", default=0, type=int, help="Save checkpoint every N epochs (0 to disable)")
parser.add_argument("--random_seed", default=None, type=int, help="Seed for random number generators")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD")
parser.add_argument("--print_freq", default=100, type=int, help="Print frequency")

# پارس کردن آرگومان‌ها
args = parser.parse_args()
# ================================================================================


def load_teacher_checkpoint(args):
    """Load pretrained teacher checkpoint from PyTorch Hub or local path"""
    ckpt = None
    
    if args.arch == 'resnet56':
        if args.pretrained:
            if args.set == 'cifar10':
                checkpoint_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt'
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
    
    elif args.arch == 'cvgg16_bn':
        if args.pretrained:
            # این مسیر را با مسیر فایل واقعی جایگزین کنید
            if args.set == 'cifar10':
                ckpt_path = 'pretrained_model/cvgg16_bn/cifar10/scores.pt'
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location='cuda:%d' % args.gpu)
                else:
                    print(f"⚠️ Teacher checkpoint not found at {ckpt_path}. Please download it manually.")
                    ckpt = None
            elif args.set == 'cifar100':
                ckpt_path = 'pretrained_model/cvgg16_bn/cifar100/scores.pt'
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location='cuda:%d' % args.gpu)
                else:
                    print(f"⚠️ Teacher checkpoint not found at {ckpt_path}. Please download it manually.")
                    ckpt = None
    
    return ckpt


def ApproxSign(mask):
    """
    Differentiable approximation of sign function (Equation 2 in paper)
    Converts continuous mask values to binary {0, 1}
    """
    out_forward = torch.sign(mask)
    mask1 = mask < -1
    mask2 = mask < 0
    mask3 = mask < 1
    out1 = (-1) * mask1.type(torch.float32) + (mask * mask + 2 * mask) * (1 - mask1.type(torch.float32))
    out2 = out1 * mask2.type(torch.float32) + (-mask * mask + 2 * mask) * (1 - mask2.type(torch.float32))
    out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
    out = out_forward.detach() - out3.detach() + out3
    out = (out + 1) / 2  # Normalize to [0, 1]
    return out


def main(args):  # <<<< اصلاح شد: args به عنوان ورودی گرفته می‌شود
    print(args)
    sys.stdout = Logger('print_process.log', sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)


def main_worker(args):
    """Main training loop for PDD (Pruning During Distillation)"""
    
    # Setup logging
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('pretrained_model/' + args.arch + '/' + args.set):
        os.makedirs('pretrained_model/' + args.arch + '/' + args.set, exist_ok=True)
    logger = get_logger('pretrained_model/' + args.arch + '/' + args.set + '/logger' + now + '.log')
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("PDD Configuration:")
    logger.info(f"  Teacher: {args.arch}")
    logger.info(f"  Student: {args.arch_s}")
    logger.info(f"  Dataset: {args.set}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  LR decay steps: {args.lr_decay_step}")
    logger.info(f"  Num classes: {args.num_classes}")
    logger.info("=" * 80)

    # ========================================================================================
    # Step 1: Create Student Model with Dynamic Masks (PDD)
    # ========================================================================================
    print("\n" + "=" * 80)
    print("Creating Student Model with Dynamic Masks...")
    print("=" * 80)
    
    if args.arch_s == 'cvgg11_bn':
        model_s = cvgg11_bn(finding_masks=True, num_classes=args.num_classes, batch_norm=True)
    elif args.arch_s == 'resnet20':
        in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]
        out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
        model_s = resnet20(finding_masks=True, in_cfg=in_cfg, out_cfg=out_cfg, 
                          num_classes=args.num_classes, option='B')
    else:
        raise ValueError(f"Unsupported student architecture: {args.arch_s}")
    
    print(f"✓ Student model created: {args.arch_s} (with differentiable masks)")

    # ========================================================================================
    # Step 2: Create Teacher Model
    # ========================================================================================
    print("\n" + "=" * 80)
    print("Creating Teacher Model...")
    print("=" * 80)
    
    if args.arch == 'cvgg16_bn':
        model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
    elif args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes, option='B', finding_masks=False)
    elif args.arch == 'resnet110':
        model = resnet110(num_classes=args.num_classes, option='B', finding_masks=False)
    else:
        raise ValueError(f"Unsupported teacher architecture: {args.arch}")
    
    print(f"✓ Teacher model created: {args.arch}")

    # ========================================================================================
    # Step 3: Load Teacher Checkpoint
    # ========================================================================================
    if args.pretrained:
        print("\n" + "=" * 80)
        print("Loading Teacher Checkpoint...")
        print("=" * 80)
        ckpt = load_teacher_checkpoint(args)
        
        if ckpt is not None:
            # Fix key names (fc -> linear, downsample -> shortcut)
            new_ckpt = {}
            for key, value in ckpt.items():
                new_key = key
                if key.startswith('fc.'):
                    new_key = key.replace('fc.', 'linear.')
                elif 'downsample' in key:
                    new_key = key.replace('downsample', 'shortcut')
                
                new_ckpt[new_key] = value
            
            try:
                model.load_state_dict(new_ckpt, strict=True)
                print("✓ Checkpoint loaded successfully!")
            except RuntimeError as e:
                print(f"✗ Error loading checkpoint: {e}")
                print("Attempting non-strict loading...")
                missing_keys, unexpected_keys = model.load_state_dict(new_ckpt, strict=False)
                
                if missing_keys:
                    print(f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
                if unexpected_keys:
                    print(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")

    # Move models to GPU
    model_s = set_gpu(args, model_s)
    model = set_gpu(args, model)

    # ========================================================================================
    # Step 4: Freeze Teacher Parameters
    # ========================================================================================
    print("\n" + "=" * 80)
    print("Freezing Teacher Model Parameters...")
    print("=" * 80)
    
    for param in model.parameters():
        param.requires_grad = False
    print("✓ All teacher parameters frozen")
    
    model.eval()  # Teacher always in eval mode

    # ========================================================================================
    # Step 5: Define Loss Functions
    # ========================================================================================
    criterion = nn.CrossEntropyLoss().cuda()
    divergence_loss = F.kl_div  # Used in train_KD for L(z_s, z_t)
    
    # ========================================================================================
    # Step 6: Load Dataset
    # ========================================================================================
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
        raise ValueError(f"Unknown dataset: {args.set}")

    # ========================================================================================
    # Step 7: Validate Teacher Accuracy
    # ========================================================================================
    print("\n" + "=" * 80)
    print("Validating Teacher Model...")
    print("=" * 80)
    
    acc1, acc5 = validate(data.val_loader, model, criterion, args)
    print(f"Teacher Accuracy - Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")
    logger.info(f"Teacher Accuracy - Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")

    # ========================================================================================
    # Step 8: Setup Optimizer and Scheduler
    # ========================================================================================
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

    # ========================================================================================
    # Step 9: Initialize Training Variables
    # ========================================================================================
    best_acc1 = 0.0
    args.start_epoch = args.start_epoch or 0
    
    # ========================================================================================
    # Step 10: Start PDD Training Loop
    # ========================================================================================
    print("\n" + "=" * 80)
    print("Starting PDD Training (Pruning During Distillation)...")
    print(f"Total Epochs: {args.epochs}")
    print("=" * 80)
    
    for epoch in range(args.start_epoch, args.epochs):
        print("\n" + "=" * 80)
        print(f"Epoch [{epoch+1}/{args.epochs}] - LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("=" * 80)
        
        # Train with knowledge distillation
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
        
        # Validate student model
        acc1, acc5 = validate(data.val_loader, model_s, criterion, args)
        
        # Update learning rate
        scheduler.step()

        # Update best metrics
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train - Top-1: {train_acc1:.2f}%, Top-5: {train_acc5:.2f}%")
        print(f"  Val   - Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")
        print(f"  Best  - Top-1: {best_acc1:.2f}%")
        
        logger.info(f"Epoch {epoch+1}: Train={train_acc1:.2f}%, Val={acc1:.2f}%, Best={best_acc1:.2f}%")

        # ========================================================================================
        # Step 11: Save Best Model and Extract Masks
        # ========================================================================================
        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                mask_list = []
                layer_num = []
                
                # Hook masks to capture their values
                model_s.hook_masks()
                
                # Forward pass to trigger hooks (CRITICAL!)
                dummy_input = torch.randn(1, 3, 32, 32).cuda(args.gpu)
                with torch.no_grad():  # No gradients needed for mask extraction
                    _ = model_s(dummy_input)
                
                # Get captured masks
                masks = model_s.get_masks()
                
                # Fallback: manual extraction if hooks failed
                if len(masks) == 0:
                    print("⚠️ Warning: No masks captured via hooks. Extracting manually...")
                    for name, module in model_s.named_modules():
                        if hasattr(module, 'mask'):
                            masks[name] = module.mask
                
                # Apply ApproxSign to get binary masks (Equation 2 in paper)
                for key in sorted(masks.keys()):  # Sort for consistency
                    mask_param = masks[key]
                    
                    # Handle both DynamicMask objects and direct tensors
                    if hasattr(mask_param, 'mask'):  # DynamicMask object
                        mask_tensor = mask_param.mask
                    else:  # Direct tensor
                        mask_tensor = mask_param
                    
                    # Convert continuous mask to binary {0, 1}
                    msk = ApproxSign(mask_tensor).squeeze()
                    total = torch.sum(msk)
                    layer_num.append(int(total.cpu().detach().numpy()))
                    mask_list.append(msk)

                # Remove hooks
                model_s.remove_hooks()
                
                # Log results
                logger.info(f"New best at epoch {epoch+1}: Accuracy = {acc1:.2f}%")
                logger.info(f"Active neurons per layer: {layer_num}")
                
                print(f"\n{'*' * 80}")
                print(f"🎉 New Best Model! Accuracy: {acc1:.2f}%")
                print(f"   Active neurons per layer: {layer_num}")
                print(f"{'*' * 80}\n")

                # Save mask and model state
                mask_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_T_{args.arch}_S_{args.arch_s}_mask.pt'
                model_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_{args.arch_s}.pt'
                
                to = {'layer_num': layer_num, 'mask': mask_list}
                torch.save(to, mask_path)
                torch.save(model_s.state_dict(), model_path)
                
                print(f"✓ Saved mask to: {mask_path}")
                print(f"✓ Saved model to: {model_path}")

    # ========================================================================================
    # Step 12: Training Complete
    # ========================================================================================
    print("\n" + "=" * 80)
    print("🎊 PDD Training Completed Successfully!")
    print(f"Best Validation Accuracy: {best_acc1:.2f}%")
    print("=" * 80)
    
    logger.info("=" * 80)
    logger.info("Training Completed!")
    logger.info(f"Best Validation Accuracy: {best_acc1:.2f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    main(args)  # <<<< اصلاح شد: args به تابع main پاس داده می‌شود
