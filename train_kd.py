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
from resnet_kd import resnet20, resnet56
from trainer.trainer import validate, train, train_KD
from utils.utils import set_random_seed, set_gpu, Logger, get_logger, get_lr
from vgg_kd import cvgg11_bn
import torch.nn.functional as F


def load_teacher_checkpoint(args):
    """Load pretrained teacher checkpoint from PyTorch Hub or local path"""
    ckpt = None
    
    if args.arch == 'resnet56':
        if args.pretrained:
            if args.set == 'cifar10':
                print("=" * 80)
                print("Downloading ResNet-56 CIFAR-10 checkpoint...")
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
                except Exception as e:
                    print(f"✗ Error downloading checkpoint: {e}")
                    raise
                        
            elif args.set == 'cifar100':
                print("=" * 80)
                print("Downloading ResNet-56 CIFAR-100 checkpoint...")
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
                print("Downloading ResNet-110 CIFAR-10 checkpoint...")
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
                print("Downloading ResNet-110 CIFAR-100 checkpoint...")
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
    
    elif args.arch == 'cvgg16_bn':
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar10/scores.pt', 
                                map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar100/scores.pt', 
                                map_location='cuda:%d' % args.gpu)
    
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

def copy_weights(model_s, pruned_model, kept_indices, args):
    device = f'cuda:{args.gpu}'
    state_dict_s = model_s.state_dict()
    state_dict_p = pruned_model.state_dict()

    # conv1 + bn1 (همیشه کامل کپی میشه)
    for k in ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var']:
        state_dict_p[k].copy_(state_dict_s[k])

    block_idx = 0
    in_channels = 16  # بعد از conv1 همیشه 16 هست

    for layer_name in ['layer1', 'layer2', 'layer3']:
        num_blocks = 3  # برای resnet20 همیشه 3 تا بلاک در هر لایه
        for b in range(num_blocks):
            prefix = f'{layer_name}.{b}.'

            # conv1 + bn1 (ورودی از لایه قبلی)
            state_dict_p[prefix + 'conv1.weight'].copy_(
                state_dict_s[prefix + 'conv1.weight'][:, :in_channels, :, :]
            )
            for bn_k in ['weight', 'bias', 'running_mean', 'running_var']:
                state_dict_p[prefix + f'bn1.{bn_k}'].copy_(
                    state_dict_s[prefix + f'bn1.{bn_k}']
                )

            # conv2 + bn2 (اینجا پرون میشه)
            out_channels = kept_indices[block_idx].shape[0]
            state_dict_p[prefix + 'conv2.weight'].copy_(
                state_dict_s[prefix + 'conv2.weight'][kept_indices[block_idx]][:, :in_channels, :, :]
            )
            for bn_k in ['weight', 'bias', 'running_mean', 'running_var']:
                state_dict_p[prefix + f'bn2.{bn_k}'].copy_(
                    state_dict_s[prefix + f'bn2.{bn_k}'][kept_indices[block_idx]]
                )

            # Shortcut — مهم‌ترین قسمت! (درست شده)
            if prefix + 'shortcut.0.weight' in state_dict_p:
                state_dict_p[prefix + 'shortcut.0.weight'].copy_(
                    state_dict_s[prefix + 'shortcut.0.weight'][kept_indices[block_idx]][:, :in_channels, :, :]
                )
                for bn_k in ['weight', 'bias', 'running_mean', 'running_var']:
                    state_dict_p[prefix + f'shortcut.1.{bn_k}'].copy_(
                        state_dict_s[prefix + f'shortcut.1.{bn_k}'][kept_indices[block_idx]]
                    )

            # آپدیت برای بلاک بعدی
            in_channels = out_channels
            block_idx += 1

    # لایه آخر
    state_dict_p['linear.weight'].copy_(state_dict_s['linear.weight'][:, :in_channels])
    state_dict_p['linear.bias'].copy_(state_dict_s['linear.bias'])

    pruned_model.load_state_dict(state_dict_p)
    return pruned_model


def main():
    print(args)
    sys.stdout = Logger('print process.log', sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)


def main_worker(args):
    """Main training loop for PDD (Pruning During Distillation)"""
    
    # Setup logging
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-M-S')
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
    logger.info(f"  Epochs: {args.epochs} (Paper uses 50 for distillation)")
    logger.info(f"  LR decay steps: {args.lr_decay_step} (Paper: 20,40)")
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
        model_s = resnet20(finding_masks=True, num_classes=args.num_classes, option='B')
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
    elif args.arch == 'cvgg19_bn':
        model = cvgg19_bn(num_classes=args.num_classes, batch_norm=True)
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
                    print(f"  Renamed: {key} -> {new_key}")
                elif 'downsample' in key:
                    new_key = key.replace('downsample', 'shortcut')
                    print(f"  Renamed: {key} -> {new_key}")
                
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
    # Step 4: Freeze Teacher Parameters (Paper Requirement)
    # ========================================================================================
    print("\n" + "=" * 80)
    print("Freezing Teacher Model Parameters...")
    print("=" * 80)
    
    for param in model.parameters():
        param.requires_grad = False
    print("✓ All teacher parameters frozen")
    
    model.eval()  # Teacher always in eval mode

    # ========================================================================================
    # Step 5: Define Loss Functions (According to Paper Equation 4)
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

    # Check teacher accuracy
    expected_acc = 93.0 if args.set == 'cifar10' else 70.0
    if acc1 < expected_acc - 5:
        print("\n" + "!" * 80)
        print("⚠ WARNING: Teacher model has lower than expected accuracy!")
        print(f"   Expected: ~{expected_acc}% for {args.arch} on {args.set.upper()}")
        print(f"   Got: {acc1:.2f}%")
        print("!" * 80)

    # ========================================================================================
    # Step 8: Setup Optimizer and Scheduler (Paper Parameters)
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
    
    # Paper specifies: decay at epochs 20 and 40 for distillation
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
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    args.start_epoch = args.start_epoch or 0
    mask_list = []
    layer_num = []
    
    # ========================================================================================
    # Step 10: Start PDD Training Loop (Paper: 50 epochs)
    # ========================================================================================
    print("\n" + "=" * 80)
    print("Starting PDD Training (Pruning During Distillation)...")
    print(f"Total Epochs: {args.epochs} (Paper recommends 50)")
    print("=" * 80)
    
    for epoch in range(args.start_epoch, args.epochs):
        print("\n" + "=" * 80)
        print(f"Epoch [{epoch+1}/{args.epochs}] - LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("=" * 80)
        
        # Train with knowledge distillation (Equation 4 in paper)
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
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train - Top-1: {train_acc1:.2f}%, Top-5: {train_acc5:.2f}%")
        print(f"  Val   - Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")
        print(f"  Best  - Top-1: {best_acc1:.2f}%, Top-5: {best_acc5:.2f}%")
        
        logger.info(f"Epoch {epoch+1}: Train={train_acc1:.2f}%, Val={acc1:.2f}%, Best={best_acc1:.2f}%")

        # ========================================================================================
        # Step 11: Save Best Model and Extract Masks (Paper Section 4)
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
    # Step 12: Pruning and Finetuning the Pruned Model
    # ========================================================================================
    print("\n" + "=" * 80)
    print("Performing Automatic Pruning and Finetuning...")
    print("=" * 80)

    # Compute kept_indices from mask_list
    kept_indices = [torch.where(msk == 1)[0].to('cuda:' + str(args.gpu)) for msk in mask_list]

    # Compute pruned_out_cfg
    pruned_out_cfg = [16] + layer_num + [layer_num[-1]]  # adjust if needed, but since out_cfg has 10, conv1 +9 + last = layer_num[-1]

    # The original out_cfg is [16,16,16,32,32,32,64,64,64,64]

    pruned_out_cfg = [16] + layer_num

    # Create pruned model
    pruned_model = resnet20(finding_masks=False, num_classes=args.num_classes, option='B')
    pruned_model = set_gpu(args, pruned_model)

    # Copy selected weights
    pruned_model = copy_weights(model_s, pruned_model, kept_indices, args)

    # Setup optimizer for finetune
    optimizer_p = torch.optim.SGD(
        pruned_model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )

    # Assume finetune for 150 epochs, decay at 60,90
    finetune_epochs = 150
    scheduler_p = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_p, 
        milestones=[60,90], 
        gamma=0.1
    )

    # Finetune loop
    best_acc1 = 0.0
    for epoch in range(finetune_epochs):
        train_acc1, train_acc5 = train(data.train_loader, pruned_model, criterion, optimizer_p, epoch, args)

        acc1, acc5 = validate(data.val_loader, pruned_model, criterion, args)

        scheduler_p.step()

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            torch.save(pruned_model.state_dict(), f'pretrained_model/{args.arch}/{args.set}/{args.set}_{args.arch_s}_pruned.pt')

    print("\n" + "=" * 80)
    print("🎊 PDD Training and Pruning Completed Successfully!")
    print(f"Best Validation Accuracy after Finetuning: {best_acc1:.2f}%")
    print("=" * 80)
    
    logger.info("=" * 80)
    logger.info("Training and Pruning Completed!")
    logger.info(f"Best Validation Accuracy after Finetuning: {best_acc1:.2f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Example usage:
    # python train_kd.py --gpu 0 --arch resnet56 --arch_s resnet20 --set cifar10 \
    #   --lr 0.01 --batch_size 256 --weight_decay 0.005 --epochs 50 \
    #   --lr_decay_step 20,40 --num_classes 10 --pretrained
    main()



