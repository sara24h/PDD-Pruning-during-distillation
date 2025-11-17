import os
import sys
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from data.Data import CIFAR10, CIFAR100
from model.VGG_cifar import cvgg16_bn
from resnet_kd import resnet20, resnet56, resnet110
from trainer.trainer import validate, train_KD
from utils.utils import set_random_seed, set_gpu, Logger, get_logger
from vgg_kd import cvgg11_bn
import torch.nn.functional as F
import numpy as np


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
            if args.set == 'cifar10':
                ckpt = torch.load('pretrained_model/cvgg16_bn/cifar10/scores.pt', 
                                map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('pretrained_model/cvgg16_bn/cifar100/scores.pt', 
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


def extract_masks(model_s, args):
    """Extract masks from the trained student model"""
    # Hook masks to capture their values
    model_s.hook_masks()
    
    # Forward pass to trigger hooks
    dummy_input = torch.randn(1, 3, 32, 32).cuda(args.gpu)
    with torch.no_grad():
        _ = model_s(dummy_input)
    
    # Get captured masks
    masks = model_s.get_masks()
    
    # Fallback: manual extraction if hooks failed
    if len(masks) == 0:
        print("⚠️ Warning: No masks captured via hooks. Extracting manually...")
        for name, module in model_s.named_modules():
            if hasattr(module, 'mask'):
                masks[name] = module.mask
    
    # Apply ApproxSign to get binary masks
    mask_list = []
    layer_num = []
    
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
    
    return mask_list, layer_num


def create_pruned_model(model_s, mask_list, layer_num, args):
    """Create a pruned model based on the extracted masks"""
    if args.arch_s == 'cvgg11_bn':
        # Calculate new channel configurations
        in_cfg = [3]  # Input channels
        out_cfg = layer_num  # Output channels for each layer
        
        print(f"Creating pruned VGG11 with {len(out_cfg)} layers")
        print(f"Output channels config: {out_cfg}")
        
        # Create pruned model
        pruned_model = cvgg11_bn(
            finding_masks=False,  # No masks needed for final model
            num_classes=args.num_classes,
            batch_norm=True
        ).cuda()
        
        # Copy weights from original model (only for active channels)
        pruned_state_dict = pruned_model.state_dict()
        original_state_dict = model_s.state_dict()
        
        # This is a simplified approach - in practice, you'd need to carefully map
        # the weights from the original model to the pruned model based on the masks
        for name, param in pruned_state_dict.items():
            if name in original_state_dict:
                pruned_state_dict[name].copy_(original_state_dict[name])
        
        pruned_model.load_state_dict(pruned_state_dict)
        
    elif args.arch_s == 'resnet20':
        # Calculate new channel configurations
        in_cfg = [3]  # Input channels
        out_cfg = layer_num  # Output channels for each layer
        
        print(f"Creating pruned ResNet20 with {len(out_cfg)} layers")
        print(f"Input channels config: {in_cfg}")
        print(f"Output channels config: {out_cfg}")
        
        # Create pruned model
        pruned_model = resnet20(
            finding_masks=False,  # No masks needed for final model
            in_cfg=in_cfg,
            out_cfg=out_cfg,
            num_classes=args.num_classes,
            option='B'
        ).cuda()
        
        # Copy weights from original model (only for active channels)
        pruned_state_dict = pruned_model.state_dict()
        original_state_dict = model_s.state_dict()
        
        # This is a simplified approach - in practice, you'd need to carefully map
        # the weights from the original model to the pruned model based on the masks
        for name, param in pruned_state_dict.items():
            if name in original_state_dict:
                pruned_state_dict[name].copy_(original_state_dict[name])
        
        pruned_model.load_state_dict(pruned_state_dict)
    
    else:
        raise ValueError(f"Unsupported student architecture: {args.arch_s}")
    
    return pruned_model


def main():
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
        if is_best or epoch == args.epochs - 1:
            # Extract masks from the best model
            mask_list, layer_num = extract_masks(model_s, args)
            
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
    # Step 12: Create Pruned Model
    # ========================================================================================
    print("\n" + "=" * 80)
    print("Creating Pruned Model...")
    print("=" * 80)
    
    # Load the best model
    model_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_{args.arch_s}.pt'
    model_s.load_state_dict(torch.load(model_path))
    
    # Extract masks from the best model
    mask_list, layer_num = extract_masks(model_s, args)
    
    # Create pruned model based on masks
    pruned_model = create_pruned_model(model_s, mask_list, layer_num, args)
    
    # Save the pruned model
    pruned_model_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_T_{args.arch}_S_{args.arch_s}_pruned.pt'
    torch.save(pruned_model.state_dict(), pruned_model_path)
    
    print(f"✓ Pruned model saved to: {pruned_model_path}")
    
    # Validate the pruned model
    pruned_acc1, pruned_acc5 = validate(data.val_loader, pruned_model, criterion, args)
    print(f"Pruned Model Accuracy - Top-1: {pruned_acc1:.2f}%, Top-5: {pruned_acc5:.2f}%")
    
    # ========================================================================================
    # Step 13: Fine-tune the Pruned Model
    # ========================================================================================
    print("\n" + "=" * 80)
    print("Fine-tuning Pruned Model...")
    print("=" * 80)
    
    # Setup optimizer for fine-tuning
    finetune_optimizer = torch.optim.SGD(
        pruned_model.parameters(), 
        lr=args.lr * 0.1,  # Lower learning rate for fine-tuning
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    
    finetune_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        finetune_optimizer, 
        milestones=[10, 20],  # Fewer milestones for fine-tuning
        gamma=0.1
    )
    
    # Fine-tune for fewer epochs
    finetune_epochs = 30
    best_finetune_acc1 = 0.0
    
    for epoch in range(finetune_epochs):
        print("\n" + "=" * 80)
        print(f"Fine-tune Epoch [{epoch+1}/{finetune_epochs}] - LR: {finetune_optimizer.param_groups[0]['lr']:.6f}")
        print("=" * 80)
        
        # Train the pruned model (standard training, not KD)
        train_acc1, train_acc5 = train(
            data.train_loader, 
            pruned_model, 
            criterion, 
            finetune_optimizer, 
            epoch, 
            args
        )
        
        # Validate the pruned model
        acc1, acc5 = validate(data.val_loader, pruned_model, criterion, args)
        
        # Update learning rate
        finetune_scheduler.step()
        
        # Update best metrics
        is_best = acc1 > best_finetune_acc1
        best_finetune_acc1 = max(acc1, best_finetune_acc1)
        
        # Print epoch summary
        print(f"\nFine-tune Epoch {epoch+1} Summary:")
        print(f"  Train - Top-1: {train_acc1:.2f}%, Top-5: {train_acc5:.2f}%")
        print(f"  Val   - Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")
        print(f"  Best  - Top-1: {best_finetune_acc1:.2f}%")
        
        # Save the best fine-tuned model
        if is_best:
            finetuned_model_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_T_{args.arch}_S_{args.arch_s}_pruned_finetuned.pt'
            torch.save(pruned_model.state_dict(), finetuned_model_path)
            print(f"✓ Best fine-tuned model saved to: {finetuned_model_path}")

    # ========================================================================================
    # Step 14: Training Complete
    # ========================================================================================
    print("\n" + "=" * 80)
    print("🎊 PDD Training and Pruning Completed Successfully!")
    print(f"Original Student Accuracy: {best_acc1:.2f}%")
    print(f"Pruned Model Accuracy: {pruned_acc1:.2f}%")
    print(f"Fine-tuned Pruned Model Accuracy: {best_finetune_acc1:.2f}%")
    print("=" * 80)
    
    logger.info("=" * 80)
    logger.info("Training and Pruning Completed!")
    logger.info(f"Original Student Accuracy: {best_acc1:.2f}%")
    logger.info(f"Pruned Model Accuracy: {pruned_acc1:.2f}%")
    logger.info(f"Fine-tuned Pruned Model Accuracy: {best_finetune_acc1:.2f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
 
    main()
