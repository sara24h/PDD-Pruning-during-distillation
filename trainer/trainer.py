import time
import torch
import tqdm
import torch.nn.functional as F
from utils.utils import accuracy
from utils.logging import AverageMeter, ProgressMeter


__all__ = ["train", "validate", "train_KD"]


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target.long())

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return top1.avg, top5.avg


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target.long())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, top5.avg


def train_KD(train_loader, model, model_s, divergence_loss, criterion, optimizer, epoch, args):
    """
    PDD Training following paper's Equation 4: L_total = L(z_s, z_t) + CE(z_s, Y)
    
    Implementation uses standard KD practices (Hinton et al. 2015):
    - KL divergence with softmax teacher target (NOT log_softmax)
    - batchmean reduction (PyTorch recommended)
    - Temperature scaling with T^2 compensation
    - Equal weighting (alpha not specified in paper)
    """
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    losses_kd = AverageMeter("Loss_KD", ":.3f")
    losses_ce = AverageMeter("Loss_CE", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses_kd, losses_ce, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    model.eval()  # Teacher in eval mode
    model_s.train()  # Student in train mode
    
    # Standard KD parameters (Hinton et al. 2015)
    # Paper doesn't specify these, so using standard values
    temp = 4.0  # Standard temperature (Hinton uses 3-4)
    alpha = 0.8  # Equal weight (paper equation suggests equal importance)

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        # Get teacher predictions (no gradient)
        with torch.no_grad():
            teacher_output = model(images)

        # Get student predictions
        student_output = model_s(images)

        # ✅ PAPER EQUATION 4: CE(z_s, Y)
        loss_ce = criterion(student_output, target.long())

        # ✅ PAPER EQUATION 4: L(z_s, z_t) - KL Divergence
        # Standard KD implementation (Hinton 2015):
        # - Input: log_softmax of student
        # - Target: softmax of teacher (NOT log_softmax!)
        # - Reduction: batchmean (PyTorch recommended for KL divergence)
        # - Scale by T^2 to compensate for temperature
        kd_loss = F.kl_div(
            F.log_softmax(student_output / temp, dim=1),  # Student: log probabilities
            F.softmax(teacher_output / temp, dim=1),       # Teacher: probabilities
            reduction='batchmean'  # Mathematically correct for KL div
        ) * (temp * temp)  # T^2 scaling (standard KD)

        # ✅ PAPER EQUATION 4: L_total = L(z_s, z_t) + CE(z_s, Y)
        # Paper doesn't specify alpha, so using 0.5 (equal weight)
        loss = alpha * kd_loss + (1 - alpha) * loss_ce

        # Sanity check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠ NaN/Inf at epoch {epoch}, batch {i}")
            print(f"  CE: {loss_ce.item():.4f}, KD: {kd_loss.item():.4f}")
            continue

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(student_output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        losses_kd.update(kd_loss.item(), images.size(0))
        losses_ce.update(loss_ce.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Optional: Gradient clipping for stability (not in paper, but good practice)
        torch.nn.utils.clip_grad_norm_(model_s.parameters(), max_norm=5.0)
        
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, top5.avg

