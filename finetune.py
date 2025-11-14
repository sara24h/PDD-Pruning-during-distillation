import datetime
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet_kd import resnet20, resnet56
from data.Data import CIFAR10, CIFAR100
from trainer.trainer import validate, train
from utils.utils import set_gpu, set_random_seed

# غیرفعال کردن هشدارها
import warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class CustomLogger:
    def __init__(self, log_file, console=sys.stdout):
        self.console = console
        self.log = open(log_file, 'w', encoding='utf-8')
    def write(self, message):
        self.console.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.console.flush()
        self.log.flush()

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Fine-tuning with Mask')
    # معماری و دیتاست
    parser.add_argument('--arch', type=str, default='resnet56', help='Teacher architecture')
    parser.add_argument('--arch_s', type=str, default='resnet20', choices=['resnet20', 'resnet20_small'],
                        help='Student architecture')
    parser.add_argument('--set', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Dataset')
    # مسیرها
    parser.add_argument('--teacher_ckpt', type=str, required=True, help='Path to teacher checkpoint')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to pruning mask file')
    parser.add_argument('--save_dir', type=str, default='./pretrained_model', help='Save directory')
    # آموزش
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--lr_decay_step', type=str, default='5,8', help='LR decay milestones (comma-separated)')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    return parser

def load_checkpoint(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model

def load_mask(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask not found: {path}")
    return torch.load(path, map_location='cpu')

def transfer_weights_resnet(student, teacher_state, mask):
    student_state = student.state_dict()
    last_select = None
    cnt = -1
    # تنها لایه‌های conv در blockهای resnet را در نظر بگیر
    block_cfg = [3, 3, 3]  # برای resnet20

    for layer_id, blocks in enumerate(block_cfg):
        for block_id in range(blocks):
            for conv_id in [1, 2]:
                cnt += 1
                key = f'layer{layer_id+1}.{block_id}.conv{conv_id}.weight'
                if key not in teacher_state or key not in student_state:
                    continue
                t_w = teacher_state[key]
                s_w = student_state[key]
                t_out, s_out = t_w.size(0), s_w.size(0)
                if t_out != s_out and cnt < len(mask['mask']):
                    indices = torch.argsort(mask['mask'][cnt])[-s_out:]
                    indices, _ = torch.sort(indices)
                    if last_select is not None:
                        for i, ti in enumerate(indices):
                            for j, tj in enumerate(last_select):
                                if i < s_out and j < s_w.size(1):
                                    s_w[i, j] = t_w[ti, tj]
                    else:
                        for i, ti in enumerate(indices):
                            if i < s_out:
                                s_w[i] = t_w[ti]
                    last_select = indices
                elif last_select is not None:
                    for i in range(t_out):
                        for j, tj in enumerate(last_select):
                            if i < s_out and j < s_w.size(1):
                                s_w[i, j] = t_w[i, tj]
                    last_select = None

    # بقیه وزن‌ها (مثل BN و classifier) را کپی کن
    for k in student_state:
        if k in teacher_state and student_state[k].shape == teacher_state[k].shape:
            student_state[k] = teacher_state[k]

    student.load_state_dict(student_state)
    return student

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    # تنظیمات seed و دستگاه
    set_random_seed(args.random_seed)
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # دیتاست
    if args.set == 'cifar10':
        data = CIFAR10(batch_size=args.batch_size, num_workers=4)
        args.num_classes = 10
    elif args.set == 'cifar100':
        data = CIFAR100(batch_size=args.batch_size, num_workers=4)
        args.num_classes = 100
    else:
        raise ValueError("Only cifar10/cifar100 supported")

    # مدل‌ها
    teacher = resnet56(num_classes=args.num_classes)
    student = resnet20(num_classes=args.num_classes)

    # بارگذاری معلم
    teacher = load_checkpoint(teacher, args.teacher_ckpt, device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = set_gpu(args, teacher)

    # بارگذاری mask و انتقال وزن به دانش‌آموز
    mask = load_mask(args.mask_path)
    student = transfer_weights_resnet(student, teacher.state_dict(), mask)
    student = set_gpu(args, student)

    # لاگ
    save_dir = os.path.join(args.save_dir, args.arch_s, args.set)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = CustomLogger(log_path)
    print(f"Log saved to: {log_path}")
    print(f"Args: {args}")

    # اعتبارسنجی معلم
    criterion = nn.CrossEntropyLoss().to(device)
    print("Validating teacher...")
    t1, t5 = validate(data.val_loader, teacher, criterion, args)
    print(f"Teacher Acc@1: {t1:.2f}%, Acc@5: {t5:.2f}%")

    # بهینه‌ساز
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    milestones = [int(x) for x in args.lr_decay_step.split(',')]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_acc1 = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train(data.train_loader, student, criterion, optimizer, epoch, args)
        acc1, acc5 = validate(data.val_loader, student, criterion, args)
        print(f"Val Acc@1: {acc1:.2f}%, Acc@5: {acc5:.2f}%")

        scheduler.step()

        # ذخیره بهترین مدل
        if acc1 > best_acc1:
            best_acc1 = acc1
            torch.save(student.state_dict(), os.path.join(save_dir, 'best_student.pt'))
            print(f"✅ New best model saved (Acc@1: {best_acc1:.2f}%)")

        # ذخیره چک‌پوینت دوره‌ای
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acc1': acc1,
            }, os.path.join(save_dir, f'ckpt_epoch_{epoch+1}.pt'))

    # ذخیره نهایی
    torch.save(student.state_dict(), os.path.join(save_dir, 'final_student.pt'))
    print(f"\nTraining finished. Best Acc@1: {best_acc1:.2f}%")
    print(f"Models saved in: {save_dir}")

if __name__ == '__main__':
    main()
