import datetime
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------------
# ResNet for CIFAR (20/56 layers)
# -----------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)

def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)

# -----------------------------
# Training & Validation
# -----------------------------
def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    device = next(model.parameters()).device
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(val_loader, model, criterion, args):
    model.eval()
    device = next(model.parameters()).device
    correct1 = correct5 = total = 0
    with torch.no_grad():
        for images, target in val_loader:
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct1 += correct[:1].sum().item()
            correct5 += correct[:5].sum().item()
            total += target.size(0)
    acc1 = 100.0 * correct1 / total
    acc5 = 100.0 * correct5 / total
    return acc1, acc5

# -----------------------------
# Utilities
# -----------------------------
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

def set_random_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_gpu(args, model):
    if torch.cuda.is_available():
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model

# -----------------------------
# Weight Transfer with Mask
# -----------------------------
def transfer_weights_resnet(student, teacher_state, mask):
    student_state = student.state_dict()
    last_select = None
    cnt = -1
    block_cfg = [3, 3, 3]  # resnet20

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

    # Copy compatible weights (BN, linear, etc.)
    for k in student_state:
        if k in teacher_state and student_state[k].shape == teacher_state[k].shape:
            student_state[k] = teacher_state[k]

    student.load_state_dict(student_state)
    return student

# -----------------------------
# Argument Parser
# -----------------------------
def setup_arg_parser():
    parser = argparse.ArgumentParser(description='PDD Fine-tuning with Mask')
    parser.add_argument('--arch', type=str, default='resnet56', help='Teacher architecture')
    parser.add_argument('--arch_s', type=str, default='resnet20', choices=['resnet20'],
                        help='Student architecture')
    parser.add_argument('--set', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Dataset')
    parser.add_argument('--teacher_ckpt', type=str, required=True, help='Path to teacher checkpoint')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to mask file')
    parser.add_argument('--save_dir', type=str, default='./results', help='Save directory')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--lr_decay_step', type=str, default='5,8', help='LR decay steps')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (use -1 for CPU)')
    parser.add_argument('--save_every', type=int, default=1, help='Save every N epochs')
    return parser

# -----------------------------
# Main
# -----------------------------
def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    set_random_seed(args.random_seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.set == 'cifar10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
        num_classes = 10
    else:
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        valset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
        num_classes = 100

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Models
    teacher = resnet56(num_classes=num_classes)
    student = resnet20(num_classes=num_classes)

    # Load teacher
    ckpt_path = args.teacher_ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    teacher.load_state_dict(state_dict, strict=False)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = teacher.to(device)

    # Load mask & transfer weights
    mask = torch.load(args.mask_path, map_location='cpu')
    student = transfer_weights_resnet(student, teacher.state_dict(), mask)
    student = student.to(device)

    # Logging
    save_dir = os.path.join(args.save_dir, args.arch_s, args.set)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = CustomLogger(log_path)
    print(f"Log saved to: {log_path}")
    print(f"Args: {args}")

    # Validate teacher
    criterion = nn.CrossEntropyLoss().to(device)
    print("Validating teacher...")
    t1, t5 = validate(val_loader, teacher, criterion, args)
    print(f"Teacher Acc@1: {t1:.2f}%, Acc@5: {t5:.2f}%")

    # Optimizer
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    milestones = [int(x) for x in args.lr_decay_step.split(',')]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_acc1 = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train(train_loader, student, criterion, optimizer, epoch, args)
        acc1, acc5 = validate(val_loader, student, criterion, args)
        print(f"Val Acc@1: {acc1:.2f}%, Acc@5: {acc5:.2f}%")
        scheduler.step()

        if acc1 > best_acc1:
            best_acc1 = acc1
            torch.save(student.state_dict(), os.path.join(save_dir, 'best_student.pth'))
            print(f"✅ Best model saved (Acc@1: {best_acc1:.2f}%)")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acc1': acc1,
            }, os.path.join(save_dir, f'ckpt_epoch_{epoch+1}.pth'))

    torch.save(student.state_dict(), os.path.join(save_dir, 'final_student.pth'))
    print(f"\n✅ Training finished. Best Acc@1: {best_acc1:.2f}%")
    print(f"Models saved in: {save_dir}")

if __name__ == '__main__':
    main()
