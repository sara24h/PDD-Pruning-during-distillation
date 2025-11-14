import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import datetime
import argparse

# راه حل 2: اصلاح معماری برای تطابق با checkpoint

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', finding_masks=False):
        super(BasicBlock, self).__init__()
        self.finding_masks = finding_masks
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if self.finding_masks:
            self.mask1 = nn.Parameter(torch.ones(planes, 1, 1, 1))
            self.mask2 = nn.Parameter(torch.ones(planes, 1, 1, 1))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        if self.finding_masks:
            out = out * self.mask1.view(1, -1, 1, 1)
        out = F.relu(self.bn1(out))
        
        out = self.conv2(out)
        if self.finding_masks:
            out = out * self.mask2.view(1, -1, 1, 1)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# معماری Teacher با نام‌گذاری مطابق checkpoint
class BasicBlockTeacher(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockTeacher, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()  # اینجا downsample استفاده می‌کنیم نه shortcut
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNetTeacher(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetTeacher, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # توجه: checkpoint از fc استفاده می‌کنه نه linear
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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
        out = self.fc(out)
        return out

# معماری Student
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, finding_masks=False, option='A'):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.finding_masks = finding_masks
        self.mask_hooks = []
        self.option = option

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        if self.finding_masks:
            self.mask0 = nn.Parameter(torch.ones(16, 1, 1, 1))
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option=self.option, finding_masks=self.finding_masks))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.finding_masks:
            out = out * self.mask0.view(1, -1, 1, 1)
        out = F.relu(self.bn1(out))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def hook_masks(self):
        self.masks_dict = {}
        self.mask_counter = 0
        
        def hook_fn(module, input, output):
            if hasattr(module, 'mask') or (hasattr(module, 'mask0')):
                if hasattr(module, 'mask0'):
                    self.masks_dict[f'mask.{self.mask_counter}'] = type('obj', (object,), {'mask': module.mask0})()
                    self.mask_counter += 1
                if hasattr(module, 'mask1'):
                    self.masks_dict[f'mask.{self.mask_counter}'] = type('obj', (object,), {'mask': module.mask1})()
                    self.mask_counter += 1
                if hasattr(module, 'mask2'):
                    self.masks_dict[f'mask.{self.mask_counter}'] = type('obj', (object,), {'mask': module.mask2})()
                    self.mask_counter += 1
        
        for name, module in self.named_modules():
            if hasattr(module, 'mask0') or hasattr(module, 'mask1') or hasattr(module, 'mask2'):
                handle = module.register_forward_hook(hook_fn)
                self.mask_hooks.append(handle)

    def remove_hooks(self):
        for handle in self.mask_hooks:
            handle.remove()
        self.mask_hooks = []

    def get_masks(self):
        return self.masks_dict

def resnet20(num_classes=10, finding_masks=False, option='A'):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, finding_masks=finding_masks, option=option)

def resnet56_teacher(num_classes=10):
    return ResNetTeacher(BasicBlockTeacher, [9, 9, 9], num_classes=num_classes)

# Dataset
class CIFAR10Data:
    def __init__(self, batch_size=128):
        import torchvision
        import torchvision.transforms as transforms
        
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

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
        self.val_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

def train_KD(train_loader, teacher_model, student_model, divergence_loss, criterion, optimizer, epoch, args):
    teacher_model.eval()
    student_model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        
        with torch.no_grad():
            teacher_output = teacher_model(data)
            teacher_soft = F.softmax(teacher_output / args.temperature, dim=1)
        
        student_output = student_model(data)
        student_soft = F.log_softmax(student_output / args.temperature, dim=1)
        
        kd_loss = divergence_loss(student_soft, teacher_soft, reduction='batchmean') * (args.temperature ** 2)
        cls_loss = criterion(student_output, target)
        loss = args.alpha * kd_loss + (1 - args.alpha) * cls_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = student_output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total, 0.0

def validate(val_loader, model, criterion, args):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total, 0.0

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

def set_random_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(filename):
    import logging
    logger = logging.getLogger('kd_logger')
    logger.handlers.clear()  # پاک کردن handlers قبلی
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def load_pretrained_teacher(model, checkpoint_path, logger):
    """بارگذاری صحیح با نام‌گذاری downsample"""
    if not os.path.exists(checkpoint_path):
        logger.info("Downloading pretrained model...")
        import urllib.request
        url = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt"
        urllib.request.urlretrieve(url, checkpoint_path)
        logger.info(f"Downloaded to {checkpoint_path}")
    
    logger.info(f"Loading from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # حذف 'module.' اگر وجود داره
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    
    state_dict = {k.replace('module.', ''): v for k, v in ckpt.items()}
    
    # بارگذاری با strict=True چون حالا نام‌ها تطابق دارن
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    
    logger.info("✓ Successfully loaded pretrained weights")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--arch_s', default='resnet20', type=str)
    parser.add_argument('--set', default='cifar10', type=str)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_decay_step', default='100,150', type=str)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--temperature', default=3.0, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--save_every', default=10, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    
    args = parser.parse_args()
    
    if args.random_seed is not None:
        set_random_seed(args.random_seed)
    
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = f'pretrained_model/resnet56/{args.set}'
    os.makedirs(log_dir, exist_ok=True)
    
    logger = get_logger(f'{log_dir}/logger_{now}.log')
    logger.info(f"Config: {args}")
    
    # Initialize models
    model_s = resnet20(num_classes=args.num_classes, finding_masks=True, option='A')
    model = resnet56_teacher(num_classes=args.num_classes)
    
    # Load teacher weights
    checkpoint_path = "cifar10_resnet56-187c023a.pt"
    model = load_pretrained_teacher(model, checkpoint_path, logger)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_s = model_s.to(device)
    
    # Freeze teacher
    for param in model.parameters():
        param.requires_grad = False
    
    # Validate teacher
    criterion = nn.CrossEntropyLoss().to(device)
    divergence_loss = F.kl_div
    data = CIFAR10Data(batch_size=args.batch_size)
    
    acc1, _ = validate(data.val_loader, model, criterion, args)
    logger.info(f"✓ Teacher accuracy: {acc1:.2f}%")
    
    if acc1 < 90.0:
        logger.error(f"⚠️ Teacher accuracy too low! ({acc1:.2f}% < 90%)")
        logger.error("Something is wrong with the model loading. Stopping.")
        return
    
    # Setup optimizer
    optimizer = torch.optim.SGD(model_s.parameters(), lr=args.lr, 
                               momentum=args.momentum, weight_decay=args.weight_decay)
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
    
    # Training
    best_acc1 = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        train_acc, _ = train_KD(data.train_loader, model, model_s, 
                                divergence_loss, criterion, optimizer, epoch, args)
        val_acc, _ = validate(data.val_loader, model_s, criterion, args)
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        
        is_best = val_acc > best_acc1
        best_acc1 = max(val_acc, best_acc1)
        
        if is_best or (epoch % args.save_every == 0) or epoch == args.epochs - 1:
            if is_best:
                mask_list = []
                layer_num = []
                
                model_s.hook_masks()
                with torch.no_grad():
                    _ = model_s(torch.randn(1, 3, 32, 32).to(device))
                masks = model_s.get_masks()
                
                for key in sorted(masks.keys()):
                    msk = ApproxSign(masks[key].mask).squeeze()
                    layer_num.append(int(torch.sum(msk).cpu().item()))
                    mask_list.append(msk)
                
                model_s.remove_hooks()
                
                logger.info(f"✓ Best: {val_acc:.2f}% | Layers: {layer_num}")
                torch.save({'layer_num': layer_num, 'mask': mask_list}, f'{log_dir}/mask_best.pt')
                torch.save(model_s.state_dict(), f'{log_dir}/{args.arch_s}_best.pt')
    
    logger.info(f"✓ Done! Best: {best_acc1:.2f}%")

if __name__ == "__main__":
    main()
