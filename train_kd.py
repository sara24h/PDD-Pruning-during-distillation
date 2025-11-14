import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import datetime
import argparse
from torch.autograd import Variable

# Define ResNet model architecture (borrowed from https://github.com/akamaster/pytorch_resnet_cifar10)
def _weights_init(m):
    classname = m.__class__.__name__
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

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                # For CIFAR10 ResNet paper uses option A
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
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

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

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
        out = self.linear(out)
        return out

def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)

def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)

# Define dataset loaders
class CIFAR10Data:
    def __init__(self):
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
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
        self.val_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

# Training and validation functions
def train_KD(train_loader, teacher_model, student_model, divergence_loss, criterion, optimizer, epoch, args):
    teacher_model.eval()
    student_model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_output = teacher_model(data)
            teacher_soft = F.softmax(teacher_output / args.temperature, dim=1)
        
        # Get student predictions
        student_output = student_model(data)
        student_soft = F.log_softmax(student_output / args.temperature, dim=1)
        
        # Knowledge distillation loss
        kd_loss = divergence_loss(student_soft, teacher_soft) * (args.temperature ** 2)
        
        # Classification loss
        cls_loss = criterion(student_output, target)
        
        # Total loss
        loss = args.alpha * kd_loss + (1 - args.alpha) * cls_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = student_output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    acc1 = 100. * correct / total
    return acc1, 0.0  # returning dummy acc5 for compatibility

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
    
    acc1 = 100. * correct / total
    return acc1, 0.0  # returning dummy acc5 for compatibility

# Helper functions
def set_random_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_gpu(args, model):
    if torch.cuda.is_available():
        model = model.cuda()
    return model

class Logger(object):
    def __init__(self, filename, stdout):
        self.stdout = stdout
        self.log = open(filename, "a")
    
    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)
    
    def flush(self):
        self.stdout.flush()
        self.log.flush()

def get_logger(filename):
    import logging
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# Main function
def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation for CIFAR10')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--arch', default='resnet56', type=str, help='teacher model architecture')
    parser.add_argument('--arch_s', default='resnet20', type=str, help='student model architecture')
    parser.add_argument('--set', default='cifar10', type=str, help='dataset name')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--lr_decay_step', default='100,150', type=str, help='learning rate decay steps')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--pretrained', action='store_true', help='use pre-trained teacher model')
    parser.add_argument('--temperature', default=3.0, type=float, help='temperature for soft targets')
    parser.add_argument('--alpha', default=0.5, type=float, help='weight for KD loss')
    parser.add_argument('--random_seed', default=42, type=int, help='random seed')
    parser.add_argument('--save_every', default=10, type=int, help='save frequency')
    
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(f'logs/process_{now}.log', sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)

def main_worker(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('pretrained_model/' + args.arch + '/' + args.set):
        os.makedirs('pretrained_model/' + args.arch + '/' + args.set, exist_ok=True)
    
    log_dir = 'pretrained_model/' + args.arch + '/' + args.set
    logger = get_logger(f'{log_dir}/logger_{now}.log')
    logger.info(f"Architecture: {args.arch}")
    logger.info(f"Dataset: {args.set}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"LR decay steps: {args.lr_decay_step}")
    logger.info(f"Number of classes: {args.num_classes}")
    
    # Initialize student model
    if args.arch_s == 'resnet20':
        model_s = resnet20(num_classes=args.num_classes)
    
    # Initialize teacher model
    if args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                checkpoint_path = "cifar10_resnet56-187c023a.pt"
                
                # Check if file exists, if not, download it
                if not os.path.exists(checkpoint_path):
                    logger.info("Downloading pretrained model...")
                    import urllib.request
                    url = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt"
                    urllib.request.urlretrieve(url, checkpoint_path)
                    logger.info(f"Downloaded pretrained model to {checkpoint_path}")
                
                # Load checkpoint
                logger.info(f"Loading pretrained model from {checkpoint_path}")
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                
                # Check if the checkpoint has 'state_dict' key
                if 'state_dict' in ckpt:
                    ckpt = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
                else:
                    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    
    # Handle loading the teacher model
    if args.pretrained:
        # Filter out mismatched keys if any
        model_state_dict = model.state_dict()
        filtered_ckpt = {}
        for k, v in ckpt.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                filtered_ckpt[k] = v
            else:
                logger.warning(f"Skipping parameter: {k} due to shape mismatch or not found in model")
        
        # Update the model's state dict with the filtered checkpoint
        model_state_dict.update(filtered_ckpt)
        model.load_state_dict(model_state_dict)
        logger.info("Successfully loaded pretrained weights")
    
    # Set GPU
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_s = model_s.to(device)
    
    # Freeze teacher model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Print model parameters and gradient status
    logger.info("Teacher model parameters (frozen):")
    for name, param in model.named_parameters():
        logger.info(f"{name}, Requires Gradient: {param.requires_grad}")
    
    logger.info("-" * 100)
    logger.info("Student model parameters (trainable):")
    for name, param in model_s.named_parameters():
        logger.info(f"{name}, Requires Gradient: {param.requires_grad}")
    
    # Evaluate teacher model
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    data = CIFAR10Data()
    
    acc1, _ = validate(data.val_loader, model, criterion, args)
    logger.info(f"Teacher model accuracy: {acc1:.2f}%")

    # Set up optimizer and scheduler for student model
    optimizer = torch.optim.SGD(model_s.parameters(), lr=args.lr, 
                               momentum=0.9, weight_decay=args.weight_decay)
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    # Training loop
    best_acc1 = 0.0
    for epoch in range(args.epochs):
        logger.info(f"Epoch [{epoch+1}/{args.epochs}]")
        
        # Train student model with knowledge distillation
        train_acc1, _ = train_KD(data.train_loader, model, model_s, 
                                lambda x, y: F.kl_div(x, y, reduction='batchmean'), 
                                criterion, optimizer, epoch, args)
        
        # Evaluate student model
        acc1, _ = validate(data.val_loader, model_s, criterion, args)
        scheduler.step()
        
        logger.info(f"Train Accuracy: {train_acc1:.2f}%")
        logger.info(f"Validation Accuracy: {acc1:.2f}%")
        
        # Save best model
        if acc1 > best_acc1:
            best_acc1 = acc1
            logger.info(f"New best accuracy: {best_acc1:.2f}%")
            
            # Save student model
            torch.save(model_s.state_dict(), f'{log_dir}/{args.set}_{args.arch_s}_best.pt')
            logger.info(f"Saved best student model to {log_dir}/{args.set}_{args.arch_s}_best.pt")
        
        # Regular saving
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(model_s.state_dict(), f'{log_dir}/{args.set}_{args.arch_s}_epoch_{epoch+1}.pt')
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
    
    logger.info(f"Training complete. Best accuracy: {best_acc1:.2f}%")

if __name__ == "__main__":
    main()
