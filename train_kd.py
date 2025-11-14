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

# Define ResNet model architecture with masking support
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

    def __init__(self, in_planes, planes, stride=1, option='A', finding_masks=False):
        super(BasicBlock, self).__init__()
        self.finding_masks = finding_masks
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Add mask parameters if finding_masks is True
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
            # Fixed: Use view to reshape mask for proper broadcasting
            out = out * self.mask1.view(1, -1, 1, 1)
        out = F.relu(self.bn1(out))
        
        out = self.conv2(out)
        if self.finding_masks:
            # Fixed: Use view to reshape mask for proper broadcasting
            out = out * self.mask2.view(1, -1, 1, 1)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, finding_masks=False):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.finding_masks = finding_masks
        self.mask_hooks = []

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
            layers.append(block(self.in_planes, planes, stride, finding_masks=self.finding_masks))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.finding_masks:
            # Fixed: Use view to reshape mask for proper broadcasting
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
        """Hook to capture mask values during forward pass"""
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
        """Remove all hooks"""
        for handle in self.mask_hooks:
            handle.remove()
        self.mask_hooks = []

    def get_masks(self):
        """Get all mask values"""
        return self.masks_dict

    def get_masks_outputs(self):
        """Get masks for debugging"""
        masks = {}
        counter = 0
        if hasattr(self, 'mask0'):
            masks[f'mask.{counter}'] = self.mask0
            counter += 1
        
        for module in self.modules():
            if hasattr(module, 'mask1'):
                masks[f'mask.{counter}'] = module.mask1
                counter += 1
            if hasattr(module, 'mask2'):
                masks[f'mask.{counter}'] = module.mask2
                counter += 1
        return masks

def resnet20(num_classes=10, finding_masks=False):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, finding_masks=finding_masks)

def resnet56(num_classes=10, finding_masks=False):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, finding_masks=finding_masks)

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
        kd_loss = divergence_loss(student_soft, teacher_soft, reduction='batchmean') * (args.temperature ** 2)
        
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
def ApproxSign(mask):
    """Approximate sign function for mask binarization"""
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
    parser = argparse.ArgumentParser(description='Knowledge Distillation for CIFAR10 with Dynamic Masking')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--arch', default='resnet56', type=str, help='teacher model architecture')
    parser.add_argument('--arch_s', default='resnet20', type=str, help='student model architecture')
    parser.add_argument('--set', default='cifar10', type=str, help='dataset name')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--lr_decay_step', default='100,150', type=str, help='learning rate decay steps')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--pretrained', action='store_true', help='use pre-trained teacher model')
    parser.add_argument('--temperature', default=3.0, type=float, help='temperature for soft targets')
    parser.add_argument('--alpha', default=0.5, type=float, help='weight for KD loss')
    parser.add_argument('--random_seed', default=42, type=int, help='random seed')
    parser.add_argument('--save_every', default=10, type=int, help='save frequency')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(f'logs/print_process_{now}.log', sys.stdout)

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
    
    # Initialize student model with masking support
    if args.arch_s == 'resnet20':
        model_s = resnet20(num_classes=args.num_classes, finding_masks=True)
    
    # Initialize teacher model (without masking)
    if args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes, finding_masks=False)
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
    
                # Filter out mismatched keys
                model_state_dict = model.state_dict()
                filtered_ckpt = {}
                for k, v in ckpt.items():
                    if k in model_state_dict and model_state_dict[k].shape == v.shape:
                        filtered_ckpt[k] = v
                    else:
                        logger.warning(f"Skipping parameter: {k} due to shape mismatch or not found in model")
                
                # Update the model's state dict
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
        logger.info(f"Parameter of the Teacher Model: {name}, Requires Gradient: {param.requires_grad}")
    
    logger.info("-" * 100)
    logger.info("Student model parameters (trainable):")
    for name, param in model_s.named_parameters():
        logger.info(f"Parameter of the Student Model: {name}, Requires Gradient: {param.requires_grad}")
    
    # Evaluate teacher model
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    divergence_loss = F.kl_div
    data = CIFAR10Data()
    
    acc1, _ = validate(data.val_loader, model, criterion, args)
    logger.info(f"Teacher model: {acc1:.2f}%")

    # Set up optimizer and scheduler for student model
    optimizer = torch.optim.SGD(model_s.parameters(), lr=args.lr, 
                               momentum=args.momentum, weight_decay=args.weight_decay)
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    # Training loop
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0
    mask_list = []
    layer_num = []
    
    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"Epoch [{epoch+1}/{args.epochs}]")
        
        # Train student model with knowledge distillation
        train_acc1, train_acc5 = train_KD(data.train_loader, model, model_s, 
                                          divergence_loss, criterion, optimizer, epoch, args)
        
        # Evaluate student model
        acc1, acc5 = validate(data.val_loader, model_s, criterion, args)
        scheduler.step()
        
        logger.info(f"Train Accuracy: {train_acc1:.2f}%")
        logger.info(f"Validation Accuracy: {acc1:.2f}%")
        
        # Remember best acc@1 and save checkpoint
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
                
                # Hook masks and extract them
                model_s.hook_masks()
                # Run a dummy forward pass to populate masks
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 32, 32).to(device)
                    _ = model_s(dummy_input)
                masks = model_s.get_masks()
                
                for key in masks.keys():
                    msk = ApproxSign(masks[key].mask).squeeze()
                    total = torch.sum(msk)
                    layer_num.append(int(total.cpu().detach().numpy()))
                    mask_list.append(msk)
                
                model_s.remove_hooks()
                
                logger.info(f"New best accuracy: {acc1:.2f}%")
                logger.info(f"Layer numbers: {layer_num}")
                
                # Save masks
                to = {'layer_num': layer_num, 'mask': mask_list}
                torch.save(to, f'{log_dir}/{args.set}_T_{args.arch}_S_{args.arch_s}_mask.pt')
                logger.info(f"Saved masks to {log_dir}/{args.set}_T_{args.arch}_S_{args.arch_s}_mask.pt")
                
                # Save student model
                torch.save(model_s.state_dict(), f'{log_dir}/{args.set}_{args.arch_s}.pt')
                logger.info(f"Saved best student model to {log_dir}/{args.set}_{args.arch_s}.pt")
            
            # Regular saving
            if save or epoch == args.epochs - 1:
                torch.save(model_s.state_dict(), f'{log_dir}/{args.set}_{args.arch_s}_epoch_{epoch+1}.pt')
                logger.info(f"Saved checkpoint at epoch {epoch+1}")
    
    logger.info(f"Training complete. Best accuracy: {best_acc1:.2f}%")
    logger.info(f"Final layer numbers: {layer_num}")

if __name__ == "__main__":
    main()
