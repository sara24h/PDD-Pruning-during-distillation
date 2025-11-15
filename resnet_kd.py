# borrow from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet56_KD']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):  # ✅ تغییر به Option B
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                # ✅ Option B برای سازگاری با pytorch-cifar-models
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


class BasicBlock_small(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):  # ✅ تغییر به Option B
        super(BasicBlock_small, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, option='B'):  # ✅ پارامتر option اضافه شد
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.option = option

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
            layers.append(block(self.in_planes, planes, stride, option=self.option))  # ✅ ارسال option
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


class ResNet_small(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, option='B'):  # ✅ پارامتر option اضافه شد
        super(ResNet_small, self).__init__()
        self.in_planes = 16
        self.option = option

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
            layers.append(block(self.in_planes, planes, stride, option=self.option))  # ✅ ارسال option
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


def resnet20(num_classes=10, option='B'):
    """ResNet-20 با Option B (پیش‌فرض) برای سازگاری با checkpoints"""
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, option=option)


def resnet32(num_classes=10, option='B'):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, option=option)


def resnet44(num_classes=10, option='B'):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, option=option)


def resnet56(num_classes=10, option='B'):
    """
    ResNet-56 با Option B (پیش‌فرض) برای سازگاری با pytorch-cifar-models
    برای استفاده از Option A اصلی: resnet56(num_classes=10, option='A')
    """
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, option=option)


def resnet110(num_classes=10, option='B'):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, option=option)


def resnet56_KD(num_classes=10, option='B'):
    return ResNet(BasicBlock, [6, 1, 7], num_classes=num_classes, option=option)


if __name__ == "__main__":
    # تست با Option B
    model_b = resnet56(num_classes=10, option='B')
    print("ResNet-56 with Option B (downsample):")
    print(f"  Parameters: {sum(p.numel() for p in model_b.parameters()) / 1e6:.2f}M")
    
    # تست با Option A
    model_a = resnet56(num_classes=10, option='A')
    print("\nResNet-56 with Option A (padding):")
    print(f"  Parameters: {sum(p.numel() for p in model_a.parameters()) / 1e6:.2f}M")
    
    # تست forward
    input_tensor = torch.rand((2, 3, 32, 32))
    output = model_b(input_tensor)
    print(f"\nOutput shape: {output.shape}")
