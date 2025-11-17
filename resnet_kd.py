# resnet_kd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']

def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class DynamicMask(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mask = nn.Parameter(torch.ones(1, channels, 1, 1))
    def forward(self, x):
        return x * self.mask

class BasicBlock_KD(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='B', finding_masks=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.finding_masks = finding_masks
        if finding_masks:
            self.mask = DynamicMask(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.finding_masks:
            out = self.mask(out)
        out += self.shortcut(x)
        return F.relu(out)

class ResNet_KD(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, finding_masks=False):
        super().__init__()
        self.in_planes = 16
        self.finding_masks = finding_masks
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
        for s in strides:
            layers.append(block(self.in_planes, planes, s, 'B', self.finding_masks))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.linear(out)

def resnet20(num_classes=10, finding_masks=False):
    return ResNet_KD(BasicBlock_KD, [3, 3, 3], num_classes, finding_masks)

def resnet56(num_classes=10, finding_masks=False):
    return ResNet_KD(BasicBlock_KD, [9, 9, 9], num_classes, finding_masks)
