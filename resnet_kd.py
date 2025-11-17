import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']

def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet_CIFAR(nn.Module):
    """
    ResNet برای CIFAR با پشتیبانی کامل از pruning ساختاری
    """
    def __init__(self, block, num_blocks, in_cfg, out_cfg, num_classes=10):
        super().__init__()
        assert len(in_cfg) == len(out_cfg) == 1 + 3*sum(num_blocks)  # conv1 + 3n blocks

        self.in_cfg = in_cfg
        self.out_cfg = out_cfg

        # conv1
        self.conv1 = nn.Conv2d(3, out_cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_cfg[0])

        # لایه‌ها
        idx = 1
        self.layer1 = self._make_layer(block, num_blocks[0], in_cfg[idx:idx+3], out_cfg[idx:idx+3], stride=1)
        idx += 3
        self.layer2 = self._make_layer(block, num_blocks[1], in_cfg[idx:idx+3], out_cfg[idx:idx+3], stride=2)
        idx += 3
        self.layer3 = self._make_layer(block, num_blocks[2], in_cfg[idx:idx+3], out_cfg[idx:idx+3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_cfg[-1], num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, n_blocks, in_cfg, out_cfg, stride):
        layers = []
        for i in range(n_blocks):
            strd = stride if i == 0 else 1
            layers.append(block(in_cfg[i], out_cfg[i], strd))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# توابع سازنده (همه مدل‌ها)
def _resnet(depth, in_cfg=None, out_cfg=None, num_classes=10):
    n = (depth - 2) // 6
    num_blocks = [n, n, n]
    total_blocks = 1 + 3*n  # conv1 + 3n blocks
    if in_cfg is None or out_cfg is None:
        in_cfg  = [3] + [16]*n + [32]*n + [64]*n
        out_cfg =       [16]*n + [32]*n + [64]*(n+1)  # +1 برای linear
    assert len(in_cfg) == len(out_cfg) == total_blocks
    return ResNet_CIFAR(BasicBlock, num_blocks, in_cfg, out_cfg, num_classes)


def resnet20(in_cfg=None, out_cfg=None, num_classes=10):
    return _resnet(20, in_cfg, out_cfg, num_classes)

def resnet32(in_cfg=None, out_cfg=None, num_classes=10):
    return _resnet(32, in_cfg, out_cfg, num_classes)

def resnet44(in_cfg=None, out_cfg=None, num_classes=10):
    return _resnet(44, in_cfg, out_cfg, num_classes)

def resnet56(in_cfg=None, out_cfg=None, num_classes=10):
    return _resnet(56, in_cfg, out_cfg, num_classes)

def resnet110(in_cfg=None, out_cfg=None, num_classes=10):
    return _resnet(110, in_cfg, out_cfg, num_classes)


# تست سریع (اگر فایل را مستقیم اجرا کنید)
if __name__ == "__main__":
    from thop import profile

    # مدل کامل
    model_full = resnet20(num_classes=10)
    flops, params = profile(model_full, inputs=(torch.randn(1,3,32,32),), verbose=False)
    print(f"Full ResNet-20  → Params: {params/1e6:.3f}M, FLOPs: {flops/1e6:.1f}M")

    # مدل هرس‌شده (مثال واقعی از مقالات)
    in_cfg  = [3, 16, 14, 13, 28, 21, 24, 47, 69, 50]
    out_cfg = [   16, 14, 13, 28, 21, 24, 47, 69, 50, 49]
    model_pruned = resnet20(in_cfg=in_cfg, out_cfg=out_cfg)
    flops, params = profile(model_pruned, inputs=(torch.randn(1,3,32,32),), verbose=False)
    print(f"Pruned ResNet-20 → Params: {params/1e6:.3f}M ({params/272474*100:.1f}% of full), "
          f"FLOPs: {flops/1e6:.1f}M")
