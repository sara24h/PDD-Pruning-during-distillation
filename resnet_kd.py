# resnet_kd.py → نسخه نهایی (اصلاح assert و محاسبه بلوک‌ها)
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
    def __init__(self, block, num_blocks, in_cfg, out_cfg, num_classes=10):
        super().__init__()
        # ResNet-CIFAR: 1 conv1 + sum(num_blocks) BasicBlocks
        # Example ResNet20: 1 + (3+3+3) = 10 conv layers
        total_blocks = 1 + sum(num_blocks)  # conv1 + all blocks
        
        # in_cfg: کانال‌های ورودی برای conv1 + هر block
        # out_cfg: کانال‌های خروجی برای conv1 + هر block + fc
        assert len(in_cfg) == total_blocks, \
            f"in_cfg length {len(in_cfg)} != expected {total_blocks} (1 conv1 + {sum(num_blocks)} blocks)"
        assert len(out_cfg) == total_blocks + 1, \
            f"out_cfg length {len(out_cfg)} != expected {total_blocks + 1} (includes fc output)"

        self.in_cfg = in_cfg
        self.out_cfg = out_cfg

        # Initial convolution
        self.conv1 = nn.Conv2d(in_cfg[0], out_cfg[0], kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_cfg[0])

        # Build 3 stages
        idx = 1  # شروع از index 1 (بعد از conv1)
        self.layer1 = self._make_layer(block, num_blocks[0], 
                                        in_cfg[idx:idx+num_blocks[0]], 
                                        out_cfg[idx:idx+num_blocks[0]], stride=1)
        idx += num_blocks[0]
        
        self.layer2 = self._make_layer(block, num_blocks[1], 
                                        in_cfg[idx:idx+num_blocks[1]], 
                                        out_cfg[idx:idx+num_blocks[1]], stride=2)
        idx += num_blocks[1]
        
        self.layer3 = self._make_layer(block, num_blocks[2], 
                                        in_cfg[idx:idx+num_blocks[2]], 
                                        out_cfg[idx:idx+num_blocks[2]], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # FC layer: آخرین out_cfg (index -1) به num_classes
        self.fc = nn.Linear(out_cfg[-1], num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, n_blocks, in_cfg, out_cfg, stride):
        """ساخت یک stage با n_blocks بلوک"""
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


# توابع سازنده
def _resnet(depth, in_cfg=None, out_cfg=None, num_classes=10):
    """
    ساخت ResNet-CIFAR با عمق دلخواه
    
    Args:
        depth: عمق شبکه (20, 32, 44, 56, 110)
        in_cfg: کانال‌های ورودی (اختیاری)
        out_cfg: کانال‌های خروجی (اختیاری)
    """
    n = (depth - 2) // 6  # تعداد block در هر stage
    num_blocks = [n, n, n]
    total_blocks = 1 + sum(num_blocks)  # conv1 + all blocks
    
    # Configuration پیش‌فرض
    if in_cfg is None or out_cfg is None:
        # in_cfg: [3 (input)] + [16]*n + [16,32,...,32] + [32,64,...,64]
        in_cfg = [3] + [16]*n + [16] + [32]*(n-1) + [32] + [64]*(n-1)
        # out_cfg: [16] + [16]*n + [32]*n + [64]*n + [64 (for fc input)]
        out_cfg = [16] + [16]*n + [32]*n + [64]*n + [64]
        
        print(f"Using default config for ResNet{depth}:")
        print(f"  n={n}, total_blocks={total_blocks}")
        print(f"  in_cfg length: {len(in_cfg)}")
        print(f"  out_cfg length: {len(out_cfg)}")
    
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


# تست
if __name__ == "__main__":
    print("="*80)
    print("Testing ResNet20 Configuration")
    print("="*80)
    
    # تست با configuration دستی
    in_cfg = [3, 16, 16, 16, 16, 32, 32, 32, 64, 64]  # 10 elements
    out_cfg = [16, 16, 16, 16, 32, 32, 32, 64, 64, 64, 10]  # 11 elements
    
    print(f"\nManual config:")
    print(f"  in_cfg:  {in_cfg} (length={len(in_cfg)})")
    print(f"  out_cfg: {out_cfg} (length={len(out_cfg)})")
    
    try:
        model = resnet20(in_cfg=in_cfg, out_cfg=out_cfg, num_classes=10)
        print(f"✓ Model created successfully!")
        
        # تست forward pass
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        print(f"✓ Forward pass successful: input {x.shape} → output {y.shape}")
        
        # شمارش پارامترها
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*80)
    print("Testing with default config")
    print("="*80)
    
    try:
        model_default = resnet20()
        x = torch.randn(2, 3, 32, 32)
        y = model_default(x)
        print(f"✓ Default model works: {y.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
