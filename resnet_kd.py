"""
ResNet for CIFAR-10 with Knowledge Distillation support and dynamic masking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']

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


class DynamicMask(nn.Module):
    """Dynamic mask layer for pruning during distillation"""
    def __init__(self, channels):
        super(DynamicMask, self).__init__()
        self.mask = nn.Parameter(torch.ones(1, channels, 1, 1))
        
    def forward(self, x):
        return x * self.mask


class BasicBlock_KD(nn.Module):
    """BasicBlock with optional dynamic masking"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B', finding_masks=False):
        super(BasicBlock_KD, self).__init__()
        self.finding_masks = finding_masks
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Add dynamic mask if finding_masks is True
        if self.finding_masks:
            self.mask = DynamicMask(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply mask if finding_masks is enabled
        if self.finding_masks:
            out = self.mask(out)
            
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_KD(nn.Module):
    """ResNet with Knowledge Distillation support"""
    def __init__(self, block, num_blocks, in_cfg=None, out_cfg=None, num_classes=10, 
                 option='B', finding_masks=False):
        super(ResNet_KD, self).__init__()
        self.in_planes = 16
        self.option = option
        self.finding_masks = finding_masks
        self.mask_hooks = []
        
        # Default configs if not provided
        if in_cfg is None:
            in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]
        if out_cfg is None:
            out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
            
        self.in_cfg = in_cfg
        self.out_cfg = out_cfg

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
            layers.append(block(self.in_planes, planes, stride, 
                              option=self.option, finding_masks=self.finding_masks))
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
    
    def hook_masks(self):
        """Hook to capture mask values"""
        self.masks_dict = {}
        
        def hook_fn(name):
            def fn(module, input, output):
                if hasattr(module, 'mask'):
                    self.masks_dict[name] = module.mask
            return fn
        
        for name, module in self.named_modules():
            if isinstance(module, DynamicMask):
                hook = module.register_forward_hook(hook_fn(name))
                self.mask_hooks.append(hook)
    
    def get_masks(self):
        """Return captured masks"""
        return self.masks_dict
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.mask_hooks:
            hook.remove()
        self.mask_hooks = []


def resnet20(num_classes=10, option='B', finding_masks=False, in_cfg=None, out_cfg=None):
    """
    ResNet-20 for Knowledge Distillation
    
    Args:
        num_classes: number of output classes
        option: 'A' or 'B' for shortcut connection
        finding_masks: whether to use dynamic masking
        in_cfg: input channel configuration
        out_cfg: output channel configuration
    """
    return ResNet_KD(BasicBlock_KD, [3, 3, 3], in_cfg=in_cfg, out_cfg=out_cfg,
                     num_classes=num_classes, option=option, finding_masks=finding_masks)


def resnet32(num_classes=10, option='B', finding_masks=False, in_cfg=None, out_cfg=None):
    return ResNet_KD(BasicBlock_KD, [5, 5, 5], in_cfg=in_cfg, out_cfg=out_cfg,
                     num_classes=num_classes, option=option, finding_masks=finding_masks)


def resnet44(num_classes=10, option='B', finding_masks=False, in_cfg=None, out_cfg=None):
    return ResNet_KD(BasicBlock_KD, [7, 7, 7], in_cfg=in_cfg, out_cfg=out_cfg,
                     num_classes=num_classes, option=option, finding_masks=finding_masks)


def resnet56(num_classes=10, option='B', finding_masks=False, in_cfg=None, out_cfg=None):
    return ResNet_KD(BasicBlock_KD, [9, 9, 9], in_cfg=in_cfg, out_cfg=out_cfg,
                     num_classes=num_classes, option=option, finding_masks=finding_masks)


def resnet110(num_classes=10, option='B', finding_masks=False, in_cfg=None, out_cfg=None):
    return ResNet_KD(BasicBlock_KD, [18, 18, 18], in_cfg=in_cfg, out_cfg=out_cfg,
                     num_classes=num_classes, option=option, finding_masks=finding_masks)


if __name__ == "__main__":
    # Test without masking
    model1 = resnet20(num_classes=10, option='B', finding_masks=False)
    print("ResNet-20 without masking:")
    print(f"  Parameters: {sum(p.numel() for p in model1.parameters()) / 1e6:.2f}M")
    
    # Test with masking
    in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]
    out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
    model2 = resnet20(num_classes=10, option='B', finding_masks=True, 
                      in_cfg=in_cfg, out_cfg=out_cfg)
    print("\nResNet-20 with masking:")
    print(f"  Parameters: {sum(p.numel() for p in model2.parameters()) / 1e6:.2f}M")
    
    # Test forward pass
    input_tensor = torch.rand((2, 3, 32, 32))
    output1 = model1(input_tensor)
    output2 = model2(input_tensor)
    print(f"\nOutput shape (no mask): {output1.shape}")
    print(f"Output shape (with mask): {output2.shape}")
    
    # Test mask extraction
    model2.hook_masks()
    _ = model2(input_tensor)
    masks = model2.get_masks()
    print(f"\nNumber of masks captured: {len(masks)}")
    model2.remove_hooks()
