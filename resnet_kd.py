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
    def __init__(self, block, num_blocks, num_classes=10, 
                 option='B', finding_masks=False):
        super(ResNet_KD, self).__init__()
        self.option = option
        self.finding_masks = finding_masks
        self.mask_hooks = []

        # Default configs
        self.out_cfg = [16,16,16,16,32,32,32,64,64,64,64]
        self.strides = [1,1,1,2,1,1,2,1,1]

        self.conv1 = nn.Conv2d(3, self.out_cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_cfg[0])

        layer1 = []
        current_in = self.out_cfg[0]
        for i in range(num_blocks[0]):
            planes = self.out_cfg[i+1]
            stride = self.strides[i]
            layer1.append(block(current_in, planes, stride, option, finding_masks))
            current_in = planes
        self.layer1 = nn.Sequential(*layer1)

        layer2 = []
        for i in range(num_blocks[1]):
            planes = self.out_cfg[num_blocks[0] + i +1]
            stride = self.strides[num_blocks[0] + i]
            layer2.append(block(current_in, planes, stride, option, finding_masks))
            current_in = planes
        self.layer2 = nn.Sequential(*layer2)

        layer3 = []
        for i in range(num_blocks[1]):
            planes = self.out_cfg[num_blocks[0] + num_blocks[1] + i +1]
            stride = self.strides[num_blocks[0] + num_blocks[1] + i]
            layer3.append(block(current_in, planes, stride, option, finding_masks))
            current_in = planes
        self.layer3 = nn.Sequential(*layer3)

        self.linear = nn.Linear(self.out_cfg[-1], num_classes)

        self.apply(_weights_init)

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


def resnet20(num_classes=10, option='B', finding_masks=False):
    return ResNet_KD(BasicBlock_KD, [3, 3, 3], num_classes=num_classes, option=option, finding_masks=finding_masks)


def resnet32(num_classes=10, option='B', finding_masks=False):
    return ResNet_KD(BasicBlock_KD, [5, 5, 5], num_classes=num_classes, option=option, finding_masks=finding_masks)


def resnet44(num_classes=10, option='B', finding_masks=False):
    return ResNet_KD(BasicBlock_KD, [7, 7, 7], num_classes=num_classes, option=option, finding_masks=finding_masks)


def resnet56(num_classes=10, option='B', finding_masks=False):
    return ResNet_KD(BasicBlock_KD, [9, 9, 9], num_classes=num_classes, option=option, finding_masks=finding_masks)


def resnet110(num_classes=10, option='B', finding_masks=False):
    return ResNet_KD(BasicBlock_KD, [18, 18, 18], num_classes=num_classes, option=option, finding_masks=finding_masks)


if __name__ == "__main__":
    # Test without masking
    model1 = resnet20(num_classes=10, option='B', finding_masks=False)
    print("ResNet-20 without masking:")
    print(f"  Parameters: {sum(p.numel() for p in model1.parameters()) / 1e6:.2f}M")
    
    # Test with masking
    model2 = resnet20(num_classes=10, option='B', finding_masks=True)
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
