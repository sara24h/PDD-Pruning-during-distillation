import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']


class Mask(nn.Module):
    def __init__(self, size=(1, 16, 1, 1), finding_masks=True):
        super(Mask, self).__init__()
        self.mask = nn.Parameter(torch.randn(size, requires_grad=True) * 0.001)
        self.size = size
        self.finding_masks = finding_masks
        self.register_buffer('binary_mask', torch.ones(size))

    def forward(self, x):
        if self.finding_masks:
            out_forward = torch.sign(self.mask)
            mask1 = self.mask < -1
            mask2 = self.mask < 0
            mask3 = self.mask < 1
            out1 = (-1) * mask1.type(torch.float32) + (self.mask * self.mask + 2 * self.mask) * (1 - mask1.type(torch.float32))
            out2 = out1 * mask2.type(torch.float32) + (-self.mask * self.mask + 2 * self.mask) * (1 - mask2.type(torch.float32))
            out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
            out = out_forward.detach() - out3.detach() + out3
            self.binary_mask = ((out + 1) / 2).detach()
            return self.binary_mask * x
        else:
            return self.binary_mask * x

    def apply_threshold(self, threshold=0.5):
        """Convert learned mask to binary mask based on threshold"""
        self.binary_mask = (self.mask > threshold).float().detach()
        self.finding_masks = False
        return self.binary_mask.sum().item()


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

    def __init__(self, in_planes, planes, stride=1, option='A', finding_masks=True):
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
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes - in_planes) // 2, planes - in_planes - (planes - in_planes) // 2), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        
        self.mask = Mask((1, planes, 1, 1), finding_masks)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.mask(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, finding_masks=True):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.finding_masks = finding_masks
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, finding_masks=finding_masks)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, finding_masks=finding_masks)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, finding_masks=finding_masks)
        self.linear = nn.Linear(64, num_classes)
        
        # برای سازگاری با مدل‌های پیش‌آموزش دیده
        self.fc = self.linear
        
        self.apply(_weights_init)
        self.handlers = []
        self.masks_outputs = {}
        self.masks = {}

    def _make_layer(self, block, planes, num_blocks, stride, finding_masks=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, finding_masks=finding_masks))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def hook_mask(self, layer, mask_name):
        def hook_function(module, input, output):
            self.masks_outputs[mask_name] = output
        return layer.register_forward_hook(hook_function)

    def hook_masks(self):
        self.remove_hooks()
        
        ind = 0
        for name, layer in self.layer1.named_children():
            layer_name = f'mask.{ind}'
            self.handlers.append(self.hook_mask(layer.mask, layer_name))
            ind += 1
            
        for name, layer in self.layer2.named_children():
            layer_name = f'mask.{ind}'
            self.handlers.append(self.hook_mask(layer.mask, layer_name))
            ind += 1
            
        for name, layer in self.layer3.named_children():
            layer_name = f'mask.{ind}'
            self.handlers.append(self.hook_mask(layer.mask, layer_name))
            ind += 1

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()
        self.handlers.clear()
        self.masks_outputs.clear()

    def get_masks(self):
        self.masks = {}
        ind = 0
        
        for name, layer in self.layer1.named_children():
            layer_name = f'mask.{ind}'
            self.masks[layer_name] = layer.mask
            ind += 1
            
        for name, layer in self.layer2.named_children():
            layer_name = f'mask.{ind}'
            self.masks[layer_name] = layer.mask
            ind += 1
            
        for name, layer in self.layer3.named_children():
            layer_name = f'mask.{ind}'
            self.masks[layer_name] = layer.mask
            ind += 1
            
        return self.masks

    def get_active_neuron_counts(self):
        active_counts = []
        masks = self.get_masks()
        for key in sorted(masks.keys()):
            mask = masks[key]
            if hasattr(mask, 'binary_mask'):
                active_count = mask.binary_mask.sum().item()
                active_counts.append(int(active_count))
        return active_counts

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def load_pretrained_weights(self, state_dict):
        new_state_dict = {}
        
        for key, value in state_dict.items():
            new_key = key
            
            # Handle DataParallel wrapping
            if new_key.startswith('module.'):
                new_key = new_key[7:]
                
            # Handle fc vs linear naming
            if new_key.startswith('fc.'):
                new_key = new_key.replace('fc.', 'linear.', 1)
                
            # Handle layer specific naming differences
            if 'downsample.0' in new_key:
                new_key = new_key.replace('downsample.0', 'shortcut.0')
            if 'downsample.1' in new_key:
                new_key = new_key.replace('downsample.1', 'shortcut.1')
            if 'downsample.conv' in new_key:
                new_key = new_key.replace('downsample.conv', 'shortcut.0')
            if 'downsample.bn' in new_key:
                new_key = new_key.replace('downsample.bn', 'shortcut.1')
                
            new_state_dict[new_key] = value
        
        # First try strict loading
        try:
            self.load_state_dict(new_state_dict, strict=True)
            print("✓ Model weights loaded successfully with strict=True")
            return [], []
        except RuntimeError as e:
            print(f"⚠ Warning: {str(e)[:200]}...")
            print("Trying with strict=False...")
            missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys ({len(missing_keys)}):")
                for k in missing_keys[:10]:
                    print(f"  - {k}")
                if len(missing_keys) > 10:
                    print(f"  ... and {len(missing_keys)-10} more")
            
            if unexpected_keys:
                print(f"Unexpected keys ({len(unexpected_keys)}):")
                for k in unexpected_keys[:10]:
                    print(f"  + {k}")
                if len(unexpected_keys) > 10:
                    print(f"  ... and {len(unexpected_keys)-10} more")
            
            print("✓ Model weights loaded with strict=False")
            return list(missing_keys), list(unexpected_keys)


def resnet20(num_classes=10, finding_masks=True):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, finding_masks=finding_masks)


def resnet32(num_classes=10, finding_masks=True):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, finding_masks=finding_masks)


def resnet44(num_classes=10, finding_masks=True):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, finding_masks=finding_masks)


def resnet56(num_classes=10, finding_masks=True):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, finding_masks=finding_masks)


def resnet110(num_classes=10, finding_masks=True):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, finding_masks=finding_masks)


# تست ساده برای تأیید عملکرد
if __name__ == "__main__":
    import numpy as np
    
    # تست مدل ResNet20
    model = resnet20(finding_masks=True)
    print("✓ ResNet20 model created successfully")
    
    # تست forward pass
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"✓ Forward pass successful. Output shape: {y.shape}")
    
    # تست هوک‌های mask
    model.hook_masks()
    masks = model.get_masks()
    print(f"✓ Number of mask layers: {len(masks)}")
    
    # تست forward pass با هوک‌ها
    y = model(x)
    print(f"✓ Forward pass with hooks successful. Captured {len(model.masks_outputs)} mask outputs")
    
    # تست شمارش نورون‌های فعال
    counts = model.get_active_neuron_counts()
    print(f"✓ Active neurons per layer: {counts}")
    print(f"✓ Total active neurons: {np.sum(counts)}")
    
    # پاک کردن هوک‌ها
    model.remove_hooks()
    print("✓ Hooks removed successfully")
    
    # تست بارگذاری وزن‌های پیش‌آموزش دیده (شبیه‌سازی)
    print("\nTesting pretrained weight loading...")
    dummy_state_dict = {
        'conv1.weight': torch.randn(16, 3, 3, 3),
        'bn1.weight': torch.randn(16),
        'bn1.bias': torch.randn(16),
        'layer1.0.conv1.weight': torch.randn(16, 16, 3, 3),
        'layer1.0.bn1.weight': torch.randn(16),
        'layer1.0.bn1.bias': torch.randn(16),
        'layer1.0.conv2.weight': torch.randn(16, 16, 3, 3),
        'layer1.0.bn2.weight': torch.randn(16),
        'layer1.0.bn2.bias': torch.randn(16),
        'layer1.0.shortcut.0.weight': torch.randn(16, 16, 1, 1),  # شبیه‌سازی کلیدهای مختلف
        'linear.weight': torch.randn(10, 64),
        'linear.bias': torch.randn(10)
    }
    
    missing, unexpected = model.load_pretrained_weights(dummy_state_dict)
    print(f"✓ Tested pretrained weight loading with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
