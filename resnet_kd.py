import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

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

    def extra_repr(self):
        s = ('size={size}, finding_masks={finding_masks}')
        return s.format(**self.__dict__)

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
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes - in_planes) // 2, planes - in_planes - (planes - in_planes) // 2), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        
        self.mask = Mask((1, planes, 1, 1), finding_masks)
        self.activation_applied = False

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.mask(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_cfg, out_cfg, num_classes=10, finding_masks=True):
        super(ResNet, self).__init__()
        self.in_planes = out_cfg[0]
        self.finding_masks = finding_masks
        
        # Initial convolution layer with configurable input/output channels
        self.conv1 = nn.Conv2d(in_cfg[0], out_cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_cfg[0])
        
        # ResNet layers with configurable channels
        self.layer1 = self._make_layer(block, in_cfg[1:4], out_cfg[1:4], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_cfg[4:7], out_cfg[4:7], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_cfg[7:10], out_cfg[7:10], num_blocks[2], stride=2)
        
        # Final classifier
        self.linear = nn.Linear(out_cfg[-1], num_classes)
        self.fc = self.linear  # برای سازگاری با مدل‌های پیش‌آموزش دیده
        
        # Hook-related attributes
        self.handlers = []
        self.masks_outputs = {}
        self.masks = {}
        
        self.apply(_weights_init)

    def _make_layer(self, block, in_cfg_slice, out_cfg_slice, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(in_cfg_slice[i], out_cfg_slice[i], stride, finding_masks=self.finding_masks))
            self.in_planes = out_cfg_slice[i] * block.expansion
        return nn.Sequential(*layers)

    def hook_mask(self, layer, mask_name):
        def hook_function(module, input, output):
            self.masks_outputs[mask_name] = output
        return layer.register_forward_hook(hook_function)

    def hook_masks(self):
        self.remove_hooks()
        ind = 0
        
        # Register hooks for all mask layers
        for layer_group in [self.layer1, self.layer2, self.layer3]:
            for block in layer_group:
                layer_name = f'mask.{ind}'
                self.handlers.append(self.hook_mask(block.mask, layer_name))
                ind += 1

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()
        self.handlers = []

    def get_masks(self):
        self.masks = {}
        ind = 0
        for layer_group in [self.layer1, self.layer2, self.layer3]:
            for block in layer_group:
                layer_name = f'mask.{ind}'
                self.masks[layer_name] = block.mask
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
        # Create a copy of the state dict to avoid modifying the original
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
                
            new_state_dict[new_key] = value
        
        # Load the weights with flexible matching
        self.load_state_dict(new_state_dict, strict=False)
        
        # Return information about missing/unexpected keys
        missing_keys, unexpected_keys = [], []
        loaded_keys = set(new_state_dict.keys())
        model_keys = set(self.state_dict().keys())
        
        missing_keys = list(model_keys - loaded_keys)
        unexpected_keys = list(loaded_keys - model_keys)
        
        return missing_keys, unexpected_keys


def resnet20(finding_masks, in_cfg, out_cfg, num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], in_cfg=in_cfg, out_cfg=out_cfg, num_classes=num_classes, finding_masks=finding_masks)


def resnet32(finding_masks, in_cfg, out_cfg, num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], in_cfg=in_cfg, out_cfg=out_cfg, num_classes=num_classes, finding_masks=finding_masks)


def resnet44(finding_masks, in_cfg, out_cfg, num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], in_cfg=in_cfg, out_cfg=out_cfg, num_classes=num_classes, finding_masks=finding_masks)


def resnet56(finding_masks, in_cfg, out_cfg, num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], in_cfg=in_cfg, out_cfg=out_cfg, num_classes=num_classes, finding_masks=finding_masks)


def resnet110(finding_masks, in_cfg, out_cfg, num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], in_cfg=in_cfg, out_cfg=out_cfg, num_classes=num_classes, finding_masks=finding_masks)


def test():
    in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]
    out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
    
    net = resnet20(finding_masks=True, in_cfg=in_cfg, out_cfg=out_cfg)
    net.hook_masks()
    masks = net.get_masks()
    
    print("Model Architecture:")
    print(net)
    print("\nNumber of masks:", len(masks))
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print("\nOutput shape:", y.shape)
    
    # Get active neuron counts
    counts = net.get_active_neuron_counts()
    print("\nActive neurons per layer:", counts)
    
    # Clean up hooks
    net.remove_hooks()

    return net


if __name__ == "__main__":
    model = test()
