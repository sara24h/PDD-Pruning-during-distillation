# borrow from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']



class Mask(nn.Module):
    def __init__(self, size=(1, 128, 1, 1), finding_masks=True):
        super(Mask, self).__init__()
        self.mask = nn.Parameter(torch.randn(size, requires_grad=True)*0.001)  # 0.001让权重变敏感
        self.size = size
        self.finding_masks = finding_masks

    def forward(self, x):
        # True
        if self.finding_masks:
            out_forward = torch.sign(self.mask)
            mask1 = self.mask < -1
            mask2 = self.mask < 0
            mask3 = self.mask < 1
            out1 = (-1) * mask1.type(torch.float32) + (self.mask * self.mask + 2 * self.mask) * (1 - mask1.type(torch.float32))
            out2 = out1 * mask2.type(torch.float32) + (-self.mask * self.mask + 2 * self.mask) * (1 - mask2.type(torch.float32))
            out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
            out = out_forward.detach() - out3.detach() + out3
            return (out+1)/2 * x
        # False
        else:
            return x

    def extra_repr(self):
        s = ('size={size}')
        return s.format(**self.__dict__)


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
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

    def __init__(self, finding_masks, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if stride != 1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (
                                    0, 0, 0, 0, (planes - in_planes) // 2, planes - in_planes - (planes - in_planes) // 2),
                                    "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (
                                    0, 0, 0, 0, (planes - in_planes) // 2, planes - in_planes - (planes - in_planes) // 2),
                                    "constant", 0))

        self.mask = Mask((1, planes * self.expansion, 1, 1), finding_masks)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.mask(out)
        return out


class ResNet(nn.Module):
    def __init__(self, finding_masks, block, num_blocks, in_cfg, out_cfg, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_cfg[0], out_cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_cfg[0])
        # self.mask = Mask((1, out_cfg[0], 1, 1), finding_masks)
        self.layer1 = self._make_layer(finding_masks, block, in_cfg, out_cfg, 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(finding_masks, block, in_cfg, out_cfg, 4, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(finding_masks, block, in_cfg, out_cfg, 7, num_blocks[2], stride=2)
        self.linear = nn.Linear(out_cfg[-1], num_classes)

        self.apply(_weights_init)
        self.handlers = []
        self.masks_outputs = {}
        self.origs_outputs = {}
        self.masks = {}
        self.get_masks()

    def _make_layer(self, finding_masks, block, in_cfg, out_cfg, i, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        count = 0
        for stride in strides:
            layers.append(block(finding_masks, in_cfg[i+count], out_cfg[i+count], stride))
            count += 1

        return nn.Sequential(*layers)

    # get masks outputs
    def hook_mask(self, layer, mask_name):
        def hook_function(module, input, output):
            self.masks_outputs[mask_name] = output

        return layer.register_forward_hook(hook_function)

    def hook_masks(self):
        ind = 0
        # layer_name = 'mask.' + str(ind)
        # ind += 1
        # self.handlers.append(self.hook_mask(self.mask, layer_name))

        for name in self.layer1._modules:
            layer = self.layer1._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.handlers.append(self.hook_mask(layer['mask'], layer_name))

        for name in self.layer2._modules:
            layer = self.layer2._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.handlers.append(self.hook_mask(layer['mask'], layer_name))

        for name in self.layer3._modules:
            layer = self.layer3._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.handlers.append(self.hook_mask(layer['mask'], layer_name))

    def get_masks_outputs(self):
        return self.masks_outputs

    # remove hooks
    def remove_hooks(self):
        while len(self.handlers) > 0:
            self.handlers.pop().remove()

    # get masks weights
    def get_masks(self):
        ind = 0
        # layer_name = 'mask.' + str(ind)
        # ind += 1
        # self.masks[layer_name] = self.mask

        for name in self.layer1._modules:
            layer = self.layer1._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.masks[layer_name] = layer['mask']

        for name in self.layer2._modules:
            layer = self.layer2._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.masks[layer_name] = layer['mask']

        for name in self.layer3._modules:
            layer = self.layer3._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.masks[layer_name] = layer['mask']

        return self.masks

    # save masks
    def save_masks(self, path):
        tmp_masks = {}
        masks = self.get_masks()
        for key in masks.keys():
            tmp_masks[key] = masks[key].mask
        torch.save(tmp_masks, path)
        return path

    # load masks
    def load_masks(self, path):
        trained_masks = torch.load(path)
        ind = 0
        # layer_name = 'mask.' + str(ind)
        # ind += 1
        # self.mask.data = trained_masks[layer_name].data

        for name in self.layer1._modules:
            layer = self.layer1._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            layer['mask'].data = trained_masks[layer_name].data

        for name in self.layer2._modules:
            layer = self.layer2._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            layer['mask'].data = trained_masks[layer_name].data

        for name in self.layer3._modules:
            layer = self.layer3._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            layer['mask'].data = trained_masks[layer_name].data

        return path

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.mask(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(finding_masks, in_cfg, out_cfg, num_classes=10):
    return ResNet(finding_masks, BasicBlock, [3, 3, 3], in_cfg=in_cfg, out_cfg=out_cfg, num_classes=num_classes)


def resnet32(finding_masks, in_cfg, out_cfg, num_classes=10):
    return ResNet(finding_masks, BasicBlock, [5, 5, 5], in_cfg, out_cfg, num_classes=num_classes)


def resnet44(finding_masks, in_cfg, out_cfg, num_classes=10):
    return ResNet(finding_masks, BasicBlock, [7, 7, 7], in_cfg, out_cfg, num_classes=num_classes)


def resnet56(finding_masks, in_cfg, out_cfg, num_classes=10):
    return ResNet(finding_masks, BasicBlock, [9, 9, 9], in_cfg, out_cfg, num_classes=num_classes)


def resnet110(finding_masks, in_cfg, out_cfg, num_classes=10):
    return ResNet(finding_masks, BasicBlock, [18, 18, 18], in_cfg, out_cfg, num_classes=num_classes)



if __name__ == "__main__":
    in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]
    out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
    model = resnet20(finding_masks=True, in_cfg=in_cfg, out_cfg=out_cfg)
    print(model)
    input = torch.rand((2, 3, 32, 32))
    print(model(input))
    model.hook_masks()

    masks_weights = model.get_masks()
    print(masks_weights['mask.0'].mask)
