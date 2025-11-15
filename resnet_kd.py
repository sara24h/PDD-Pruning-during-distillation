import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']


class Mask(nn.Module):
    """
    کلاس Mask برای یادگیری ماسک‌های باینری روی کانال‌ها
    این ماسک‌ها مشخص می‌کنند کدام نورون‌ها فعال بمانند و کدام‌ها هرس شوند
    """
    def __init__(self, size=(1, 16, 1, 1), finding_masks=True):
        super(Mask, self).__init__()
        # پارامتر قابل یادگیری که تصمیم می‌گیرد هر کانال فعال باشد یا خیر
        self.mask = nn.Parameter(torch.randn(size, requires_grad=True) * 0.001)
        self.size = size
        self.finding_masks = finding_masks
        # بافر برای ذخیره ماسک باینری نهایی (0 یا 1)
        self.register_buffer('binary_mask', torch.ones(size))

    def forward(self, x):
        if self.finding_masks:
            # محاسبه ماسک باینری با استفاده از یک تابع صاف (smooth) که مشتق‌پذیر است
            out_forward = torch.sign(self.mask)
            mask1 = self.mask < -1
            mask2 = self.mask < 0
            mask3 = self.mask < 1
            out1 = (-1) * mask1.type(torch.float32) + (self.mask * self.mask + 2 * self.mask) * (1 - mask1.type(torch.float32))
            out2 = out1 * mask2.type(torch.float32) + (-self.mask * self.mask + 2 * self.mask) * (1 - mask2.type(torch.float32))
            out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
            # ترفند Straight-Through Estimator برای بک‌پراپ
            out = out_forward.detach() - out3.detach() + out3
            # تبدیل به مقادیر 0 و 1
            self.binary_mask = ((out + 1) / 2).detach()
            return self.binary_mask * x
        else:
            # در حالت استنتاج، فقط از ماسک باینری ثابت استفاده می‌کنیم
            return self.binary_mask * x

    def apply_threshold(self, threshold=0.5):
        """تبدیل ماسک یادگیری شده به ماسک باینری بر اساس آستانه"""
        self.binary_mask = (self.mask > threshold).float().detach()
        self.finding_masks = False
        return self.binary_mask.sum().item()


def _weights_init(m):
    """مقداردهی اولیه وزن‌ها با استفاده از روش Kaiming"""
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    """لایه‌ای برای اعمال یک تابع دلخواه"""
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    """
    بلوک پایه ResNet برای CIFAR-10
    این بلوک شامل دو لایه کانولوشنی، نرمال‌سازی دسته‌ای، و یک اتصال باقیمانده است
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', finding_masks=True):
        super(BasicBlock, self).__init__()
        # اولین لایه کانولوشنی
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # دومین لایه کانولوشنی
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # مسیر میانبر (shortcut connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                # گزینه A: استفاده از padding برای تطبیق ابعاد (بدون پارامتر اضافی)
                self.shortcut = LambdaLayer(lambda x:
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes - in_planes) // 2, planes - in_planes - (planes - in_planes) // 2), "constant", 0))
            elif option == 'B':
                # گزینه B: استفاده از کانولوشن 1x1 برای تطبیق ابعاد
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        
        # ماسک برای هرس کانال‌ها
        self.mask = Mask((1, planes, 1, 1), finding_masks)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        # اعمال ماسک روی خروجی
        out = self.mask(out)
        return out


class ResNet(nn.Module):
    """
    معماری ResNet برای CIFAR-10 با قابلیت هرس
    این کلاس از in_cfg و out_cfg برای تنظیم تعداد کانال‌ها در هر لایه استفاده می‌کند
    """
    def __init__(self, block, num_blocks, num_classes=10, finding_masks=True, in_cfg=None, out_cfg=None):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.finding_masks = finding_masks
        
        # تنظیمات پیش‌فرض برای کانال‌ها اگر ارائه نشده باشند
        if in_cfg is None:
            in_cfg = [[16] * num_blocks[0], [32] * num_blocks[1], [64] * num_blocks[2]]
        if out_cfg is None:
            out_cfg = [[16] * num_blocks[0], [32] * num_blocks[1], [64] * num_blocks[2]]
        
        self.in_cfg = in_cfg
        self.out_cfg = out_cfg
        
        # لایه کانولوشنی اولیه
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # سه مرحله اصلی ResNet با استفاده از تنظیمات کانال
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, 
                                       finding_masks=finding_masks, 
                                       in_cfg=in_cfg[0], 
                                       out_cfg=out_cfg[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, 
                                       finding_masks=finding_masks, 
                                       in_cfg=in_cfg[1], 
                                       out_cfg=out_cfg[1])
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, 
                                       finding_masks=finding_masks, 
                                       in_cfg=in_cfg[2], 
                                       out_cfg=out_cfg[2])
        
        # لایه کاملاً متصل نهایی
        # آخرین کانال از آخرین لایه را به عنوان ورودی استفاده می‌کنیم
        final_channels = out_cfg[2][-1] if out_cfg else 64
        self.linear = nn.Linear(final_channels, num_classes)
        
        # برای سازگاری با مدل‌های پیش‌آموزش دیده که از fc استفاده می‌کنند
        self.fc = self.linear
        
        # مقداردهی اولیه وزن‌ها
        self.apply(_weights_init)
        
        # برای ذخیره هوک‌ها و خروجی‌های ماسک
        self.handlers = []
        self.masks_outputs = {}
        self.masks = {}

    def _make_layer(self, block, planes, num_blocks, stride, finding_masks=True, in_cfg=None, out_cfg=None):
        """
        ساخت یک مرحله (stage) از شبکه ResNet
        هر مرحله شامل چندین بلوک پایه است
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        for i, stride in enumerate(strides):
            # تعداد کانال‌های ورودی و خروجی از تنظیمات
            in_planes = in_cfg[i] if in_cfg else self.in_planes
            out_planes = out_cfg[i] if out_cfg else planes
            
            # ساخت بلوک با تنظیمات مشخص شده
            layers.append(block(in_planes, out_planes, stride, finding_masks=finding_masks))
            
            # به‌روزرسانی تعداد کانال‌های ورودی برای بلوک بعدی
            self.in_planes = out_planes * block.expansion
            
        return nn.Sequential(*layers)

    def hook_mask(self, layer, mask_name):
        """ثبت یک هوک برای دریافت خروجی یک لایه ماسک"""
        def hook_function(module, input, output):
            self.masks_outputs[mask_name] = output
        return layer.register_forward_hook(hook_function)

    def hook_masks(self):
        """ثبت هوک برای تمام لایه‌های ماسک در شبکه"""
        self.remove_hooks()
        
        ind = 0
        # هوک کردن ماسک‌های layer1
        for name, layer in self.layer1.named_children():
            layer_name = f'mask.{ind}'
            self.handlers.append(self.hook_mask(layer.mask, layer_name))
            ind += 1
        
        # هوک کردن ماسک‌های layer2
        for name, layer in self.layer2.named_children():
            layer_name = f'mask.{ind}'
            self.handlers.append(self.hook_mask(layer.mask, layer_name))
            ind += 1
        
        # هوک کردن ماسک‌های layer3
        for name, layer in self.layer3.named_children():
            layer_name = f'mask.{ind}'
            self.handlers.append(self.hook_mask(layer.mask, layer_name))
            ind += 1

    def remove_hooks(self):
        """حذف تمام هوک‌های ثبت شده"""
        for handler in self.handlers:
            handler.remove()
        self.handlers.clear()
        self.masks_outputs.clear()

    def get_masks(self):
        """دریافت تمام ماسک‌های موجود در شبکه"""
        self.masks = {}
        ind = 0
        
        # جمع‌آوری ماسک‌ها از layer1
        for name, layer in self.layer1.named_children():
            layer_name = f'mask.{ind}'
            self.masks[layer_name] = layer.mask
            ind += 1
        
        # جمع‌آوری ماسک‌ها از layer2
        for name, layer in self.layer2.named_children():
            layer_name = f'mask.{ind}'
            self.masks[layer_name] = layer.mask
            ind += 1
        
        # جمع‌آوری ماسک‌ها از layer3
        for name, layer in self.layer3.named_children():
            layer_name = f'mask.{ind}'
            self.masks[layer_name] = layer.mask
            ind += 1
            
        return self.masks

    def get_active_neuron_counts(self):
        """شمارش تعداد نورون‌های فعال در هر لایه"""
        active_counts = []
        masks = self.get_masks()
        for key in sorted(masks.keys()):
            mask = masks[key]
            if hasattr(mask, 'binary_mask'):
                active_count = mask.binary_mask.sum().item()
                active_counts.append(int(active_count))
        return active_counts

    def forward(self, x):
        """پاس رو به جلو در شبکه"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # میانگین‌گیری فضایی
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def load_pretrained_weights(self, state_dict):
        """
        بارگذاری وزن‌های پیش‌آموزش دیده با مدیریت اختلافات نام‌گذاری
        این متد سعی می‌کند نام‌های مختلف را با هم تطبیق دهد
        """
        new_state_dict = {}
        
        for key, value in state_dict.items():
            new_key = key
            
            # مدیریت حالتی که مدل با DataParallel wrap شده باشد
            if new_key.startswith('module.'):
                new_key = new_key[7:]
                
            # تطبیق نام‌های fc و linear
            if new_key.startswith('fc.'):
                new_key = new_key.replace('fc.', 'linear.', 1)
                
            # تطبیق نام‌های shortcut و downsample
            if 'downsample.0' in new_key:
                new_key = new_key.replace('downsample.0', 'shortcut.0')
            if 'downsample.1' in new_key:
                new_key = new_key.replace('downsample.1', 'shortcut.1')
            if 'downsample.conv' in new_key:
                new_key = new_key.replace('downsample.conv', 'shortcut.0')
            if 'downsample.bn' in new_key:
                new_key = new_key.replace('downsample.bn', 'shortcut.1')
                
            new_state_dict[new_key] = value
        
        # تلاش برای بارگذاری با strict=True
        try:
            self.load_state_dict(new_state_dict, strict=True)
            print("✓ Model weights loaded successfully with strict=True")
            return [], []
        except RuntimeError as e:
            # اگر با strict=True کار نکرد، از strict=False استفاده می‌کنیم
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


# توابع کمکی برای ساخت مدل‌های مختلف
# حالا همه این توابع از in_cfg و out_cfg پشتیبانی می‌کنند

def resnet20(num_classes=10, finding_masks=True, in_cfg=None, out_cfg=None):
    """ResNet-20 برای CIFAR-10"""
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, 
                  finding_masks=finding_masks, in_cfg=in_cfg, out_cfg=out_cfg)


def resnet32(num_classes=10, finding_masks=True, in_cfg=None, out_cfg=None):
    """ResNet-32 برای CIFAR-10"""
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, 
                  finding_masks=finding_masks, in_cfg=in_cfg, out_cfg=out_cfg)


def resnet44(num_classes=10, finding_masks=True, in_cfg=None, out_cfg=None):
    """ResNet-44 برای CIFAR-10"""
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, 
                  finding_masks=finding_masks, in_cfg=in_cfg, out_cfg=out_cfg)


def resnet56(num_classes=10, finding_masks=True, in_cfg=None, out_cfg=None):
    """ResNet-56 برای CIFAR-10"""
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, 
                  finding_masks=finding_masks, in_cfg=in_cfg, out_cfg=out_cfg)


def resnet110(num_classes=10, finding_masks=True, in_cfg=None, out_cfg=None):
    """ResNet-110 برای CIFAR-10"""
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, 
                  finding_masks=finding_masks, in_cfg=in_cfg, out_cfg=out_cfg)


# تست عملکرد
if __name__ == "__main__":
    import numpy as np
    
    print("=" * 80)
    print("تست مدل ResNet با قابلیت هرس")
    print("=" * 80)
    
    # تست 1: ساخت مدل استاندارد بدون هرس
    print("\n1. ساخت مدل استاندارد ResNet20...")
    model = resnet20(finding_masks=True)
    print("✓ ResNet20 model created successfully")
    
    # تست 2: پاس رو به جلو
    print("\n2. تست forward pass...")
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"✓ Forward pass successful. Output shape: {y.shape}")
    
    # تست 3: تست با تنظیمات سفارشی (شبیه‌سازی هرس)
    print("\n3. ساخت مدل با تنظیمات سفارشی (هرس شده)...")
    # شبیه‌سازی یک مدل هرس شده که برخی کانال‌ها کمتر دارد
    in_cfg_custom = [[16, 14, 12], [32, 28, 24], [64, 56, 48]]
    out_cfg_custom = [[14, 12, 10], [28, 24, 20], [56, 48, 40]]
    model_pruned = resnet20(finding_masks=True, in_cfg=in_cfg_custom, out_cfg=out_cfg_custom)
    print("✓ Pruned ResNet20 model created successfully")
    
    # تست 4: پاس رو به جلو با مدل هرس شده
    print("\n4. تست forward pass با مدل هرس شده...")
    y_pruned = model_pruned(x)
    print(f"✓ Forward pass successful. Output shape: {y_pruned.shape}")
    
    # تست 5: هوک‌های mask
    print("\n5. تست هوک‌های mask...")
    model.hook_masks()
    masks = model.get_masks()
    print(f"✓ Number of mask layers: {len(masks)}")
    
    # تست 6: پاس رو به جلو با هوک‌ها
    print("\n6. تست forward pass با هوک‌ها...")
    y = model(x)
    print(f"✓ Forward pass with hooks successful. Captured {len(model.masks_outputs)} mask outputs")
    
    # تست 7: شمارش نورون‌های فعال
    print("\n7. شمارش نورون‌های فعال...")
    counts = model.get_active_neuron_counts()
    print(f"✓ Active neurons per layer: {counts}")
    print(f"✓ Total active neurons: {np.sum(counts)}")
    
    # تست 8: پاک کردن هوک‌ها
    print("\n8. پاک کردن هوک‌ها...")
    model.remove_hooks()
    print("✓ Hooks removed successfully")
    
    # تست 9: محاسبه تعداد پارامترها
    print("\n9. محاسبه تعداد پارامترها...")
    total_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in model_pruned.parameters())
    print(f"✓ Standard model parameters: {total_params:,}")
    print(f"✓ Pruned model parameters: {pruned_params:,}")
    print(f"✓ Parameter reduction: {(1 - pruned_params/total_params)*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("همه تست‌ها با موفقیت انجام شد!")
    print("=" * 80)
