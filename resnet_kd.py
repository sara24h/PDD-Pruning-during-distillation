import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['resnet20_auto', 'resnet56_auto']


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class DynamicMask(nn.Module):
    """
    ماسک دینامیک که خودش تصمیم می‌گیرد کدام کانال‌ها حذف شوند
    """
    def __init__(self, channels):
        super(DynamicMask, self).__init__()
        # مقداردهی اولیه با عدد مثبت (همه کانال‌ها فعال)
        self.mask = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.channels = channels
        
    def forward(self, x):
        """
        در training: از ماسک کامل استفاده می‌کند
        در inference: می‌توان با threshold کانال‌ها را حذف کرد
        """
        return x * self.mask
    
    def get_active_channels(self, threshold=0.0):
        """
        تعداد کانال‌های فعال را محاسبه می‌کند
        threshold: آستانه برای تعیین کانال فعال (پیش‌فرض: 0)
        """
        with torch.no_grad():
            binary_mask = (self.mask.squeeze() > threshold).float()
            return int(binary_mask.sum().item())
    
    def get_binary_mask(self, threshold=0.0):
        """
        ماسک باینری را برمی‌گرداند (0 یا 1)
        """
        with torch.no_grad():
            return (self.mask.squeeze() > threshold).float()


class BasicBlock_AutoPrune(nn.Module):
    """
    BasicBlock با قابلیت هرس خودکار
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B', use_pruning=False):
        super(BasicBlock_AutoPrune, self).__init__()
        self.use_pruning = use_pruning
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # ماسک دینامیک برای هرس
        if self.use_pruning:
            self.mask = DynamicMask(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], 
                                  (0, 0, 0, 0, planes//4, planes//4), 
                                  "constant", 0)
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, 
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # اعمال ماسک در صورت فعال بودن pruning
        if self.use_pruning:
            out = self.mask(out)
            
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_AutoPrune(nn.Module):
    """
    ResNet با قابلیت هرس خودکار کانال‌ها
    نیازی به تعیین دستی in_cfg و out_cfg ندارد
    """
    def __init__(self, block, num_blocks, num_classes=10, 
                 option='B', use_pruning=False):
        super(ResNet_AutoPrune, self).__init__()
        self.in_planes = 16
        self.option = option
        self.use_pruning = use_pruning
        self.mask_hooks = []
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, 
                              padding=1, bias=False)
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
                              option=self.option, 
                              use_pruning=self.use_pruning))
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
    
    def get_pruning_stats(self, threshold=0.0):
       
        stats = {
            'layer_names': [],
            'total_channels': [],
            'active_channels': [],
            'pruned_channels': [],
            'sparsity': []
        }
        
        for name, module in self.named_modules():
            if isinstance(module, DynamicMask):
                total = module.channels
                active = module.get_active_channels(threshold)
                pruned = total - active
                sparsity = (pruned / total) * 100
                
                stats['layer_names'].append(name)
                stats['total_channels'].append(total)
                stats['active_channels'].append(active)
                stats['pruned_channels'].append(pruned)
                stats['sparsity'].append(sparsity)
        
        return stats
    
    def print_pruning_stats(self, threshold=0.0):
        """
        آمار هرس را چاپ می‌کند
        """
        stats = self.get_pruning_stats(threshold)
        
        print("\n" + "="*100)
        print("Automatic Pruning Statistics (Threshold = {:.2f})".format(threshold))
        print("="*100)
        print(f"{'Layer Name':<40} {'Total':<10} {'Active':<10} {'Pruned':<10} {'Sparsity':<10}")
        print("-"*100)
        
        total_all = 0
        active_all = 0
        
        for i in range(len(stats['layer_names'])):
            layer = stats['layer_names'][i]
            total = stats['total_channels'][i]
            active = stats['active_channels'][i]
            pruned = stats['pruned_channels'][i]
            sparsity = stats['sparsity'][i]
            
            print(f"{layer:<40} {total:<10} {active:<10} {pruned:<10} {sparsity:<10.2f}%")
            
            total_all += total
            active_all += active
        
        print("-"*100)
        overall_sparsity = ((total_all - active_all) / total_all) * 100
        print(f"{'OVERALL':<40} {total_all:<10} {active_all:<10} "
              f"{total_all - active_all:<10} {overall_sparsity:<10.2f}%")
        print("="*100 + "\n")
        
        return stats
    
    def extract_pruned_architecture(self, threshold=0.0):
        """
        معماری هرس‌شده را استخراج می‌کند (برای ساخت مدل کوچک)
        
        Returns:
            dict: شامل in_cfg و out_cfg
        """
        in_cfg = [3]  # کانال ورودی اولیه (RGB)
        out_cfg = []
        
        for name, module in self.named_modules():
            if isinstance(module, DynamicMask):
                active_channels = module.get_active_channels(threshold)
                out_cfg.append(active_channels)
                
                # کانال ورودی لایه بعدی = کانال خروجی لایه فعلی
                if len(out_cfg) < 9:  # تا 9 لایه (3 لایه × 3 بلوک)
                    in_cfg.append(active_channels)
        
        return {
            'in_cfg': in_cfg,
            'out_cfg': out_cfg,
            'threshold': threshold
        }
    
    def apply_pruning(self, threshold=0.0):
        """
        ماسک‌ها را به صورت باینری (0 یا 1) تبدیل می‌کند
        این تابع را بعد از آموزش صدا بزنید
        """
        print(f"\nApplying binary pruning with threshold = {threshold}...")
        
        for module in self.modules():
            if isinstance(module, DynamicMask):
                with torch.no_grad():
                    binary_mask = module.get_binary_mask(threshold)
                    module.mask.data = binary_mask.view_as(module.mask)
        
        print("✓ Pruning applied successfully!")
        self.print_pruning_stats(threshold=threshold)


def resnet20_auto(num_classes=10, option='B', use_pruning=False):

    return ResNet_AutoPrune(BasicBlock_AutoPrune, [3, 3, 3], 
                           num_classes=num_classes, 
                           option=option, 
                           use_pruning=use_pruning)


def resnet56_auto(num_classes=10, option='B', use_pruning=False):
   
    return ResNet_AutoPrune(BasicBlock_AutoPrune, [9, 9, 9], 
                           num_classes=num_classes, 
                           option=option, 
                           use_pruning=use_pruning)


# ============================================================================
# Test Script
# ============================================================================
if __name__ == "__main__":
    print("Testing Automatic Pruning ResNet...\n")
    
    # 1. ساخت مدل با pruning
    model = resnet20_auto(num_classes=10, option='B', use_pruning=True)
    model.eval()
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 2. تست forward pass
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"Output shape: {output.shape}\n")
    
    # 3. شبیه‌سازی آموزش (ماسک‌ها را تصادفی تغییر می‌دهیم)
    print("Simulating training (randomizing masks)...")
    for module in model.modules():
        if isinstance(module, DynamicMask):
            # مقادیر تصادفی بین -1 و 1
            module.mask.data = torch.randn_like(module.mask) * 0.5
    
    # 4. نمایش آمار با threshold‌های مختلف
    print("\n" + "="*100)
    print("Testing different thresholds:")
    print("="*100)
    
    for threshold in [0.0, 0.1, 0.2, 0.3]:
        stats = model.print_pruning_stats(threshold=threshold)
    
    # 5. استخراج معماری هرس‌شده
    print("\nExtracting pruned architecture...")
    arch_config = model.extract_pruned_architecture(threshold=0.2)
    print(f"\nPruned Architecture (threshold=0.2):")
    print(f"  in_cfg:  {arch_config['in_cfg']}")
    print(f"  out_cfg: {arch_config['out_cfg']}")
    
    # 6. اعمال هرس نهایی
    model.apply_pruning(threshold=0.2)
    
    # 7. تست forward pass بعد از pruning
    output_pruned = model(x)
    print(f"\nOutput shape after pruning: {output_pruned.shape}")
    print("\n✓ All tests passed!")
