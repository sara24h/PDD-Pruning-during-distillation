import datetime
import os
import sys
import argparse
import inspect
import warnings

# حذف هشدارهای CUDA
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # غیرفعال کردن لاگ‌های TensorFlow
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# وارد کردن ماژول‌های مورد نیاز
try:
    from args import args
except ImportError:
    # ساخت کلاس args جایگزین در صورت عدم وجود ماژول args
    class Args:
        pass
    args = Args()

from data.Data import CIFAR10, CIFAR100
from trainer.trainer import validate, train
from utils.utils import set_gpu, get_logger, Logger, set_random_seed

# بررسی وجود ماژول‌های موردنیاز و وارد کردن آن‌ها
try:
    from resnet_kd import resnet20, resnet56
except ImportError:
    from models.resnet_kd import resnet20, resnet56

try:
    from vgg_kd import cvgg11_bn, cvgg11_bn_small, cvgg16_bn
except ImportError:
    from models.vgg_kd import cvgg11_bn, cvgg11_bn_small, cvgg16_bn

class CustomLogger:
    """کلاس سفارشی برای لاگ‌گیری"""
    def __init__(self, log_file, console=sys.stdout):
        self.console = console
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.console.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.console.flush()
        self.log.flush()

def setup_arg_parser():
    """تنظیم تحلیل‌گر آرگومان‌های خط فرمان"""
    parser = argparse.ArgumentParser(description='PDD Fine-tuning')
    
    # تنظیمات اصلی
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--arch', type=str, default='resnet56', help='Teacher architecture')
    parser.add_argument('--arch_s', type=str, default='resnet20_small', 
                       choices=['resnet20_small', 'cvgg11_bn_small'],
                       help='Student architecture')
    parser.add_argument('--set', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100'],
                       help='Dataset name')
    
    # تنظیمات آموزش
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr_decay_step', type=str, default='5,8', help='LR decay steps')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every n epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch')
    
    # مسیرهای فایل
    parser.add_argument('--mask_path', type=str, default='', help='Path to mask file')
    parser.add_argument('--teacher_ckpt', type=str, default='', help='Path to teacher checkpoint')
    parser.add_argument('--save_dir', type=str, default='pretrained_model', help='Directory to save models')
    
    return parser

def get_model_arguments(model_func):
    """دریافت آرگومان‌های قابل قبول برای یک تابع مدل"""
    return inspect.getfullargspec(model_func).args

def load_checkpoint(model, path, device='cuda'):
    """بارگذاری چک‌پوینت با انعطاف‌پذیری بیشتر"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")
    
    try:
        checkpoint = torch.load(path, map_location=device)
        # بررسی اینکه آیا فایل شامل state_dict است یا خود وزن‌ها
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

def load_mask(path):
    """بارگذاری فایل mask با مدیریت خطا"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask file not found at {path}")
    
    try:
        return torch.load(path)
    except Exception as e:
        print(f"Error loading mask: {e}")
        raise

def load_vgg_model(student_model, teacher_state_dict, mask):
    """بارگذاری وزن‌های معلم به دانش‌آموز برای معماری VGG"""
    student_state = student_model.state_dict()
    last_select_index = None
    cnt = -1
    
    # تنظیم تعداد فیلتر در لایه آخر
    if hasattr(student_model, 'classifier') and len(student_model.classifier) > 1:
        if isinstance(student_model.classifier[1], nn.Linear) and 'layer_num' in mask:
            student_model.classifier[1] = nn.Linear(mask['layer_num'][-1], student_model.classifier[1].out_features)
    
    for name, module in student_model.named_modules():
        name = name.replace('module.', '')
        
        if isinstance(module, nn.Conv2d):
            cnt += 1
            weight_name = f"{name}.weight"
            
            if weight_name not in teacher_state_dict or weight_name not in student_state:
                continue
                
            teacher_weight = teacher_state_dict[weight_name]
            student_weight = student_state[weight_name]
            teacher_filters = teacher_weight.size(0)
            student_filters = student_weight.size(0)
            
            try:
                if teacher_filters != student_filters and cnt < len(mask['mask']):
                    # انتخاب فیلترهای مهم بر اساس mask
                    select_index = torch.argsort(mask['mask'][cnt])[teacher_filters - student_filters:]
                    select_index.sort()
                    
                    if last_select_index is not None and len(last_select_index) > 0:
                        for i_idx, i in enumerate(select_index):
                            for j_idx, j in enumerate(last_select_index):
                                if i_idx < student_weight.size(0) and j_idx < student_weight.size(1):
                                    student_state[weight_name][i_idx][j_idx] = teacher_weight[i][j]
                    else:
                        for i_idx, i in enumerate(select_index):
                            if i_idx < student_weight.size(0):
                                student_state[weight_name][i_idx] = teacher_weight[i]
                    
                    last_select_index = select_index
                elif last_select_index is not None and cnt < len(mask['mask']):
                    for i in range(teacher_filters):
                        for j_idx, j in enumerate(last_select_index):
                            if i < student_weight.size(0) and j_idx < student_weight.size(1):
                                student_state[weight_name][i][j_idx] = teacher_weight[i][j]
            except Exception as e:
                print(f"Error processing weight {weight_name}: {e}")
                continue
    
    student_model.load_state_dict(student_state)
    return student_model

def load_resnet_model(student_model, teacher_state_dict, mask, layer=20):
    """بارگذاری وزن‌های معلم به دانش‌آموز برای معماری ResNet"""
    student_state = student_model.state_dict()
    last_select_index = None
    cnt = -1
    
    # پیکربندی معماری ResNet20
    cfg = {20: [3, 3, 3]}
    current_cfg = cfg[layer]
    all_conv_weights = []
    
    for layer_idx, num_blocks in enumerate(current_cfg):
        layer_name = f'layer{layer_idx + 1}.'
        for block_idx in range(num_blocks):
            cnt += 1
            for conv_idx in range(2):
                conv_name = f'{layer_name}{block_idx}.conv{conv_idx + 1}'
                weight_name = f'{conv_name}.weight'
                all_conv_weights.append(weight_name)
                
                if weight_name not in teacher_state_dict or weight_name not in student_state:
                    continue
                    
                teacher_weight = teacher_state_dict[weight_name]
                student_weight = student_state[weight_name]
                teacher_filters = teacher_weight.size(0)
                student_filters = student_weight.size(0)
                
                try:
                    if teacher_filters != student_filters and cnt < len(mask['mask']):
                        # انتخاب فیلترهای مهم
                        select_index = torch.argsort(mask['mask'][cnt])[teacher_filters - student_filters:]
                        select_index.sort()
                        
                        if last_select_index is not None and len(last_select_index) > 0:
                            for i_idx, i in enumerate(select_index):
                                for j_idx, j in enumerate(last_select_index):
                                    if i_idx < student_weight.size(0) and j_idx < student_weight.size(1):
                                        student_state[weight_name][i_idx][j_idx] = teacher_weight[i][j]
                        else:
                            for i_idx, i in enumerate(select_index):
                                if i_idx < student_weight.size(0):
                                    student_state[weight_name][i_idx] = teacher_weight[i]
                        
                        last_select_index = select_index
                    elif last_select_index is not None and cnt < len(mask['mask']):
                        for i in range(teacher_filters):
                            for j_idx, j in enumerate(last_select_index):
                                if i < student_weight.size(0) and j_idx < student_weight.size(1):
                                    student_state[weight_name][i][j_idx] = teacher_weight[i][j]
                        last_select_index = None
                except Exception as e:
                    print(f"Error processing weight {weight_name}: {e}")
                    continue
    
    # کپی کردن وزن‌های باقیمانده
    for name, param in student_state.items():
        if name in teacher_state_dict and param.size() == teacher_state_dict[name].size():
            student_state[name] = teacher_state_dict[name]
    
    student_model.load_state_dict(student_state)
    return student_model

def main_worker(args):
    """تابع اصلی فاین‌تیون"""
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # تنظیم دیرکتوری ذخیره‌سازی
    save_dir = os.path.join(args.save_dir, args.arch_s, args.set)
    os.makedirs(save_dir, exist_ok=True)
    
    # راه‌اندازی لاگ‌گیری
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file = os.path.join(save_dir, f'finetune_log_{now}.log')
    sys.stdout = CustomLogger(log_file, sys.stdout)
    
    logger = get_logger(log_file)
    logger.info(f"Starting fine-tuning with configuration: {args}")
    
    # بارگذاری داده‌ها
    logger.info(f"Loading dataset: {args.set}")
    if args.set == 'cifar10':
        data = CIFAR10()
    elif args.set == 'cifar100':
        data = CIFAR100()
    else:
        raise ValueError(f"Unsupported dataset: {args.set}")
    
    # بارگذاری فایل mask
    if not args.mask_path:
        args.mask_path = os.path.join(save_dir, 'mask.pt')
    logger.info(f"Loading mask from: {args.mask_path}")
    mask = load_mask(args.mask_path)
    
    # ساخت و بارگذاری مدل‌ها
    logger.info(f"Creating teacher model: {args.arch}")
    logger.info(f"Creating student model: {args.arch_s}")
    
    if args.arch_s == 'cvgg11_bn_small':
        # مدل‌های مربوط به VGG
        teacher_model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
        student_model = cvgg11_bn_small(num_classes=args.num_classes, batch_norm=True)
        
        # بارگذاری چک‌پوینت مدل معلم
        if not args.teacher_ckpt:
            args.teacher_ckpt = os.path.join(save_dir, 'teacher_model.pt')
        logger.info(f"Loading teacher checkpoint from: {args.teacher_ckpt}")
        load_checkpoint(teacher_model, args.teacher_ckpt, device)
        
        # تنظیم تعداد فیلتر در لایه آخر
        if hasattr(student_model, 'classifier') and len(student_model.classifier) > 1:
            student_model.classifier[1] = nn.Linear(mask['layer_num'][-1], 512)
        
        # کپی کردن وزن‌ها از معلم به دانش‌آموز
        logger.info("Transferring weights from teacher to student (VGG)")
        student_model = load_vgg_model(student_model, teacher_model.state_dict(), mask)
    
    elif args.arch_s == 'resnet20_small':
        # مدل‌های مربوط به ResNet
        teacher_model = resnet56(num_classes=args.num_classes)
        student_model = resnet20(num_classes=args.num_classes)
        
        # بارگذاری چک‌پوینت مدل معلم
        if not args.teacher_ckpt:
            args.teacher_ckpt = os.path.join(save_dir, 'teacher_model.pt')
        logger.info(f"Loading teacher checkpoint from: {args.teacher_ckpt}")
        load_checkpoint(teacher_model, args.teacher_ckpt, device)
        
        # کپی کردن وزن‌ها از معلم به دانش‌آموز
        logger.info("Transferring weights from teacher to student (ResNet)")
        student_model = load_resnet_model(student_model, teacher_model.state_dict(), mask, layer=20)
    
    # تنظیم GPU
    teacher_model = set_gpu(args, teacher_model)
    student_model = set_gpu(args, student_model)
    
    # فریز کردن وزن‌های معلم
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    logger.info(f"Teacher model architecture:\n{teacher_model}")
    logger.info(f"Student model architecture:\n{student_model}")
    
    # تعریف تابع زیان و بهینه‌ساز
    criterion = nn.CrossEntropyLoss().to(device)
    
    # اعتبارسنجی مدل معلم
    logger.info("Validating teacher model...")
    teacher_acc1, teacher_acc5 = validate(data.val_loader, teacher_model, criterion, args)
    logger.info(f"Teacher model validation accuracy: {teacher_acc1:.2f}%")
    
    # تنظیم بهینه‌ساز
    optimizer = optim.SGD(student_model.parameters(), 
                         lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay)
    
    # تنظیم کاهش نرخ یادگیری
    lr_decay_steps = [int(x) for x in args.lr_decay_step.split(',')]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_steps, gamma=0.1)
    
    best_acc1 = 0.0
    best_acc5 = 0.0
    
    # شروع فاین‌تیون
    logger.info(f"Starting fine-tuning for {args.epochs} epochs...")
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # آموزش
        logger.info("Training...")
        train_acc1, train_acc5 = train(data.train_loader, student_model, criterion, optimizer, epoch, args)
        logger.info(f"Train accuracy: {train_acc1:.2f}% (top-1), {train_acc5:.2f}% (top-5)")
        
        # اعتبارسنجی
        logger.info("Validating...")
        acc1, acc5 = validate(data.val_loader, student_model, criterion, args)
        logger.info(f"Validation accuracy: {acc1:.2f}% (top-1), {acc5:.2f}% (top-5)")
        
        # به‌روزرسانی نرخ یادگیری
        scheduler.step()
        
        # ذخیره بهترین مدل
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_acc5 = acc5
            logger.info(f"New best accuracy: {best_acc1:.2f}%")
            
            # ذخیره مدل بهترین
            best_model_path = os.path.join(save_dir, f'best_model_T_{args.arch}_S_{args.arch_s}_{args.set}.pt')
            torch.save(student_model.state_dict(), best_model_path)
            logger.info(f"Best model saved to {best_model_path}")
        
        # ذخیره چک‌پوینت
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # ذخیره مدل نهایی
    final_model_path = os.path.join(save_dir, f'final_model_T_{args.arch}_S_{args.arch_s}_{args.set}.pt')
    torch.save(student_model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    logger.info(f"Best validation accuracy: {best_acc1:.2f}% (top-1), {best_acc5:.2f}% (top-5)")
    
    return best_acc1, best_acc5

def main():
    """تابع اصلی برنامه"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # تنظیم seed برای تکرارپذیری
    if args.random_seed is not None:
        set_random_seed(args.random_seed)
    
    # اجرای فاین‌تیون
    best_acc1, best_acc5 = main_worker(args)
    
    print(f"Fine-tuning completed. Best accuracy: {best_acc1:.2f}% (top-1), {best_acc5:.2f}% (top-5)")

if __name__ == "__main__":
    # مثال استفاده:
    # python finetune.py --gpu 0 --arch_s resnet20_small --set cifar10 --lr 0.01 --batch_size 128 --weight_decay 0.005 --epochs 10 --lr_decay_step 5,8 --num_classes 10 --arch resnet56 --mask_path path/to/mask.pt --teacher_ckpt path/to/teacher.pt
    main()
