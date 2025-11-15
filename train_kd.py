import os
import sys
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from args import args
import datetime
from data.Data import CIFAR10, CIFAR100
from model.VGG_cifar import cvgg16_bn, cvgg19_bn
from model.samll_resnet import resnet56, resnet110
from resnet_kd import resnet20
from trainer.trainer import validate, train, train_KD
from utils.utils import set_random_seed, set_gpu, Logger, get_logger, get_lr
from vgg_kd import cvgg11_bn
import torch.nn.functional as F

def main():
    print(args)
    sys.stdout = Logger('print process.log', sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)


def main_worker(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('pretrained_model/' + args.arch + '/' + args.set):
        os.makedirs('pretrained_model/' + args.arch + '/' + args.set, exist_ok=True)
    logger = get_logger('pretrained_model/' + args.arch + '/' + args.set + '/logger' + now + '.log')
    logger.info(args.arch)
    logger.info(args.set)
    logger.info(args.batch_size)
    logger.info(args.weight_decay)
    logger.info(args.lr)
    logger.info(args.epochs)
    logger.info(args.lr_decay_step)
    logger.info(args.num_classes)

    # ===== ساخت مدل دانش‌آموز (Student Model) =====
    if args.arch_s == 'cvgg11_bn':
        model_s = cvgg11_bn(finding_masks=True, num_classes=args.num_classes, batch_norm=True)
    elif args.arch_s == 'resnet20':
        in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]
        out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
        model_s = resnet20(finding_masks=True, in_cfg=in_cfg, out_cfg=out_cfg, num_classes=args.num_classes)
    else:
        raise ValueError(f"معماری دانش‌آموز نامعتبر: {args.arch_s}")

    # ===== ساخت مدل معلم (Teacher Model) و بارگذاری چک‌پوینت =====
    ckpt = None  # مقداردهی اولیه برای جلوگیری از UnboundLocalError
    
    if args.arch == 'cvgg16_bn':
        model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar10/scores.pt', 
                                map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar100/scores.pt', 
                                map_location='cuda:%d' % args.gpu)
    
    elif args.arch == 'cvgg19_bn':
        model = cvgg19_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar10/scores.pt', 
                                map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar100/scores.pt', 
                                map_location='cuda:%d' % args.gpu)
    
    elif args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                # استفاده از چک‌پوینت جدید از GitHub برای CIFAR-10
                checkpoint_path = '/public/ly/Dynamic_Graph_Construction/pretrained_model/cifar10_resnet56-187c023a.pt'
                
                # بررسی وجود فایل چک‌پوینت
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(
                        f"چک‌پوینت ResNet-56 در مسیر {checkpoint_path} یافت نشد.\n"
                        f"لطفاً فایل را از لینک زیر دانلود کنید:\n"
                        f"https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt"
                    )
                
                logger.info(f"در حال بارگذاری چک‌پوینت ResNet-56 از GitHub...")
                ckpt_loaded = torch.load(checkpoint_path, map_location='cuda:%d' % args.gpu)
                
                # استخراج state_dict از ساختارهای مختلف چک‌پوینت
                if isinstance(ckpt_loaded, dict):
                    if 'state_dict' in ckpt_loaded:
                        ckpt = ckpt_loaded['state_dict']
                        logger.info("state_dict از کلید 'state_dict' استخراج شد")
                    elif 'model' in ckpt_loaded:
                        ckpt = ckpt_loaded['model']
                        logger.info("state_dict از کلید 'model' استخراج شد")
                    else:
                        ckpt = ckpt_loaded
                        logger.info("دیکشنری مستقیماً به عنوان state_dict استفاده شد")
                else:
                    ckpt = ckpt_loaded
                    logger.info("چک‌پوینت مستقیماً state_dict است")
                
                # حذف پیشوند 'module.' اگر وجود دارد
                if any(k.startswith('module.') for k in ckpt.keys()):
                    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
                    logger.info("پیشوند 'module.' از نام لایه‌ها حذف شد")
                
                logger.info(f"تعداد کل پارامترهای بارگذاری شده: {len(ckpt)}")
                
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56/cifar100/scores.pt', 
                                map_location='cuda:%d' % args.gpu)
    
    elif args.arch == 'resnet110':
        model = resnet110(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110.th', 
                                map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110/cifar100/scores.pt', 
                                map_location='cuda:%d' % args.gpu)
    else:
        raise ValueError(f"معماری معلم نامعتبر: {args.arch}")

    # ===== بارگذاری وزن‌ها به مدل معلم =====
    # این قسمت فقط اگر pretrained=True باشد اجرا می‌شود
    if args.pretrained:
        if ckpt is None:
            raise RuntimeError(
                f"فلگ --pretrained فعال است اما چک‌پوینت برای {args.arch} روی {args.set} یافت نشد.\n"
                f"لطفاً مطمئن شوید که فایل چک‌پوینت در مسیر صحیح قرار دارد."
            )
        
        try:
            model.load_state_dict(ckpt, strict=True)
            logger.info("✓ وزن‌های مدل معلم با موفقیت بارگذاری شد (strict mode)")
        except RuntimeError as e:
            logger.warning(f"⚠ خطا در بارگذاری با strict=True: {str(e)[:200]}...")
            logger.info("در حال تلاش برای بارگذاری با strict=False...")
            missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
            if missing_keys:
                logger.warning(f"کلیدهای گمشده: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"کلیدهای اضافی: {unexpected_keys[:5]}...")
            logger.info("✓ وزن‌ها با حالت non-strict بارگذاری شدند")
    else:
        logger.warning("⚠ فلگ --pretrained غیرفعال است. مدل معلم با وزن‌های تصادفی شروع می‌شود!")
        logger.warning("⚠ برای Knowledge Distillation مؤثر، حتماً از مدل معلم آموزش دیده استفاده کنید.")
    
    # ===== انتقال مدل‌ها به GPU =====
    model_s = set_gpu(args, model_s)
    model = set_gpu(args, model)

    # ===== غیرفعال کردن گرادیان برای مدل معلم =====
    # مدل معلم فقط برای استخراج دانش استفاده می‌شود و نیازی به آموزش ندارد
    for param in model.parameters():
        param.requires_grad = False

    # نمایش وضعیت گرادیان پارامترهای مدل معلم
    logger.info("وضعیت پارامترهای مدل معلم:")
    for name, param in model.named_parameters():
        print(f"Teacher Model - {name}: requires_grad={param.requires_grad}")

    print('-' * 100)
    
    # نمایش وضعیت گرادیان پارامترهای مدل دانش‌آموز
    logger.info("وضعیت پارامترهای مدل دانش‌آموز:")
    for name, param in model_s.named_parameters():
        print(f"Student Model - {name}: requires_grad={param.requires_grad}")
    
    # قرار دادن مدل معلم در حالت ارزیابی
    model.eval()

    # ===== تعریف توابع Loss =====
    divergence_loss = F.kl_div
    criterion = nn.CrossEntropyLoss().cuda()

    if args.set == 'cifar10':
        data = CIFAR10()
    elif args.set == 'cifar100':
        data = CIFAR100()
    else:
        raise ValueError(f"دیتاست نامعتبر: {args.set}")

    # ===== ارزیابی اولیه مدل معلم =====
    logger.info("در حال ارزیابی دقت مدل معلم...")
    acc1, acc5 = validate(data.val_loader, model, criterion, args)
    print(f"Teacher model accuracy: Top-1={acc1:.2f}%, Top-5={acc5:.2f}%")
    logger.info(f"دقت اولیه مدل معلم - Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")

    # ===== تنظیم Optimizer و Scheduler =====
    optimizer = torch.optim.SGD(model_s.parameters(), lr=args.lr, 
                               momentum=args.momentum, weight_decay=args.weight_decay)
    
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    # ===== متغیرهای ذخیره بهترین نتایج =====
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    args.start_epoch = args.start_epoch or 0
    mask_list = []
    layer_num = []
    
    # ===== حلقه آموزش اصلی =====
    logger.info("=" * 80)
    logger.info("شروع فرآیند آموزش با Knowledge Distillation...")
    logger.info("=" * 80)
    
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        # آموزش مدل دانش‌آموز با Knowledge Distillation
        train_acc1, train_acc5 = train_KD(data.train_loader, model, model_s, 
                                          divergence_loss, criterion, optimizer, epoch, args)
        
        # ارزیابی مدل دانش‌آموز
        acc1, acc5 = validate(data.val_loader, model_s, criterion, args)
        
        # نمایش نتایج epoch
        logger.info(f"Epoch {epoch + 1} - Train: Top-1={train_acc1:.2f}%, Top-5={train_acc5:.2f}%")
        logger.info(f"Epoch {epoch + 1} - Val: Top-1={acc1:.2f}%, Top-5={acc5:.2f}%")
        
        # به‌روزرسانی learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.6f}")

        # بررسی و ذخیره بهترین مدل
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        
        if is_best:
            logger.info(f"★ بهترین مدل جدید! دقت: {acc1:.2f}%")
        
        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                # استخراج و ذخیره mask‌ها
                mask_list = []
                layer_num = []
                model_s.hook_masks()
                masks = model_s.get_masks()
                
                for key in masks.keys():
                    msk = ApproxSign(masks[key].mask).squeeze()
                    total = torch.sum(msk)
                    layer_num.append(int(total.cpu().detach().numpy()))
                    mask_list.append(msk)

                model_s.remove_hooks()
                logger.info(f"تعداد نورون‌های فعال در هر لایه: {layer_num}")

                # ذخیره فایل‌ها
                to = {'layer_num': layer_num, 'mask': mask_list}
                mask_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_T_{args.arch}_S_{args.arch_s}_mask.pt'
                model_path = f'pretrained_model/{args.arch}/{args.set}/{args.set}_{args.arch_s}.pt'
                
                torch.save(to, mask_path)
                torch.save(model_s.state_dict(), model_path)
                
                logger.info(f"✓ Mask ذخیره شد: {mask_path}")
                logger.info(f"✓ مدل ذخیره شد: {model_path}")

    # ===== نمایش نتایج نهایی =====
    logger.info("\n" + "=" * 80)
    logger.info("آموزش به پایان رسید!")
    logger.info(f"بهترین دقت مدل دانش‌آموز:")
    logger.info(f"  Validation - Top-1: {best_acc1:.2f}%, Top-5: {best_acc5:.2f}%")
    logger.info(f"  Training - Top-1: {best_train_acc1:.2f}%, Top-5: {best_train_acc5:.2f}%")
    logger.info("=" * 80)


def ApproxSign(mask):
   
    out_forward = torch.sign(mask)
    
    mask1 = mask < -1
    mask2 = mask < 0
    mask3 = mask < 1
    
    out1 = (-1) * mask1.type(torch.float32) + (mask * mask + 2 * mask) * (1 - mask1.type(torch.float32))
    out2 = out1 * mask2.type(torch.float32) + (-mask * mask + 2 * mask) * (1 - mask2.type(torch.float32))
    out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
    
    out = out_forward.detach() - out3.detach() + out3
    out = (out + 1) / 2
    
    return out


if __name__ == "__main__":

    
    main()
