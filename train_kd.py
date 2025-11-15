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
import urllib.request


def download_pretrained_model(url, save_path):
   
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # بررسی وجود فایل
    if os.path.exists(save_path):
        print(f"✓ فایل از قبل موجود است: {save_path}")
        return True
    
    # دانلود فایل
    try:
        print(f"در حال دانلود از: {url}")
        print(f"ذخیره در: {save_path}")
        urllib.request.urlretrieve(url, save_path)
        print(f"✓ دانلود با موفقیت انجام شد!")
        return True
    except Exception as e:
        print(f"✗ خطا در دانلود: {str(e)}")
        return False


def main():
    print(args)
    sys.stdout = Logger('print process.log', sys.stdout)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    main_worker(args)


def main_worker(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    base_path = '/kaggle/working/pretrained_model'
    
    if not os.path.isdir(base_path + '/' + args.arch + '/' + args.set):
        os.makedirs(base_path + '/' + args.arch + '/' + args.set, exist_ok=True)
    
    logger = get_logger(base_path + '/' + args.arch + '/' + args.set + '/logger' + now + '.log')
    logger.info(args.arch)
    logger.info(args.set)
    logger.info(args.batch_size)
    logger.info(args.weight_decay)
    logger.info(args.lr)
    logger.info(args.epochs)
    logger.info(args.lr_decay_step)
    logger.info(args.num_classes)

    # ===== ساخت مدل دانش‌آموز (Student Model) =====
    logger.info("=" * 80)
    logger.info("مرحله 1: ساخت مدل دانش‌آموز (Student Model)")
    logger.info("=" * 80)
    
    if args.arch_s == 'cvgg11_bn':
        model_s = cvgg11_bn(finding_masks=True, num_classes=args.num_classes, batch_norm=True)
        logger.info(f"✓ مدل دانش‌آموز VGG11 ساخته شد")
    elif args.arch_s == 'resnet20':
        in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]
        out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
        model_s = resnet20(finding_masks=True, in_cfg=in_cfg, out_cfg=out_cfg, num_classes=args.num_classes)
        logger.info(f"✓ مدل دانش‌آموز ResNet20 ساخته شد")
    else:
        raise ValueError(f"معماری دانش‌آموز نامعتبر: {args.arch_s}")

    # ===== ساخت مدل معلم (Teacher Model) و دانلود/بارگذاری چک‌پوینت =====
    logger.info("\n" + "=" * 80)
    logger.info("مرحله 2: ساخت و بارگذاری مدل معلم (Teacher Model)")
    logger.info("=" * 80)
    
    ckpt = None
    
    if args.arch == 'cvgg16_bn':
        model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
        logger.info(f"✓ مدل معلم VGG16 ساخته شد")
        
        if args.pretrained:
            checkpoint_path = f'{base_path}/cvgg16_bn/{args.set}/scores.pt'
            logger.warning(f"⚠ برای VGG16، چک‌پوینت باید از قبل در {checkpoint_path} موجود باشد")
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location='cuda:%d' % args.gpu)
            else:
                logger.error(f"✗ فایل چک‌پوینت یافت نشد: {checkpoint_path}")
    
    elif args.arch == 'cvgg19_bn':
        model = cvgg19_bn(num_classes=args.num_classes, batch_norm=True)
        logger.info(f"✓ مدل معلم VGG19 ساخته شد")
        
        if args.pretrained:
            checkpoint_path = f'{base_path}/cvgg19_bn/{args.set}/scores.pt'
            logger.warning(f"⚠ برای VGG19، چک‌پوینت باید از قبل در {checkpoint_path} موجود باشد")
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location='cuda:%d' % args.gpu)
            else:
                logger.error(f"✗ فایل چک‌پوینت یافت نشد: {checkpoint_path}")
    
    elif args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes)
        logger.info(f"✓ مدل معلم ResNet56 ساخته شد")
        
        if args.pretrained:
            if args.set == 'cifar10':
                # مسیر و URL برای دانلود چک‌پوینت ResNet-56
                checkpoint_path = f'{base_path}/cifar10_resnet56-187c023a.pt'
                checkpoint_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt'
                
                logger.info(f"بررسی وجود چک‌پوینت ResNet-56...")
                
                # دانلود فایل در صورت عدم وجود
                if not os.path.exists(checkpoint_path):
                    logger.info(f"چک‌پوینت یافت نشد. شروع دانلود...")
                    success = download_pretrained_model(checkpoint_url, checkpoint_path)
                    
                    if not success:
                        raise FileNotFoundError(
                            f"خطا در دانلود چک‌پوینت ResNet-56.\n"
                            f"لطفاً به صورت دستی از لینک زیر دانلود کنید:\n"
                            f"{checkpoint_url}\n"
                            f"و در مسیر زیر قرار دهید:\n"
                            f"{checkpoint_path}"
                        )
                
                logger.info(f"در حال بارگذاری چک‌پوینت از: {checkpoint_path}")
                ckpt_loaded = torch.load(checkpoint_path, map_location='cuda:%d' % args.gpu)
                
                # استخراج state_dict از ساختارهای مختلف چک‌پوینت
                if isinstance(ckpt_loaded, dict):
                    if 'state_dict' in ckpt_loaded:
                        ckpt = ckpt_loaded['state_dict']
                        logger.info("→ state_dict از کلید 'state_dict' استخراج شد")
                    elif 'model' in ckpt_loaded:
                        ckpt = ckpt_loaded['model']
                        logger.info("→ state_dict از کلید 'model' استخراج شد")
                    else:
                        ckpt = ckpt_loaded
                        logger.info("→ دیکشنری مستقیماً به عنوان state_dict استفاده شد")
                else:
                    ckpt = ckpt_loaded
                    logger.info("→ چک‌پوینت مستقیماً state_dict است")
                
                # حذف پیشوند 'module.' اگر وجود دارد
                if any(k.startswith('module.') for k in ckpt.keys()):
                    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
                    logger.info("→ پیشوند 'module.' از نام لایه‌ها حذف شد")
                
                logger.info(f"✓ تعداد کل پارامترهای بارگذاری شده: {len(ckpt)}")
                
            elif args.set == 'cifar100':
                checkpoint_path = f'{base_path}/resnet56/cifar100/scores.pt'
                logger.warning(f"⚠ برای CIFAR-100، چک‌پوینت باید از قبل در {checkpoint_path} موجود باشد")
                if os.path.exists(checkpoint_path):
                    ckpt = torch.load(checkpoint_path, map_location='cuda:%d' % args.gpu)
                else:
                    logger.error(f"✗ فایل چک‌پوینت یافت نشد: {checkpoint_path}")
    
    elif args.arch == 'resnet110':
        model = resnet110(num_classes=args.num_classes)
        logger.info(f"✓ مدل معلم ResNet110 ساخته شد")
        
        if args.pretrained:
            if args.set == 'cifar10':
                checkpoint_path = f'{base_path}/resnet110.th'
                logger.warning(f"⚠ برای ResNet110، چک‌پوینت باید از قبل در {checkpoint_path} موجود باشد")
                if os.path.exists(checkpoint_path):
                    save = torch.load(checkpoint_path, map_location='cuda:%d' % args.gpu)
                    ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
                else:
                    logger.error(f"✗ فایل چک‌پوینت یافت نشد: {checkpoint_path}")
            elif args.set == 'cifar100':
                checkpoint_path = f'{base_path}/resnet110/cifar100/scores.pt'
                if os.path.exists(checkpoint_path):
                    ckpt = torch.load(checkpoint_path, map_location='cuda:%d' % args.gpu)
                else:
                    logger.error(f"✗ فایل چک‌پوینت یافت نشد: {checkpoint_path}")
    else:
        raise ValueError(f"معماری معلم نامعتبر: {args.arch}")

    # ===== بارگذاری وزن‌ها به مدل معلم =====
    logger.info("\n" + "=" * 80)
    logger.info("مرحله 3: بارگذاری وزن‌های پیش‌آموزش دیده به مدل معلم")
    logger.info("=" * 80)
    
    if args.pretrained:
        if ckpt is None:
            raise RuntimeError(
                f"✗ فلگ --pretrained فعال است اما چک‌پوینت برای {args.arch} روی {args.set} یافت نشد.\n"
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
    logger.info("\n" + "=" * 80)
    logger.info("مرحله 4: انتقال مدل‌ها به GPU و تنظیم پارامترها")
    logger.info("=" * 80)
    
    model_s = set_gpu(args, model_s)
    model = set_gpu(args, model)
    logger.info("✓ مدل‌ها به GPU منتقل شدند")

    # ===== غیرفعال کردن گرادیان برای مدل معلم =====
    # مدل معلم فقط برای استخراج دانش استفاده می‌شود و نیازی به آموزش ندارد
    for param in model.parameters():
        param.requires_grad = False
    logger.info("✓ گرادیان‌های مدل معلم غیرفعال شدند (مدل معلم فقط برای استنتاج استفاده می‌شود)")

    # نمایش تعداد پارامترها
    teacher_params = sum(p.numel() for p in model.parameters())
    student_params = sum(p.numel() for p in model_s.parameters())
    student_trainable = sum(p.numel() for p in model_s.parameters() if p.requires_grad)
    
    logger.info(f"\nآمار پارامترها:")
    logger.info(f"  مدل معلم: {teacher_params:,} پارامتر")
    logger.info(f"  مدل دانش‌آموز: {student_params:,} پارامتر")
    logger.info(f"  پارامترهای قابل آموزش دانش‌آموز: {student_trainable:,}")
    logger.info(f"  نسبت فشرده‌سازی: {teacher_params/student_params:.2f}x")
    
    # قرار دادن مدل معلم در حالت ارزیابی
    model.eval()
    logger.info("✓ مدل معلم در حالت ارزیابی (eval mode) قرار گرفت")

    # ===== تعریف توابع Loss =====
    logger.info("\n" + "=" * 80)
    logger.info("مرحله 5: تعریف توابع هزینه و بارگذاری داده‌ها")
    logger.info("=" * 80)
    
    divergence_loss = F.kl_div
    criterion = nn.CrossEntropyLoss().cuda()
    logger.info("✓ توابع هزینه تعریف شدند (KL Divergence + CrossEntropy)")

    if args.set == 'cifar10':
        data = CIFAR10()
        logger.info("✓ دیتاست CIFAR-10 بارگذاری شد")
    elif args.set == 'cifar100':
        data = CIFAR100()
        logger.info("✓ دیتاست CIFAR-100 بارگذاری شد")
    else:
        raise ValueError(f"دیتاست نامعتبر: {args.set}")

    # ===== ارزیابی اولیه مدل معلم =====
    logger.info("\n" + "=" * 80)
    logger.info("مرحله 6: ارزیابی دقت اولیه مدل معلم")
    logger.info("=" * 80)
    
    acc1, acc5 = validate(data.val_loader, model, criterion, args)
    print(f"\n{'='*60}")
    print(f"دقت مدل معلم (Teacher Model Accuracy):")
    print(f"  Top-1: {acc1:.2f}%")
    print(f"  Top-5: {acc5:.2f}%")
    print(f"{'='*60}\n")
    logger.info(f"✓ دقت اولیه مدل معلم - Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")

    # ===== تنظیم Optimizer و Scheduler =====
    logger.info("\n" + "=" * 80)
    logger.info("مرحله 7: تنظیم بهینه‌ساز و زمان‌بندی نرخ یادگیری")
    logger.info("=" * 80)
    
    optimizer = torch.optim.SGD(model_s.parameters(), lr=args.lr, 
                               momentum=args.momentum, weight_decay=args.weight_decay)
    logger.info(f"✓ بهینه‌ساز SGD:")
    logger.info(f"    نرخ یادگیری اولیه: {args.lr}")
    logger.info(f"    مومنتوم: {args.momentum}")
    logger.info(f"    وزن کاهش: {args.weight_decay}")
    
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
    logger.info(f"✓ زمان‌بندی MultiStepLR با نقاط کاهش در epoch‌های: {lr_decay_step}")

    # ===== متغیرهای ذخیره بهترین نتایج =====
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    args.start_epoch = args.start_epoch or 0
    mask_list = []
    layer_num = []
    
    # ===== حلقه آموزش اصلی =====
    logger.info("\n" + "=" * 80)
    logger.info("مرحله 8: شروع فرآیند آموزش با Knowledge Distillation")
    logger.info("=" * 80)
    logger.info(f"تعداد کل epoch‌ها: {args.epochs}")
    logger.info(f"اندازه batch: {args.batch_size}")
    logger.info("=" * 80 + "\n")
    
    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"{'='*80}")
        logger.info(f"\n--- شروع Epoch {epoch + 1}/{args.epochs} ---")
        
        # نمایش نرخ یادگیری فعلی
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"نرخ یادگیری فعلی: {current_lr:.6f}")
        
        # آموزش مدل دانش‌آموز با Knowledge Distillation
        logger.info("در حال آموزش...")
        train_acc1, train_acc5 = train_KD(data.train_loader, model, model_s, 
                                          divergence_loss, criterion, optimizer, epoch, args)
        
        # ارزیابی مدل دانش‌آموز
        logger.info("در حال ارزیابی...")
        acc1, acc5 = validate(data.val_loader, model_s, criterion, args)
        
        # نمایش نتایج epoch
        print(f"\nنتایج Epoch {epoch + 1}:")
        print(f"  Train → Top-1: {train_acc1:.2f}%, Top-5: {train_acc5:.2f}%")
        print(f"  Val   → Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")
        
        logger.info(f"Epoch {epoch + 1} - آموزش: Top-1={train_acc1:.2f}%, Top-5={train_acc5:.2f}%")
        logger.info(f"Epoch {epoch + 1} - اعتبارسنجی: Top-1={acc1:.2f}%, Top-5={acc5:.2f}%")
        
        # به‌روزرسانی learning rate
        scheduler.step()

        # بررسی و ذخیره بهترین مدل
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        
        if is_best:
            print(f"★ بهترین مدل جدید! دقت بهبود یافت به: {acc1:.2f}%")
            logger.info(f"★ بهترین مدل جدید! دقت: {acc1:.2f}%")
        
        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                logger.info("در حال استخراج و ذخیره mask‌ها...")
                
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

                # ذخیره فایل‌ها در مسیر Kaggle
                to = {'layer_num': layer_num, 'mask': mask_list}
                mask_path = f'{base_path}/{args.arch}/{args.set}/{args.set}_T_{args.arch}_S_{args.arch_s}_mask.pt'
                model_path = f'{base_path}/{args.arch}/{args.set}/{args.set}_{args.arch_s}.pt'
                
                # ایجاد پوشه در صورت نیاز
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                
                torch.save(to, mask_path)
                torch.save(model_s.state_dict(), model_path)
                
                logger.info(f"✓ Mask ذخیره شد: {mask_path}")
                logger.info(f"✓ مدل ذخیره شد: {model_path}")
                print(f"✓ فایل‌ها ذخیره شدند در: {os.path.dirname(mask_path)}")

    # ===== نمایش نتایج نهایی =====
    print("\n" + "=" * 80)
    print("آموزش به پایان رسید!")
    print("=" * 80)
    print(f"\nبهترین نتایج مدل دانش‌آموز:")
    print(f"  Validation:")
    print(f"    Top-1 Accuracy: {best_acc1:.2f}%")
    print(f"    Top-5 Accuracy: {best_acc5:.2f}%")
    print(f"  Training:")
    print(f"    Top-1 Accuracy: {best_train_acc1:.2f}%")
    print(f"    Top-5 Accuracy: {best_train_acc5:.2f}%")
    print("=" * 80 + "\n")
    
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
    
    # محاسبه تقریب قطعه‌ای سه بخشی
    out1 = (-1) * mask1.type(torch.float32) + (mask * mask + 2 * mask) * (1 - mask1.type(torch.float32))
    out2 = out1 * mask2.type(torch.float32) + (-mask * mask + 2 * mask) * (1 - mask2.type(torch.float32))
    out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
    
    # ترکیب forward و backward با استفاده از straight-through estimator
    out = out_forward.detach() - out3.detach() + out3
    
    # نرمال‌سازی به بازه [0, 1]
    out = (out + 1) / 2
    
    return out


if __name__ == "__main__":
    main()
