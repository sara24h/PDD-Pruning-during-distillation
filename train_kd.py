import os
import sys
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
#from Hrank_resnet import resnet_110
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

    if args.arch_s == 'cvgg11_bn':
        model_s = cvgg11_bn(finding_masks=True, num_classes=args.num_classes, batch_norm=True)
    elif args.arch_s == 'resnet20':
        in_cfg = [3, 16, 16, 16, 32, 32, 32, 64, 64, 64]
        out_cfg = [16, 16, 16, 32, 32, 32, 64, 64, 64, 64]
        model_s = resnet20(finding_masks=True, in_cfg=in_cfg, out_cfg=out_cfg, num_classes=args.num_classes)

    if args.arch == 'cvgg16_bn':
        model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_bn':
        model = cvgg19_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                # استفاده از فایل دانلود شده از GitHub
                ckpt_path = 'pretrained_model/cifar10_resnet56-187c023a.pt'
                
                # دانلود خودکار اگر فایل وجود نداشته باشد
                if not os.path.exists(ckpt_path):
                    print(f"Downloading pretrained ResNet56 model...")
                    os.makedirs('pretrained_model', exist_ok=True)
                    import urllib.request
                    url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt'
                    urllib.request.urlretrieve(url, ckpt_path)
                    print(f"Download completed: {ckpt_path}")
                
                # بارگذاری و تطبیق کلیدهای state_dict
                ckpt_original = torch.load(ckpt_path, map_location='cuda:%d' % args.gpu)
                ckpt = {}
                
                # تبدیل کلیدها برای سازگاری با مدل
                for key, value in ckpt_original.items():
                    # تبدیل fc به linear
                    if key.startswith('fc.'):
                        new_key = key.replace('fc.', 'linear.')
                        ckpt[new_key] = value
                    # حذف downsample layers که در مدل شما وجود ندارند
                    elif 'downsample' not in key:
                        ckpt[key] = value
                
                print(f"Loaded {len(ckpt)} parameters from pretrained model")
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet110':
        model = resnet110(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)

    model.load_state_dict(ckpt)
    model_s = set_gpu(args, model_s)
    model = set_gpu(args, model)

    # 將模型參數的 requires_grad 設置為 False
    for param in model.parameters():
        param.requires_grad = False

    # 打印模型參數及其是否計算梯度的情況
    for name, param in model.named_parameters():
        print(f"Parameter of the Teacher Model: {name}, Requires Gradient: {param.requires_grad}")

    print('-'*100)
    for name, param in model_s.named_parameters():
        print(f"Parameter of the Student Model: {name}, Requires Gradient: {param.requires_grad}")
    model.eval()

    divergence_loss = F.kl_div
    criterion = nn.CrossEntropyLoss().cuda()
    data = CIFAR100()

    acc1, acc5 = validate(data.val_loader, model, criterion, args)
    print("Teacher model: {}".format(acc1))

    optimizer = torch.optim.SGD(model_s.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # print(model_s.parameters())
    # multi lr
    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    # create recorder
    args.start_epoch = args.start_epoch or 0
    mask_list = []
    layer_num = []
    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        # train_acc1, train_acc5 = train(data.train_loader, model, criterion, optimizer, epoch, args)
        train_acc1, train_acc5 = train_KD(data.train_loader, model, model_s, divergence_loss, criterion, optimizer, epoch, args)
        acc1, acc5 = validate(data.val_loader, model_s, criterion, args)
        # masks_outputs = model_s.get_masks_outputs()
        # print(masks_outputs['mask.0'])  # 查看剪枝的權重
        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                mask_list = []
                layer_num = []
                model_s.hook_masks()
                masks = model_s.get_masks()  # 如果模型最優，保存mask
                for key in masks.keys():
                    msk = ApproxSign(masks[key].mask).squeeze()
                    total = torch.sum(msk)
                    # print(masks[key].mask.squeeze(), msk)
                    # print(total, torch.sum(masks[key].mask.squeeze()))  # 查看mask是否更新,更新了
                    layer_num.append(int(total.cpu().detach().numpy()))
                    mask_list.append(msk)

                model_s.remove_hooks()
                logger.info(acc1)
                logger.info(layer_num)


                to = {'layer_num': layer_num, 'mask': mask_list}
                torch.save(to, 'pretrained_model/' + args.arch + '/' + args.set + "/{}_T_{}_S_{}_mask.pt".format(args.set, args.arch, args.arch_s))
                torch.save(model_s.state_dict(), 'pretrained_model/' + args.arch + '/' + args.set + "/{}_{}.pt".format(args.set, args.arch_s))



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
    # setup: python train_kd.py --gpu 3 --arch resnet56 --set cifar10 --lr 0.01 --batch_size 256 --weight_decay 0.005 --epochs 50 --lr_decay_step 20,40  --num_classes 10 --pretrained --arch_s cvgg11_bn
    main()

