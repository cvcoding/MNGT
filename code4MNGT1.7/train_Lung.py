# -*- coding: utf-8 -*-
'''
Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47
'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import pandas as pd
import csv
from models import *
from models.vit import ViT
from utils import progress_bar
import os
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import numpy as np
import random
from PIL import Image
from torch.utils.data.dataset import ConcatDataset

print(torch.__version__)  # 1.1.0
print(torchvision.__version__)  # 0.3.0

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.set_num_threads(1)

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='add image augumentations')
# parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='32')
parser.add_argument('--n_epochs', type=int, default='0')
parser.add_argument('--n_epochs_tafter', type=int, default='300')
parser.add_argument('--patch', default='16', type=int)
parser.add_argument('--cos', default='1', action='store_true', help='Train with cosine annealing scheduling')

# * Mixup params
parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--num_classes', type=int, default=2)

args = parser.parse_args()

if args.cos:
    from warmup_scheduler import GradualWarmupScheduler
if args.aug:
    import albumentations
bs = int(args.bs)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    # transforms.Scale(128),
    transforms.Resize((128, 128)),
    # transforms.CenterCrop(128),
    # transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[0.1307], std=[0.3081]),
])

##############kaishi

testset = torchvision.datasets.ImageFolder(root='..\LungImages\\chest_xray\\test/', transform=transform_test)
# testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)


transform = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.RandomCrop(128, padding=4),
                                transforms.RandomHorizontalFlip(),
                                # transforms.Grayscale(1),
                                transforms.ToTensor(),
                                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
# 定义文件dataset
# training_dir = "./data/CIFAR10/trainval/"  # 训练集地址
# folder_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)


# siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
#                                             transform=transform,
#                                             should_invert=False)
# for i in range(11):
#     temp = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
#                                             transform=transform,
#                                             should_invert=False)
#     siamese_dataset = ConcatDataset([siamese_dataset, temp])

# # 定义图像dataloader
# train_dataloader = DataLoader(siamese_dataset,
#                               shuffle=True,
#                               batch_size=bs)
train_after_dataset = torchvision.datasets.ImageFolder(root='..\LungImages\\chest_xray\\train', transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

for i in range(1):
    temp = torchvision.datasets.ImageFolder(root='..\LungImages\\chest_xray\\train', transform=transform)
    train_after_dataset = ConcatDataset([train_after_dataset, temp])

trainafterloader = torch.utils.data.DataLoader(train_after_dataset, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)


# 自定义ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

####################jieshu

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.net == 'res18':
    net = ResNet18()
elif args.net == 'vgg':
    net = VGG('VGG19')
elif args.net == 'res34':
    net = ResNet34()
elif args.net == 'res50':
    net = ResNet50()
elif args.net == 'res101':
    net = ResNet101()
elif args.net == "vit":
    # ViT for cifar10
    net = ViT(
        image_size=128,
        patch_size=args.patch,
        kernel_size=5,
        downsample=0.5,
        batch_size=args.bs,
        num_classes=args.num_classes,
        dim=48,
        depth=10,
        heads=8,
        mlp_dim=48,
        patch_stride=2,
        patch_pading=1,
        in_chans=3,
        dropout=0.1,  # 0.1
        emb_dropout=0.1,  # 0.1
        expansion_factor=2
    )

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)  # make parallel
#     cudnn.benchmark = True
net = net.to(device)

# net = torch.load('model4cifarnores.pkl')

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in net.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in net.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

# from thop import profile, clever_format
# input_shape = (3, 128, 128)
# input_tensor = torch.randn(1, *input_shape).to(device)
# flops, params = profile(net, inputs=(input_tensor,))
# flops, params = clever_format([flops, params], "%.3f")
# print("FLOPs: %s" %(flops))
# print("params: %s" %(params))

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CL
criterion = ContrastiveLoss()
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
criterion_ce = SoftTargetCrossEntropy()  # nn.CrossEntropyLoss()

# reduce LR on Plateau
if args.opt == "adam":
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # optimizer2 = optim.Adam(net.parameters(), lr=args.lr)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    optimizer2 = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)

elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
if not args.cos:
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, min_lr=1e-3 * 1e-5, factor=0.5)
    scheduler2 = lr_scheduler.ReduceLROnPlateau(optimizer2, 'min', patience=5, verbose=True, min_lr=1e-3 * 1e-5, factor=0.5)

else:
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs - 1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=2, total_epoch=8, after_scheduler=scheduler_cosine)
    scheduler_cosine2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, args.n_epochs_tafter - 1)
    scheduler2 = GradualWarmupScheduler(optimizer2, multiplier=2, total_epoch=8, after_scheduler=scheduler_cosine2)

counter = []
loss_history = []
iteration_number = 0

import time
from timm.data import Mixup
mixup_fn = None
mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
if mixup_active:
    mixup_fn = Mixup(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes=args.num_classes)

def train_after(epoch):
    print('\nEpoch: %d' % epoch)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, min_lr=1e-3 * 1e-5,
    #                                            factor=0.5)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainafterloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if len(inputs) % 2 != 0:
            continue
        if mixup_fn is not None:
            inputs, targets1 = mixup_fn(inputs, targets)
        optimizer2.zero_grad()
        outputs = net(inputs)
        loss = criterion_ce(outputs, targets1)
        loss.backward()
        optimizer2.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer2.param_groups[0]["lr"]:.5f}'
    print(content)

    return train_loss / (batch_idx + 1)


##### Validation
import time

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            targets1 = F.one_hot(targets, num_classes=args.num_classes)
            loss = criterion_ce(outputs, targets1)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Update scheduler
    # if not args.cos:
    #     scheduler.step(test_loss)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.net + '-{}-ckpt.t7'.format(args.patch))
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer2.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    # with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
    #     appender.write(content + "\n")
    return test_loss, acc


list_loss = []
list_acc = []


def fuc():
    val_loss_best = 1e5
    acc_max = 0
    # train_loss_best = 1e5
    # temppr = torch.load('model4imagenet.pth')
    # net.load_state_dict(temppr, strict=False)
    for epoch in range(start_epoch, args.n_epochs_tafter):
        trainloss = train_after(epoch)

        time_start = time.time()
        val_loss, acc = test(epoch)
        time_end = time.time()
        print('time cost', time_end - time_start, 's')

        if acc > acc_max:
            acc_max = acc
        print('Max accuracy = ', acc_max)
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            torch.save(net.state_dict(), 'model4imagenet.pth')

        scheduler2.step(epoch - 1)

        list_loss.append(val_loss)
        list_acc.append(acc)

if __name__ == '__main__':
    fuc()

