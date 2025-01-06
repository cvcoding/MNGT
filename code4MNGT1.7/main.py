import torch
from byol_pytorch import BYOL
from torchvision import models
import argparse
from models import *
from models.vit import ViT
from utils import progress_bar
from sklearn.metrics import roc_auc_score
import os
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
from torch.utils.data.dataset import ConcatDataset
from utilsfile.mask_utils import create_subgraph_mask2coords, create_rectangle_mask, create_rectangle_mask2coords, create_bond_mask2coords
from utilsfile.public_utils import setup_device
from skimage.feature import corner_harris
from skimage.color import rgb2gray
from skimage.restoration import denoise_tv_chambolle
from skimage.feature import corner_peaks
from utilsfile.harris import CornerDetection
import time


# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='add image augumentations')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='8')  #64
parser.add_argument('--data_address', default='..\LungImages\\chest_xray_3c\\test', type=str)
parser.add_argument('--n_epochs', type=int, default='0')
parser.add_argument('--n_epochs_tafter', type=int, default='100')
parser.add_argument('--dim', type=int, default='256')
parser.add_argument('--imagesize', type=int, default='1024')
parser.add_argument('--patch', default='16', type=int)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--cos', default='True', action='store_true', help='Train with cosine annealing scheduling')
args = parser.parse_args()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
size = int(args.imagesize)

vit = ViT(
        image_size=int(args.imagesize),
        patch_size=args.patch,
        kernel_size=5,
        downsample=0.7,
        batch_size=args.bs,
        num_classes=args.num_classes,
        dim=args.dim,
        depthin=12,
        depthout=16,
        heads=8,
        mlp_dim=args.dim,
        patch_stride=2,
        patch_pading=1,
        in_chans=3,
        dropout=0.5,  # 0.1
        emb_dropout=0.5,  # 0.1
        expansion_factor=2
    ).to(device)

learner = BYOL(
    vit,
    image_size=args.imagesize,
    hidden_layer='to_cls_token',
    projection_size=args.dim,
    projection_hidden_size=args.dim  # 4*
)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in learner.online_encoder.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in learner.online_encoder.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

# from thop import profile, clever_format
# input_shape = (3, 1024, 1024)
# input_tensor = torch.randn(1, *input_shape).to(device)
# flops, params = profile(learner.online_encoder, inputs=(input_tensor,))
# flops, params = clever_format([flops, params], "%.3f")
# print("FLOPs: %s" %(flops))
# print("params: %s" %(params))


from warmup_scheduler import GradualWarmupScheduler
opt = torch.optim.Adam(learner.parameters(), lr=args.lr)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(args.n_epochs_tafter / 2) + 1)
scheduler = GradualWarmupScheduler(opt, multiplier=2.0, total_epoch=int(args.n_epochs_tafter / 2) + 1,
                                        after_scheduler=scheduler_cosine)

transform_test = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

##############kaishi

testset = torchvision.datasets.ImageFolder(root='..\LungImages\\chest_xray_3c\\test/', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=0)
transf = transforms.ToTensor()
unloader = transforms.ToPILImage()
harris_detector = CornerDetection().to(device)

transform4siamancetrain = transforms.Compose([transforms.Resize((size, size)),
                                transforms.RandomCrop(size, padding=2),
                                transforms.RandomRotation(degrees=45, fill=(255, 255, 255)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])

transform = transforms.Compose([transforms.Resize((size, size)),
                                transforms.RandomCrop(size, padding=2),
                                # transforms.RandomHorizontalFlip(),
                                # transforms.RandomVerticalFlip(),
                                # transforms.RandomGrayscale(p=0.2),
                                transforms.RandomRotation(degrees=5, fill=(255, 255, 255)),
                                # transforms.GaussianBlur(kernel_size=3, sigma=(2.0, 2.0)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
train_after_dataset = torchvision.datasets.ImageFolder(root='..\LungImages\\chest_xray_3c\\train', transform=transform)
for i in range(0):
    temp = torchvision.datasets.ImageFolder(root='..\LungImages\\chest_xray_3c\\train', transform=transform)
    train_after_dataset = ConcatDataset([train_after_dataset, temp])

trainafterloader = torch.utils.data.DataLoader(train_after_dataset, batch_size=int(args.bs), shuffle=True, num_workers=0)
if args.cos:
    from warmup_scheduler import GradualWarmupScheduler


def extract_image_blocks(image, coordinates, label):
    """
    提取图像块的函数
    :param image: 输入图像，形状为 (C, H, W)
    :param coordinates: 坐标数组，形状为 (N, 2)，其中 N 是坐标的数量
    :param block_size: 块的大小，形状为 (2,)，表示高度和宽度
    :return: 提取的图像块列表，每个块的形状为 (B, C, H_block, W_block)
    """
    if label == 0:
        image = rgb2gray(image)
        image = transf(image).to(device)
    else:
        image = torch.mean(image, dim=0).unsqueeze(0)

    # blocks = []
    # for coord in coordinates:
    #     x, y = coord
    #     top = max(0, y - block_size[0] // 2)
    #     bottom = min(image.shape[-2], y + block_size[0] // 2 + 1)
    #     left = max(0, x - block_size[1] // 2)
    #     right = min(image.shape[-1], x + block_size[1] // 2 + 1)
    #     block = image[:, top:bottom, left:right]
    #     blocks.append(block)
    # blocks = torch.stack(blocks, dim=0).squeeze()
    # blocks_min = torch.min(blocks, dim=1)[0]
    # blocks_min = torch.min(blocks_min, dim=1)[0]
    # blocks_bool = torch.lt(blocks_min, 0.98).unsqueeze(1).repeat(1, 2)
    coeff = 10
    coordinates_x = coordinates[:, 0]
    coordinates_y = coordinates[:, 1]
    right = image.shape[-1]
    bottom = image.shape[-2]
    mask_x = ((coordinates_x < right//coeff) | (coordinates_x > right-right//coeff))
    mask_y = ((coordinates_y < bottom//coeff) | (coordinates_y > bottom-bottom//coeff))

    # for coord in coordinates:
    #     x, y = coord
    #     right = image.shape[-1]
    #     bottom = image.shape[-2]
    #     if x < right/coeff or x > right-right/coeff or y < bottom/coeff or y > bottom-bottom/coeff:
    #         block = torch.tensor(0)
    #     else:
    #         block = torch.tensor(1)
    #     blocks.append(block)
    # blocks_bool = torch.stack(blocks, dim=0).squeeze() > 0
    blocks_bool = torch.logical_not(torch.logical_or(mask_x, mask_y))

    return blocks_bool.to(device).unsqueeze(1).repeat(1, 2)


def augment_oneimage(img0):
    coords = harris_detector(img0.unsqueeze(0).to(device))

    block_size = []
    blocks_bool = extract_image_blocks(img0.cpu(), coords, 1)
    coords = coords.masked_select(blocks_bool).view(-1, 2)

    maskatomordelete = random.randint(0, 2)
    number4maskedatom = 1
    number4delete = 1
    number4deletegraph = 1
    number4topk_in_graph = 5
    imgsz = int(args.imagesize)
    if coords.size(0) < number4topk_in_graph:
        for _ in range(number4topk_in_graph):
            temp = torch.tensor(
                [torch.randint(imgsz // 3, 2 * imgsz // 3, (1,)), torch.randint(imgsz // 3, 2 * imgsz // 3, (1,))]).to(
                device).unsqueeze(0)
            coords = torch.cat((coords, temp), dim=0)

    if maskatomordelete == 0:
        random_rows = torch.randint(0, coords.size(0), (number4maskedatom,))
        selected_rows = coords[random_rows]
        mask1 = \
            create_rectangle_mask2coords(selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)),
                                         mask_shape=(16, 16))

        random_rows = torch.randint(0, coords.size(0), (number4delete,))
        selected_rows = coords[random_rows]
        mask2 = \
            create_bond_mask2coords(coords, selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)))

    elif maskatomordelete == 1:
        random_rows = torch.randint(0, coords.size(0), (number4delete,))
        selected_rows = coords[random_rows]
        mask1 = \
            create_bond_mask2coords(coords, selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)))

        random_rows = torch.randint(0, coords.size(0), (number4deletegraph,))
        selected_rows = coords[random_rows]
        mask2 = \
            create_subgraph_mask2coords(coords, selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)))

    else:
        random_rows = torch.randint(0, coords.size(0), (number4deletegraph,))
        selected_rows = coords[random_rows]
        mask1 = \
            create_subgraph_mask2coords(coords, selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)))
        random_rows = torch.randint(0, coords.size(0), (number4maskedatom,))
        selected_rows = coords[random_rows]
        mask2 = \
            create_rectangle_mask2coords(selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)),
                                         mask_shape=(16, 16))

    imgtemp = img0.to(device)
    input_masked1 = imgtemp.clone()
    for j in range(3):  # 3 channels
        input_masked1[j][mask1 == 1] = 1  # =>present it using black blocks
    img1 = unloader(input_masked1).convert('RGB')
    del mask1

    input_masked2 = imgtemp.clone()
    for j in range(3):  # 3 channels
        input_masked2[j][mask2 == 1] = 1  # =>present it using black blocks
    img2 = unloader(input_masked2).convert('RGB')
    del mask2

    img1 = transform(img1)
    img2 = transform(img2)

    # random_num1 = random.randint(0, 3)
    # random_num2 = random.randint(0, 3)
    # img1 = torch.rot90(img1, k=random_num1, dims=[1, 2])
    # img2 = torch.rot90(img2, k=random_num2, dims=[1, 2])

    return img1, img2


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=False):
        self.imageFolderDataset = imageFolderDataset
        self.transform4siamancetrain = transform4siamancetrain
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)  # 任选一个
        img0 = Image.open(img0_tuple[0]).convert('RGB')
        img0 = img0.resize((size, size))

        # hh = rgb2gray(img0)
        # plt.imshow(hh, cmap="gray")
        # plt.show()
        # hh = denoise_tv_chambolle(hh, weight=0.2, channel_axis=-1)
        # measured_image = corner_harris(torch.from_numpy(hh))
        # coords = corner_peaks(measured_image, min_distance=10)

        coords = harris_detector(transf(img0).unsqueeze(0).to(device))
        blocks_bool = extract_image_blocks(img0, coords, 0)
        coords = coords.masked_select(blocks_bool).view(-1, 2)

        maskatomordelete = random.randint(0, 1)
        number4maskedatom = 3
        number4delete = 3
        number4deletegraph = 3
        number4topk_in_graph = 5
        imgsz = int(args.imagesize)
        if coords.size(0) < number4topk_in_graph:
            for _ in range(number4topk_in_graph):
                temp = torch.tensor([torch.randint(imgsz//3, 2*imgsz//3, (1,)), torch.randint(imgsz//3, 2*imgsz//3, (1,))]).to(device).unsqueeze(0)
                coords = torch.cat((coords, temp), dim=0)

        if maskatomordelete == 0:
            random_rows = torch.randint(0, coords.size(0), (number4maskedatom,))
            selected_rows = coords[random_rows]
            mask1 = \
            create_rectangle_mask2coords(selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)), mask_shape=(16, 16))

            # random_rows = torch.randint(0, coords.size(0), (number4delete,))
            # selected_rows = coords[random_rows]
            # mask2 = \
            #     create_bond_mask2coords(coords, selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)))

        elif maskatomordelete == 1:
            random_rows = torch.randint(0, coords.size(0), (number4maskedatom,))
            selected_rows = coords[random_rows]
            mask1 = \
            create_rectangle_mask2coords(selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)), mask_shape=(16, 16))

            # random_rows = torch.randint(0, coords.size(0), (number4maskedatom,))
            # selected_rows = coords[random_rows]
            # mask2 = \
            #     create_rectangle_mask2coords(selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)),
            #                              mask_shape=(16, 16))

        elif maskatomordelete == 2:
            random_rows = torch.randint(0, coords.size(0), (number4delete,))
            selected_rows = coords[random_rows]
            mask1 = \
                create_bond_mask2coords(coords, selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)))

            # random_rows = torch.randint(0, coords.size(0), (number4deletegraph,))
            # selected_rows = coords[random_rows]
            # mask2 = \
            #     create_subgraph_mask2coords(coords, selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)))

        else:
            random_rows = torch.randint(0, coords.size(0), (number4deletegraph,))
            selected_rows = coords[random_rows]
            mask1 = \
                create_subgraph_mask2coords(coords, selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)))
            # random_rows = torch.randint(0, coords.size(0), (number4maskedatom,))
            # selected_rows = coords[random_rows]
            # mask2 = \
            #     create_rectangle_mask2coords(selected_rows, shape=(1, int(args.imagesize), int(args.imagesize)),
            #                                  mask_shape=(16, 16))

        imgtemp = transf(img0).to(device)
        input_masked1 = imgtemp.clone()
        for j in range(3):  # 3 channels
            input_masked1[j][mask1 == 1] = 1  # =>present it using black blocks
        img1 = unloader(input_masked1).convert('RGB')
        del mask1

        # input_masked2 = imgtemp.clone()
        # for j in range(3):  # 3 channels
        #     input_masked2[j][mask2 == 1] = 1  # =>present it using black blocks
        # img2 = unloader(input_masked2).convert('RGB')
        # del mask2
        #
        # coords = coords.cpu().numpy()
        # plt.imshow(img1, cmap="brg")  #
        # # plt.plot(coords[:, 1], coords[:, 0], "+b", markersize=15)
        # # plt.axis("off")
        # plt.show()
        # plt.draw()
        # plt.savefig('./img/pic-{}.png'.format(index))
        #
        if self.transform4siamancetrain is not None:
            img1 = self.transform4siamancetrain(img1)
            img2 = self.transform4siamancetrain(img0)
        #
        # random_num1 = random.randint(0, 3)
        # while True:
        #     random_num2 = random.randint(0, 3)
        #     if random_num2 != random_num1:
        #         break
        # img1 = torch.rot90(img0, k=random_num1, dims=[1, 2])
        # img2 = torch.rot90(img0, k=random_num2, dims=[1, 2])

        return img1, img2

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# 定义文件datasetm Saving..
# Tue Nov 14 00:27:36 2023 Epoch 35, test loss: 1.56787, acc: 88.67925, roc_auc_avg: 0.89025
# best acc=88
training_dir = args.data_address  # 训练集地址
folder_dataset = torchvision.datasets.ImageFolder(root=training_dir)

# 定义图像dataloader
criterion_ce = nn.CrossEntropyLoss().to(device)

mean = [0.485, 0.456, 0.406]  # 这些是 ImageNet 的 RGB 通道的均值
std = [0.229, 0.224, 0.225]  # 这些是 ImageNet 的 RGB 通道的标准差

def train_after(epoch):
    print('\nEpoch: %d' % epoch)
    learner.train()
    # learner.online_encoder.required_grad = False
    train_loss = 0
    correct = 0
    total = 0
    accumulation = 16

    for batch_idx, (inputs, targets) in enumerate(trainafterloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # img = inputs.float()
        # images_denorm = img * torch.tensor(std).view(-1, 1, 1).to(device) + torch.tensor(mean).view(-1, 1, 1).to(device)
        # images_denorm = images_denorm.type_as(img)
        # augmented_img1 = unloader(images_denorm[0, :, :, :]).convert('RGB')
        # plt.imshow(augmented_img1, cmap="brg")
        # plt.show()

        _, _, outputs = learner.online_encoder(inputs)

        # img1 = torch.empty(inputs.size()).cuda()
        # img2 = torch.empty(inputs.size()).cuda()
        # for i in range(inputs.size(0)):
        #     img1[i, :, :, :], img2[i, :, :, :] = augment_oneimage(inputs[i, :, :])
        # loss0 = learner(img1, img2)

        loss = criterion_ce(outputs, targets)  #+ 0.1*loss0

        loss.backward()
        if ((batch_idx + 1) % accumulation) == 0:
            opt.step()
            opt.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    torch.save(learner.state_dict(), './improved-net.pth')
    torch.save(learner.online_encoder.state_dict(), './online_encoder.pth')

    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {opt.param_groups[0]["lr"]:.5f}'
    print(content)

    scheduler.step(epoch)

    return train_loss / (batch_idx + 1)


from sklearn.metrics import f1_score, recall_score
def test(epoch):
    global best_acc
    learner.eval()
    test_loss = 0
    correct = 0
    total = 0
    total4roc = 0
    roc_auc = 0

    batch_f1s = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # empty = torch.Tensor()
            _, _, outputs = learner.online_encoder(inputs)
            loss = criterion_ce(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

            # outputssig = outputs  # torch.sigmoid(outputs)
            # _, tempp = outputssig.max(1)
            # try:
            #     roc_auc += roc_auc_score(targets.cpu(), tempp.cpu())
            #     total4roc += 1
            # except ValueError:
            #     pass
            batch_f1 = f1_score(targets.cpu(), predicted.cpu(), average='micro')
            batch_f1s.append(batch_f1)
        average_f1 = np.mean(batch_f1s)
        print(f"Average F1-score: {average_f1:.4f}")



    # Save checkpoint.
    acc = 100. * correct / total
    # if total4roc == 0:
    #     roc_auc_avg = roc_auc / (total4roc+1)
    # else:
    #     roc_auc_avg = roc_auc / total4roc
    if acc > best_acc:
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, test loss: {test_loss:.5f}, acc: {(acc):.5f}, average_f1: {(average_f1):.4f}'
    print(content)
    # with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
    #     appender.write(content + "\n")
    return test_loss, acc


# learner.load_state_dict(torch.load('improved-net.pth'), strict=False)

# min_loss = 1e5
# accumulation = 4   #4
# for interation in range(args.n_epochs):
#     print('interation=%d'%interation)
#     torch.cuda.synchronize()
#     start = time.time()
#     train_loss = 0
#     learner.train()
#
#     for i, data in enumerate(train_dataloader, 0):
#
#         img1, img2 = data
#         # img0维度为torch.Size([32, 1, 100, 100])，32是batch，label为torch.Size([32, 1])
#         img1, img2 = img1.to(device), img2.to(device)  # 数据移至GPU
#
#         loss = learner(img1, img2)
#         train_loss += loss.item()
#         loss.backward()
#         if ((i + 1) % accumulation) == 0:
#             opt.step()
#             opt.zero_grad()
#
#     learner.update_moving_average()  # update moving average of target encoder
#
#     if train_loss < min_loss:
#         min_loss = train_loss
#         # save your improved network
#         torch.save(learner.state_dict(), './improved-net.pth')
#         torch.save(learner.online_encoder.state_dict(), './online_encoder.pth')
#
#     content = time.ctime() + ' ' + f'Epoch {interation}, Train loss: {train_loss:.2f}'
#     print(content)
#
#     torch.cuda.synchronize()
#     end = time.time()
#     # print("cost time", end-start, "s")

############ train using label $###################################
###################################################################


# optimizer2 = optim.Adam(vit.parameters(), lr=args.lr)
# scheduler_cosine2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, int(args.n_epochs_tafter / 2) + 1)
# scheduler2 = GradualWarmupScheduler(optimizer2, multiplier=2.0, total_epoch=int(args.n_epochs_tafter / 2) + 1,
#                                         after_scheduler=scheduler_cosine2)

#Thu Jan 18 00:02:24 2024 Epoch 21, test loss: 1.37011, acc: 88.67925, roc_auc_avg: 0.90809

best_acc = 0  # best test accuracy
opt.zero_grad()
# learner.load_state_dict(torch.load('improved-net-13.pth'), strict=False)
for epoch in range(0, args.n_epochs_tafter):
    trainloss = train_after(epoch)

    # learner.update_moving_average()

    test_loss, acc = test(epoch)
    print('best acc=%d' % best_acc)
