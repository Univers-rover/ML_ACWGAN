import os
import random
import argparse

import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.autograd import Variable

from utils.draw import DrawGen
from utils.gp_function import gradient_penality
from utils.initialize import initialize_weights
from utils.My_dataset import MyDataSet
from utils.utils import weights_init, compute_acc, label_to_onehot
from net.Generator import Generator_128
from net.Discriminator import Discriminator_128

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 固定参数
PARTS = ["cooler", "value", "pump", "accumulator"]

# 可修改参数
LEARNING_RATE = 1e-4  # 5e-5
BATCH_SIZE = 64
IMG_SHAPE = (3, 128, 128)  # 图像shape
Z_DIM = 100 + 16  # 噪声维度
NUM_CLASSES = 3  # 类别数量
NUM_EPOCHS = 400  # 训练周期
FEATURES_DISC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 1  # 一轮训练多少次判别器,5
GEN_ITERATIONS = 1  # 一轮训练多少次生成,5
WEIGHT_CLIP = 0.01
LAMBDA_GP = 10  # 梯度惩罚系数
SENSOR = 'FS1'
PART_NUM = 3

PART = PARTS[3]
if PART_NUM == 1 | PART_NUM == 3:
    NUM_CLASSES = 4

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=[IMG_SHAPE[1], IMG_SHAPE[1]], antialias=True),
])  # normalize 标准化至均值为0，标准差为1，使模型容易收敛

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
])  # normalize 标准化至均值为0，标准差为1，使模型容易收敛

dataset_path = 'E:\\Graduate Thesis\\images\\real_image\\128\\' + SENSOR + '\\'
train_dataset = MyDataSet(dataset_path=dataset_path, transform=transform)
dataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# 实例模型
netG = Generator_128(Z_DIM)  # 3 + 3 + 4 + 4 + 2
netD = Discriminator_128(NUM_CLASSES)  #

netG.apply(weights_init)
netD.apply(weights_init)


# loss functions
dis_criterion = nn.BCELoss()
aux_criterion = nn.NLLLoss()

# tensor placeholders
input = torch.FloatTensor(BATCH_SIZE, 3, args.imageSize, args.imageSize)
noise = torch.FloatTensor(BATCH_SIZE, Z_DIM, 1, 1)
eval_noise = torch.FloatTensor(BATCH_SIZE, Z_DIM, 1, 1).normal_(0, 1)
dis_label = torch.FloatTensor(BATCH_SIZE)
aux_label = torch.LongTensor(BATCH_SIZE)
real_label = 1  # 可以修改为0.9
fake_label = 0

# 数据转换到cuda
if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()
    dis_criterion.cuda()
    aux_criterion.cuda()
    input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
    noise, eval_noise = noise.cuda(), eval_noise.cuda()

# define variables
input = Variable(input)
noise = Variable(noise)
eval_noise = Variable(eval_noise)
dis_label = Variable(dis_label)
aux_label = Variable(aux_label)
# noise for evaluation
eval_noise_ = np.random.normal(0, 1, (BATCH_SIZE, Z_DIM))
eval_label = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
eval_onehot = np.zeros((BATCH_SIZE, NUM_CLASSES))
eval_onehot[np.arange(BATCH_SIZE), eval_label] = 1
eval_noise_[np.arange(BATCH_SIZE), :NUM_CLASSES] = eval_onehot[np.arange(BATCH_SIZE)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(BATCH_SIZE, Z_DIM, 1, 1))

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

avg_loss_D = 0.0
avg_loss_G = 0.0
avg_loss_A = 0.0
for epoch in range(NUM_EPOCHS):
    loop = tqdm(dataLoader, total=len(dataLoader))
    for i, (real_img, label) in enumerate(loop):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        batch_size = real_img.size(0)
        input.data.resize_as_(real_img).copy_(real_img)
        dis_label.data.resize_(batch_size).fill_(real_label).cuda()
        # aux_label.data.resize_(batch_size).copy_(label)  # 训练判别器用单标签/多标签？
        noise, aux_label = label_to_onehot(BATCH_SIZE, Z_DIM, label, PART_NUM, device=device)
        dis_output, aux_output = netD(input)

        dis_errD_real = dis_criterion(dis_output, dis_label)
        aux_errD_real = aux_criterion(aux_output, aux_label)  # 多标签损失如何计算？
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        D_x = dis_output.data.mean()

        # compute the current classification accuracy
        accuracy = compute_acc(aux_output, aux_label)

        # train with fake
        fake = netG(noise)
        dis_label.data.fill_(fake_label)
        dis_output, aux_output = netD(fake.detach())
        dis_errD_fake = dis_criterion(dis_output, dis_label)  # 真假判定损失
        aux_errD_fake = aux_criterion(aux_output, aux_label)  # 分类损失
        errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        D_G_z1 = dis_output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        dis_label.data.fill_(real_label)  # fake labels are real for generator cost
        dis_output, aux_output = netD(fake)
        dis_errG = dis_criterion(dis_output, dis_label)  # 真假判定损失
        aux_errG = aux_criterion(aux_output, aux_label)  # 分类损失
        errG = dis_errG + aux_errG
        errG.backward()
        D_G_z2 = dis_output.data.mean()
        optimizerG.step()

        # compute the average loss
        curr_iter = epoch * len(dataLoader) + i
        all_loss_G = avg_loss_G * curr_iter
        all_loss_D = avg_loss_D * curr_iter
        all_loss_A = avg_loss_A * curr_iter
        all_loss_G += errG.data
        all_loss_D += errD.data
        all_loss_A += accuracy
        avg_loss_G = all_loss_G / (curr_iter + 1)
        avg_loss_D = all_loss_D / (curr_iter + 1)
        avg_loss_A = all_loss_A / (curr_iter + 1)

        # print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
        #      % (epoch, args.niter, i, len(dataLoader),
        #         errD.data[0], avg_loss_D, errG.data[0], avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))
        loop.set_description(f'Epoch [{epoch}/{NUM_EPOCHS}]')
        loop.set_postfix('D_loss: {:.3f} G_loss: {:.3f} batch acc: {:.3f}'.format(errD.data, errG.data, accuracy))

    DrawGen(netG, epoch, eval_noise)
    # do checkpointing

torch.save(obj=netG, f='models/gen_' + SENSOR + str(IMG_SHAPE[1]) + '.pth')
torch.save(obj=netD, f='models/disc_' + SENSOR + str(IMG_SHAPE[1]) + '.pth')
