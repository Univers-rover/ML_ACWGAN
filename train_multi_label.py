import os
import random
import argparse

import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
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
SENSOR = 'PS2'
PART_NUM = 3

num_classes = [3, 4, 3, 4, 2]
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

dataset_path = 'D:\\Graduate Thesis\\images\\real_image\\128\\' + SENSOR + '\\'
train_dataset = MyDataSet(dataset_path=dataset_path, transform=transform)
dataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# 实例模型
netG = Generator_128(Z_DIM)  # 3 + 3 + 4 + 4 + 2
netD_1 = Discriminator_128(num_classes[0])  #
netD_2 = Discriminator_128(num_classes[1])  #
netD_3 = Discriminator_128(num_classes[2])  #
netD_4 = Discriminator_128(num_classes[3])  #
netD_5 = Discriminator_128(num_classes[4])  #

netG.apply(weights_init)
netD_1.apply(weights_init)
netD_2.apply(weights_init)
netD_3.apply(weights_init)
netD_4.apply(weights_init)
netD_5.apply(weights_init)

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
    netG.cuda()
    netD_1.cuda()
    netD_2.cuda()
    netD_3.cuda()
    netD_4.cuda()
    netD_5.cuda()
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
eval_label = pd.read_csv('test_label.csv', encoding="utf-8", header=None)
eval_label = torch.from_numpy(np.array(eval_label) - 1)
eval_noise, eval_label = label_to_onehot(eval_label.shape[0], Z_DIM, eval_label, PART_NUM, device=device)
# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerD_1 = optim.Adam(netD_1.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerD_2 = optim.Adam(netD_2.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerD_3 = optim.Adam(netD_3.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerD_4 = optim.Adam(netD_4.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerD_5 = optim.Adam(netD_5.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

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
        netD_1.zero_grad()
        netD_2.zero_grad()
        netD_3.zero_grad()
        netD_4.zero_grad()
        netD_5.zero_grad()
        batch_size = real_img.size(0)
        input.data.resize_as_(real_img).copy_(real_img)
        dis_label.data.resize_(batch_size).fill_(real_label).cuda()
        # aux_label.data.resize_(batch_size).copy_(label)  # 训练判别器用单标签/多标签？
        noise, aux_label = label_to_onehot(BATCH_SIZE, Z_DIM, label, PART_NUM, device=device)
        dis_output_1, aux_output_1 = netD_1(input)
        dis_output_2, aux_output_2 = netD_2(input)
        dis_output_3, aux_output_3 = netD_3(input)
        dis_output_4, aux_output_4 = netD_4(input)
        dis_output_5, aux_output_5 = netD_5(input)

        dis_errD_real_1 = dis_criterion(dis_output_1, dis_label)  # 真假判定损失
        aux_errD_real_1 = aux_criterion(aux_output_1, label[:, 0])  # 多标签损失如何计算？
        dis_errD_real_2 = dis_criterion(dis_output_2, dis_label)  # 判别器2
        aux_errD_real_2 = aux_criterion(aux_output_2, label[:, 1])  #
        dis_errD_real_3 = dis_criterion(dis_output_3, dis_label)  # 判别器3
        aux_errD_real_3 = aux_criterion(aux_output_3, label[:, 2])  #
        dis_errD_real_4 = dis_criterion(dis_output_4, dis_label)  # 判别器4
        aux_errD_real_4 = aux_criterion(aux_output_4, label[:, 3])  #
        dis_errD_real_5 = dis_criterion(dis_output_5, dis_label)  # 判别器5
        aux_errD_real_5 = aux_criterion(aux_output_5, label[:, 4])  #

        errD_real_1 = dis_errD_real_1 + aux_errD_real_1  # 判别器1
        errD_real_1.backward()
        errD_real_2 = dis_errD_real_2 + aux_errD_real_2  # 判别器2
        errD_real_2.backward()
        errD_real_3 = dis_errD_real_3 + aux_errD_real_3  # 判别器3
        errD_real_3.backward()
        errD_real_4 = dis_errD_real_4 + aux_errD_real_4  # 判别器4
        errD_real_4.backward()
        errD_real_5 = dis_errD_real_5 + aux_errD_real_5  # 判别器5
        errD_real_5.backward()
        # D_x = dis_output.data.mean()

        # compute the current classification accuracy
        accuracy = np.ones(5, dtype=float)
        accuracy[0] = compute_acc(aux_output_1, label[:, 0])
        accuracy[1] = compute_acc(aux_output_2, label[:, 1])
        accuracy[2] = compute_acc(aux_output_3, label[:, 2])
        accuracy[3] = compute_acc(aux_output_4, label[:, 3])
        accuracy[4] = compute_acc(aux_output_5, label[:, 4])

        # train with fake
        fake = netG(noise)
        dis_label.data.fill_(fake_label)
        dis_output_1, aux_output_1 = netD_1(fake.detach())
        dis_output_2, aux_output_2 = netD_2(fake.detach())
        dis_output_3, aux_output_3 = netD_3(fake.detach())
        dis_output_4, aux_output_4 = netD_4(fake.detach())
        dis_output_5, aux_output_5 = netD_5(fake.detach())
        dis_errD_fake_1 = dis_criterion(dis_output_1, dis_label)  # 真假判定损失
        aux_errD_fake_1 = aux_criterion(aux_output_1, label[:, 0])  # 分类损失
        dis_errD_fake_2 = dis_criterion(dis_output_2, dis_label)  # 判别器2
        aux_errD_fake_2 = aux_criterion(aux_output_2, label[:, 1])
        dis_errD_fake_3 = dis_criterion(dis_output_3, dis_label)  # 判别器3
        aux_errD_fake_3 = aux_criterion(aux_output_3, label[:, 2])
        dis_errD_fake_4 = dis_criterion(dis_output_4, dis_label)  # 判别器4
        aux_errD_fake_4 = aux_criterion(aux_output_4, label[:, 3])
        dis_errD_fake_5 = dis_criterion(dis_output_5, dis_label)  # 判别器5
        aux_errD_fake_5 = aux_criterion(aux_output_5, label[:, 4])

        errD_fake_1 = dis_errD_fake_1 + aux_errD_fake_1  # 判别器1假图片损失反向传播
        errD_fake_1.backward()
        errD_fake_2 = dis_errD_fake_2 + aux_errD_fake_2  # 判别器2假图片损失反向传播
        errD_fake_2.backward()
        errD_fake_3 = dis_errD_fake_3 + aux_errD_fake_3  # 判别器3假图片损失反向传播
        errD_fake_3.backward()
        errD_fake_4 = dis_errD_fake_4 + aux_errD_fake_4  # 判别器4假图片损失反向传播
        errD_fake_4.backward()
        errD_fake_5 = dis_errD_fake_5 + aux_errD_fake_5  # 判别器5假图片损失反向传播
        errD_fake_5.backward()
        # D_G_z1 = dis_output.data.mean()
        errD_1 = errD_real_1 + errD_fake_1
        optimizerD_1.step()
        errD_2 = errD_real_2 + errD_fake_2
        optimizerD_2.step()
        errD_3 = errD_real_3 + errD_fake_3
        optimizerD_3.step()
        errD_4 = errD_real_4 + errD_fake_4
        optimizerD_4.step()
        errD_5 = errD_real_5 + errD_fake_5
        optimizerD_5.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        dis_label.data.fill_(real_label)  # fake labels are real for generator cost
        dis_output_1, aux_output_1 = netD_1(fake)
        dis_errG_1 = dis_criterion(dis_output_1, dis_label)  # 真假判定损失
        aux_errG_1 = aux_criterion(aux_output_1, label[:, 0])  # 分类损失
        dis_output_2, aux_output_2 = netD_2(fake)  # 判别器2
        dis_errG_2 = dis_criterion(dis_output_2, dis_label)
        aux_errG_2 = aux_criterion(aux_output_2, label[:, 1])
        dis_output_3, aux_output_3 = netD_3(fake)  # 判别器3
        dis_errG_3 = dis_criterion(dis_output_3, dis_label)
        aux_errG_3 = aux_criterion(aux_output_3, label[:, 2])
        dis_output_4, aux_output_4 = netD_4(fake)  # 判别器4
        dis_errG_4 = dis_criterion(dis_output_4, dis_label)
        aux_errG_4 = aux_criterion(aux_output_4, label[:, 3])
        dis_output_5, aux_output_5 = netD_5(fake)  # 判别器5
        dis_errG_5 = dis_criterion(dis_output_5, dis_label)
        aux_errG_5 = aux_criterion(aux_output_5, label[:, 4])
        
        errG = (dis_errG_1 + aux_errG_1 + dis_errG_2 + aux_errG_2 +
                dis_errG_3 + aux_errG_3 + dis_errG_4 + aux_errG_4 + dis_errG_5 + aux_errG_5)/5
        errG.backward()
        # D_G_z2 = dis_output.data.mean()
        optimizerG.step()

        # compute the average loss
        errD = (errD_1 + errD_2 + errD_3 + errD_4 + errD_5)/5
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
        loop.set_postfix({'D_loss': '{:.3f}'.format(errD.data), 'G_loss': '{:.3f}'.format(errG.data),
                          'acc': '{0:1.3f}'.format(accuracy.mean())})
        # loop.set_postfix('D_loss: {:.3f} G_loss: {:.3f} batch acc: {:.3f}'.format(errD.data, errG.data, accuracy.mean())
        #                  .format(errD.data, errG.data, accuracy.mean()))

    # do checkpointing
    if (epoch + 1) % 5 == 0:
        DrawGen(netG, epoch, eval_noise)
    if (epoch + 1) % 100 == 0:
        torch.save(obj=netG, f='models/gen_' + SENSOR + '_epoch' + str((epoch + 1)) + '.pth')

torch.save(obj=netG, f='models/gen_' + SENSOR + '_rgb.pth')
torch.save(obj=netD_1, f='models/disc_' + SENSOR + '_rgb.pth')
