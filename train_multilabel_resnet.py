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

from net.model import Generator
# from net.model import Discriminator
from utils.draw import DrawGen
from utils.gp_function import gradient_penality, cacl_gradient_penalty
from utils.initialize import initialize_weights
from utils.My_dataset import MyDataSet
from utils.utils import weights_init, compute_acc, label_to_onehot, cal_distances
from net.Generator import *
from net.Discriminator import *
from net.ResNet18 import *

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
BATCH_SIZE = 32
IMG_SHAPE = (3, 128, 128)  # 图像shape
Z_DIM = 100 + 16  # 噪声维度
NUM_CLASSES = 3  # 类别数量
NUM_EPOCHS = 600  # 训练周期
FEATURES_DISC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 1  # 一轮训练多少次判别器,5
GEN_ITERATIONS = 1  # 一轮训练多少次生成,5
WEIGHT_CLIP = 0.01
LAMBDA_GP = 10  # 梯度惩罚系数
SENSOR = 'FS1'
PART_NUM = 3

num_classes = [3, 4, 3, 4, 2]
PART = PARTS[3]
if PART_NUM == 1 | PART_NUM == 3:
    NUM_CLASSES = 4

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([128, 128])
])  # normalize 标准化至均值为0，标准差为1，使模型容易收敛

dataset_path = 'D:\\Graduate Thesis\\images\\real_image\\128\\' + SENSOR + '\\'
train_dataset = MyDataSet(dataset_path=dataset_path, transform=transform)
dataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# 实例模型
def add_sn(model):
    """谱归一化,传入模型实例即可"""
    for name, layer in model.named_children():
        model.add_module(name, add_sn(layer))
        if isinstance(model, (nn.Conv2d, nn.Linear)):
            return nn.utils.spectral_norm(model)
        else:
            return model
    return model


netG = Generator_128(Z_DIM)  # 3 + 3 + 4 + 4 + 2
netD = Discriminator()
netD = add_sn(netD)
netD_1 = ResNet(num_classes=num_classes[0])  #
netD_2 = ResNet(num_classes=num_classes[1])  #
netD_3 = ResNet(num_classes=num_classes[2])  #
netD_4 = ResNet(num_classes=num_classes[3])  #
netD_5 = ResNet(num_classes=num_classes[4])  #

netG.apply(weights_init)
netD.apply(weights_init)
netD_1.apply(weights_init)
netD_2.apply(weights_init)
netD_3.apply(weights_init)
netD_4.apply(weights_init)
netD_5.apply(weights_init)

# loss functions
dis_criterion = nn.BCELoss()
# aux_criterion = nn.NLLLoss()
aux_criterion = nn.CrossEntropyLoss()

# tensor placeholders
input = torch.FloatTensor(BATCH_SIZE, 3, IMG_SHAPE[1], IMG_SHAPE[2])
real = torch.FloatTensor(BATCH_SIZE, 3, IMG_SHAPE[1], IMG_SHAPE[2])
noise = torch.FloatTensor(BATCH_SIZE, Z_DIM, 1, 1)
# eval_noise = torch.FloatTensor(BATCH_SIZE, Z_DIM, 1, 1).normal_(0, 1)
dis_label = torch.FloatTensor(BATCH_SIZE)
aux_label = torch.LongTensor(BATCH_SIZE)
real_label = 1  # 可以修改为0.9
fake_label = 0

# 数据转换到cuda
if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    netD_1.cuda()
    netD_2.cuda()
    netD_3.cuda()
    netD_4.cuda()
    netD_5.cuda()
    dis_criterion.cuda()
    aux_criterion.cuda()
    input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
    noise = noise.cuda()
    real = real.cuda()
    # eval_noise =  eval_noise.cuda()

# define variables
input = Variable(input)
noise = Variable(noise)
dis_label = Variable(dis_label)
aux_label = Variable(aux_label)
# noise for evaluation
eval_label = pd.read_csv('test_label.csv', encoding="utf-8", header=None)
eval_label = torch.from_numpy(np.array(eval_label) - 1)
eval_noise, eval_label = label_to_onehot(eval_label.shape[0], Z_DIM, eval_label, PART_NUM, device=device)

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerD_1 = optim.Adam(netD_1.parameters(), lr=args.lr*0.01, betas=(args.beta1, 0.999))
optimizerD_2 = optim.Adam(netD_2.parameters(), lr=args.lr*0.01, betas=(args.beta1, 0.999))
optimizerD_3 = optim.Adam(netD_3.parameters(), lr=args.lr*0.01, betas=(args.beta1, 0.999))
optimizerD_4 = optim.Adam(netD_4.parameters(), lr=args.lr*0.01, betas=(args.beta1, 0.999))
optimizerD_5 = optim.Adam(netD_5.parameters(), lr=args.lr*0.01, betas=(args.beta1, 0.999))

avg_loss_D = 0.0
avg_loss_G = 0.0
avg_loss_A = 0.0
gama = 10
all_dist = []
loss_log = []

epoch_dis_errD_real = 0
epoch_dis_errD_fake = 0
epoch_errD_gp = 0
epoch_dis_errG = 0
epoch_aux_errG = 0
epoch_accuracy = np.ones(5, dtype=float)

for epoch in range(NUM_EPOCHS):
    loop = tqdm(dataLoader, total=len(dataLoader))
    epoch_dis_errD_real = 0
    epoch_dis_errD_fake = 0
    epoch_errD_gp = 0
    epoch_dis_errG = 0
    epoch_aux_errG = 0
    epoch_accuracy = np.ones(5, dtype=float)
    for i, (real_img, label) in enumerate(loop):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        netD_1.zero_grad()
        netD_2.zero_grad()
        netD_3.zero_grad()
        netD_4.zero_grad()
        netD_5.zero_grad()
        batch_size = real_img.size(0)
        input.data.resize_as_(real_img).copy_(real_img)
        dis_label.data.resize_(batch_size).fill_(real_label).cuda()
        noise, aux_label = label_to_onehot(BATCH_SIZE, Z_DIM, label, PART_NUM, device=device)
        dis_output = netD(input)

        dis_errD_real = - torch.mean(dis_output)  # 真假判定损失
        if epoch == 0 or epoch % 2 == 0:
            aux_output_1 = netD_1(input)
            aux_output_2 = netD_2(input)
            aux_output_3 = netD_3(input)
            aux_output_4 = netD_4(input)
            aux_output_5 = netD_5(input)
            aux_errD_real_1 = aux_criterion(aux_output_1, label[:, 0])  # 多标签损失如何计算？
            aux_errD_real_2 = aux_criterion(aux_output_2, label[:, 1])  #
            aux_errD_real_3 = aux_criterion(aux_output_3, label[:, 2])  #
            aux_errD_real_4 = aux_criterion(aux_output_4, label[:, 3])  #
            aux_errD_real_5 = aux_criterion(aux_output_5, label[:, 4])  #
            aux_errD_real_1.backward()  # 判别器1
            aux_errD_real_2.backward()  # 判别器2
            aux_errD_real_3.backward()  # 判别器3
            aux_errD_real_4.backward()  # 判别器4
            aux_errD_real_5.backward()  # 判别器5

        # train with fake
        fake = netG(noise).detach()
        dis_label.data.fill_(fake_label)
        dis_output = netD(fake)

        dis_errD_fake = torch.mean(dis_output)  # 真假判定损失
        real.data.resize_as_(real_img).copy_(real_img)
        errD_gp = cacl_gradient_penalty(netD, real, fake)
        errD = dis_errD_real + dis_errD_fake + errD_gp * LAMBDA_GP
        errD.backward()  # 反向传播
        optimizerD.step()
        if epoch == 0 or epoch % 2 == 0:
            optimizerD_1.step()
            optimizerD_2.step()
            optimizerD_3.step()
            optimizerD_4.step()
            optimizerD_5.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        for p in netD.parameters():
            # reduce memory usage
            p.requires_grad_(False)
        fake = netG(noise)
        dis_output = netD(fake)
        aux_output_1 = netD_1(fake)
        aux_output_2 = netD_2(fake)
        aux_output_3 = netD_3(fake)
        aux_output_4 = netD_4(fake)
        aux_output_5 = netD_5(fake)
        dis_errG = - torch.mean(dis_output)
        aux_errG_1 = aux_criterion(aux_output_1, label[:, 0])  # 分类损失
        aux_errG_2 = aux_criterion(aux_output_2, label[:, 1])  # 分类损失
        aux_errG_3 = aux_criterion(aux_output_3, label[:, 2])  # 分类损失
        aux_errG_4 = aux_criterion(aux_output_4, label[:, 3])  # 分类损失
        aux_errG_5 = aux_criterion(aux_output_5, label[:, 4])  # 分类损失
        aux_errG = (aux_errG_1 + aux_errG_2 + aux_errG_3 + aux_errG_4 + aux_errG_5) / gama
        errG = dis_errG + aux_errG
        errG.backward()
        optimizerG.step()
        for p in netD.parameters():
            p.requires_grad_(True)
        # compute the current classification accuracy
        accuracy = np.ones(5, dtype=float)
        accuracy[0] = compute_acc(aux_output_1, label[:, 0])
        accuracy[1] = compute_acc(aux_output_2, label[:, 1])
        accuracy[2] = compute_acc(aux_output_3, label[:, 2])
        accuracy[3] = compute_acc(aux_output_4, label[:, 3])
        accuracy[4] = compute_acc(aux_output_5, label[:, 4])

        # compute the average loss
        # errD_AU = (errD_1 + errD_2 + errD_3 + errD_4 + errD_5) / 5  # 分类器的平均损失
        epoch_dis_errD_real = epoch_dis_errD_real + dis_errD_real
        epoch_dis_errD_fake = epoch_dis_errD_fake + dis_errD_fake
        epoch_errD_gp = epoch_errD_gp + errD_gp
        epoch_dis_errG = epoch_dis_errG + dis_errG
        epoch_aux_errG = epoch_aux_errG + aux_errG
        epoch_accuracy = epoch_accuracy + accuracy

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

        loop.set_description(f'Epoch [{epoch}/{NUM_EPOCHS}]')
        loop.set_postfix({'D_loss_real': '{:.3f}'.format(dis_errD_real.data), 'D_loss_fake': '{:.3f}'.format(dis_errD_fake.data),
                          'D_loss_gp': '{:.3f}'.format(errD_gp.data),
                          'G_dis_loss': '{:.3f}'.format(dis_errG.data), 'G_cls_loss': '{:.3f}'.format(aux_errG.data),
                          'ac1': '{0:1.3f}'.format(accuracy[0]), 'ac2': '{0:1.3f}'.format(accuracy[1]),
                          'ac3': '{0:1.3f}'.format(accuracy[2]), 'ac4': '{0:1.3f}'.format(accuracy[3])})
    # do checkpointing
    epoch_dis_errD_real = epoch_dis_errD_real / len(dataLoader)
    epoch_dis_errD_fake = epoch_dis_errD_fake / len(dataLoader)
    epoch_errD_gp = epoch_errD_gp / len(dataLoader)
    epoch_dis_errG = epoch_dis_errG / len(dataLoader)
    epoch_aux_errG = epoch_aux_errG / len(dataLoader)
    epoch_accuracy = epoch_accuracy / len(dataLoader)

    temp_loss_log = [epoch_dis_errD_real.data.cpu().numpy(), epoch_dis_errD_fake.data.cpu().numpy(), epoch_errD_gp.data.cpu().numpy(),
                     epoch_dis_errG.data.cpu().numpy(), epoch_aux_errG.data.cpu().numpy(),
                     epoch_accuracy[0], epoch_accuracy[1], epoch_accuracy[2], epoch_accuracy[3], epoch_accuracy[4]]
    loss_log.append(temp_loss_log)
    dist = cal_distances(input, fake)  # 计算距离
    xx = dist.squeeze(0).cpu().numpy()
    all_dist.append(xx)
    if epoch < 10 or (epoch + 1) % 5 == 0:
        DrawGen(netG, epoch, eval_noise, SENSOR)
    if (epoch + 1) % 100 == 0:
        torch.save(obj=netG, f='models/' + SENSOR + '/gen_epoch' + str((epoch + 1)) + '.pth')
        torch.save(obj=netD, f='models/' + SENSOR + '/dis_epoch' + str((epoch + 1)) + '.pth')
distances = pd.DataFrame(all_dist)
distances.to_csv('./models/' + SENSOR + '/distances.csv', encoding='utf-8', index=False)
loss_log = pd.DataFrame(loss_log)
loss_log.to_csv('./models/' + SENSOR + '/loss_log.csv', encoding='utf-8', index=False)

torch.save(obj=netG, f='models/' + SENSOR + '/gen.pth')
torch.save(obj=netD, f='models/' + SENSOR + '/dis.pth')
