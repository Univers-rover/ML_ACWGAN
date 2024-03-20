"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/2 12:08
"""

import torch
from torch import nn
import torchvision
from torchinfo import summary
from torchvision import transforms
from net.spectral_normalization import SpectralNorm


class Discriminator_128(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator_128, self).__init__()

        # Convolution 1
        self.conv2d_1 = nn.Conv2d(3, 16, 3, 2, 1, bias=False),
        nn.init.xavier_uniform(self.conv2d_1.weight.data, 1.)
        self.conv1 = nn.Sequential(
            SpectralNorm(self.conv2d_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2d_2 = nn.Conv2d(16, 32, 3, 1, 0, bias=False),
        self.conv2 = nn.Sequential(
            SpectralNorm(self.conv2d_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv2d_3 = nn.Conv2d(32, 64, 3, 2, 1, bias=False),
        self.conv3 = nn.Sequential(
            SpectralNorm(self.conv2d_3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv2d_4 = nn.Conv2d(64, 128, 3, 1, 0, bias=False),
        self.conv4 = nn.Sequential(
            SpectralNorm(self.conv2d_4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv2d_5 = nn.Conv2d(128, 256, 3, 2, 1, bias=False),
        self.conv5 = nn.Sequential(
            SpectralNorm(self.conv2d_5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv2d_6 = nn.Conv2d(256, 512, 3, 1, 0, bias=False),
        self.conv6 = nn.Sequential(
            SpectralNorm(self.conv2d_6),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(13*13*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(13*13*512, num_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Tanh()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        flat6 = conv6.view(-1, 13*13*512)
        fc_dis = self.fc_dis(flat6)
        fc_aux = self.fc_aux(flat6)

        classes = self.softmax(fc_aux)  # 输出
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes


class Discriminator_224(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator_224, self).__init__()

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(5*5*256, 1)


    def forward(self, input):
        conv1 = self.conv1(input)  # size 64
        conv2 = self.conv2(conv1)  # size 62
        conv3 = self.conv3(conv2)  # size 31
        conv4 = self.conv4(conv3)  # size 29
        conv5 = self.conv5(conv4)  # size 15
        conv6 = self.conv6(conv5)  # size 13
        flat6 = conv6.view(-1, 5*5*256)
        fc_dis = self.fc_dis(flat6)
        # fc_aux = self.fc_aux(flat6)

        realfake = fc_dis.view(-1, 1).squeeze(1)
        # realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake


class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(13*13*512, 1)
        # softmax and sigmoid
        # self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def forward(self, input):
        conv1 = self.conv1(input)  # size 64
        conv2 = self.conv2(conv1)  # size 62
        conv3 = self.conv3(conv2)  # size 31
        conv4 = self.conv4(conv3)  # size 29
        conv5 = self.conv5(conv4)  # size 15
        conv6 = self.conv6(conv5)  # size 13
        flat6 = conv6.view(-1, 13*13*512)
        fc_dis = self.fc_dis(flat6)
        realfake = fc_dis.view(-1, 1).squeeze(1)
        # realfake = self.tanh(fc_dis).view(-1, 1).squeeze(1)
        return realfake


class Discriminator_Gray(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator_Gray, self).__init__()

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(13*13*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(13*13*512, num_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        flat6 = conv6.view(-1, 13*13*512)
        fc_dis = self.fc_dis(flat6)
        fc_aux = self.fc_aux(flat6)

        classes = self.softmax(fc_aux)  # 输出
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes


