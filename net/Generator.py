"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/2 12:08
"""

import torch
from torch import nn


class Generator_128(nn.Module):
    def __init__(self, nz):
        super(Generator_128, self).__init__()
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(self.nz, 768)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 6
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 8, 2, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, input_x):
        input_x = input_x.view(-1, self.nz)
        x = self.fc1(input_x)
        x = x.view(-1, 768, 1, 1)
        x = self.tconv2(x)  # size 5
        x = self.tconv3(x)  # size 13
        x = self.tconv4(x)  # size 29
        x = self.tconv5(x)  # size 61
        out = self.tconv6(x)  # size 128
        return out


class Generator_224(nn.Module):
    def __init__(self, nz):
        super(Generator_224, self).__init__()
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(self.nz, 3072)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(192, 168, 5, 2, 0, bias=False),
            nn.BatchNorm2d(168),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(168, 122, 5, 2, 0, bias=False),
            nn.BatchNorm2d(122),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(122, 96, 5, 2, 0, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 6
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 8, 2, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, input_x):
        input_x = input_x.view(-1, self.nz)
        x = self.fc1(input_x)
        x = x.view(-1, 192, 4, 4)
        x = self.tconv2(x)  # size 11
        x = self.tconv3(x)  # size 25
        x = self.tconv4(x)  # size 53
        x = self.tconv5(x)  # size 109
        out = self.tconv6(x)  # size 224
        return out


class Generator_gray(nn.Module):
    def __init__(self, nz):
        super(Generator_gray, self).__init__()
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(self.nz, 768)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 8, 2, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, input_x):
        input_x = input_x.view(-1, self.nz)
        x = self.fc1(input_x)
        x = x.view(-1, 768, 1, 1)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        out = self.tconv6(x)
        return out
