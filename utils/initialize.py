import torch
from torch import nn
import warnings


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal(m.weight.data, 0.0, 0.02)


warnings.filterwarnings('ignore')
