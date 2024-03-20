import torch
import numpy as np
import pandas as pd
from torchvision.utils import save_image  # 用于生成图片、保持图片
from utils.utils import label_to_onehot

Z_DIM = 100 + 16
IMG_SIZE = 128
IMG_SHAPE = (1, IMG_SIZE, IMG_SIZE)  # 图像shape
SAMPLE_NUM = 50
SENSOR = 'PS2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pth_gen = 'D:\\Graduate Thesis\\python\\acWgangp\\models\\' + SENSOR + '\\gen.pth'  # 训练好的模型文件
pth_dis = 'D:\\Graduate Thesis\\python\\acWgangp\\models\\' + SENSOR + '\\dis.pth'  # 训练好的模型文件
IMG_path = 'D:\\Graduate Thesis\\images\\gen_image' + \
           '\\' + 'ACWGAN_128' + '\\' + SENSOR + '\\'

eval_label = pd.read_csv('test_label1.csv', encoding="utf-8", header=None)
eval_label = torch.from_numpy(np.array(eval_label) - 1)
noise, label = label_to_onehot(eval_label.shape[0], Z_DIM, eval_label, 1, device=device)
net_G = torch.load(pth_gen)
net_D = torch.load(pth_dis)

total_sample = 0
eval_value = np.empty((0, 1))
for i in range(SAMPLE_NUM):
    fake_img = net_G(noise)
    DIS = net_D(fake_img)
    DIS = DIS.cpu().detach().numpy()
    eval_value = np.append(eval_value, DIS)
    for sample in range(label.shape[0]):
        SAVE_path = IMG_path + str(total_sample + 1) + '.png'
        total_sample += 1
        save_image(fake_img[sample, :, :, :], SAVE_path, nrow=1, normalize=False)

eval_save = pd.DataFrame({'eval': eval_value})
eval_save.to_csv(IMG_path + "eval.csv", index=False, sep=',')
