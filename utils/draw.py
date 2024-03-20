import numpy as np
# import matplotlib.pyplot as plt
from torchvision.utils import save_image  # 用于生成图片、保持图片


# 绘图函数
def DrawGen(model, epoch, test_input, sensor):
    """
    :param model: 生成器训练的模型
    :param epoch: 迭代次数
    :param test_input: 对产生的噪声生成图像
    :param test_label: 噪声对应标签
    :return:
    """
    result = model(test_input).detach()
    # 将维度为1的进行压缩
    # --------------------------------------------------------------
    # vutilsImg = vutils.make_grid(result, padding=2, normalize=True)
    # fig = plt.figure(figsize=(4, 4))
    # plt.imshow(np.transpose(vutilsImg, (1, 2, 0)))
    # plt.axis('off')
    # plt.show()
    # --------------------------------------------------------------

    save_image(result.data, 'images/' + sensor + '/ACGAN_128_%d.png' % (epoch + 1), nrow=8, normalize=False)

    '''
    # result = np.squeeze(result.numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plot_img = np.transpose(result[i], (1, 2, 0))
        plt.imshow(plot_img)
        plt.axis('off')
    plt.savefig('images/{}.png'.format(epoch))
    # --------------------------------------------------------------
    '''
