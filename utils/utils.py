import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc


def label_to_onehot(batch_size, z_dim, label, part_num, device):
    num_classes = [3, 4, 3, 4, 2]
    noise = torch.tensor(np.random.normal(0, 1, (batch_size, z_dim - sum(num_classes))),
                         dtype=torch.float, device=device)  # 先生成[batch_size, z_dim]的正态分布噪声
    noise.data.resize_(batch_size, z_dim - sum(num_classes), 1, 1)  # resize一下

    label_onehot = torch.LongTensor(batch_size, sum(num_classes), 1, 1)  # 初始化一个存放onehot的变量
    label_onehot.cuda()
    col = 0
    for i in range(len(num_classes)):
        label0 = label[:, i]
        label0 = label0.cpu().numpy()
        class_onehot0 = np.zeros((batch_size, num_classes[i]))  # 全0的[batch_size, num_classes]
        class_onehot0[np.arange(batch_size), label0] = 1
        class_onehot0 = torch.from_numpy(class_onehot0).cuda()
        # x = torch.from_numpy(np.arange(batch_size)).cuda()
        # class_onehot0[x, label[:, i]] = torch.Tensor(1).cuda()  # 变成one-hot
        label_onehot[:, col:col + num_classes[i], 0, 0] = class_onehot0
        col = col + num_classes[i]
    noise = torch.cat((noise.view(noise.shape[0], -1, 1, 1), label_onehot.cuda()), 1)
    aux_label = label[:, part_num]

    return noise.view(noise.shape[0], -1, 1, 1), aux_label.cuda()


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
        source: 源域数据，行表示样本数目，列表示样本数据维度
        target: 目标域数据 同source
        kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
        kernel_num: 取不同高斯核的数量
        fix_sigma: 是否固定，如果固定，则为单核MMD
    Return:
        sum(kernel_val): 多个核矩阵之和
    '''

    n_samples = int(source.size()[0])+int(target.size()[0])
    # 求矩阵的行数，即两个域的的样本总数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差
    # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
    L2_distance_square = ((total0-total1)**2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance_square) / (n_samples**2-n_samples)
    # 多核MMD
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # print(bandwidth_list)
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
        source: 源域数据，行表示样本数目，列表示样本数据维度
        target: 目标域数据 同source
        kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
        kernel_num: 取不同高斯核的数量
        fix_sigma: 是否固定，如果固定，则为单核MMD
    Return:
        loss: MMD loss
    '''

    source_num = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    target_num = int(target.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = torch.mean(kernels[:source_num, :source_num])
    YY = torch.mean(kernels[source_num:, source_num:])
    XY = torch.mean(kernels[:source_num, source_num:])
    YX = torch.mean(kernels[source_num:, :source_num])
    loss = XX + YY - XY - YX
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算


def cal_distances(source, target):
    source = Variable(source.view(source.shape[0], -1))
    target = Variable(target.view(target.shape[0], -1))

    # 计算欧式距离
    pdist = nn.PairwiseDistance(p=2)
    eu_d = pdist(source, target)
    eu_d = eu_d.mean().unsqueeze(dim=0).unsqueeze(dim=1)
    # 余弦相似度
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_d = cos(source, target)
    cos_d = cos_d.mean().unsqueeze(dim=0).unsqueeze(dim=1)

    # KL散度
    kl_d = F.kl_div(source.softmax(-1).log(), target.softmax(-1), reduction='mean').unsqueeze(dim=0).unsqueeze(dim=1)

    # mmd
    mm_d = mmd_rbf(source, target).unsqueeze(dim=0).unsqueeze(dim=1)

    distances = torch.cat((eu_d, cos_d, kl_d, mm_d), 1)
    return distances
