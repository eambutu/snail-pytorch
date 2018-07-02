from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    """convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)

def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, 3, padding=1)),
        ('bn', nn.BatchNorm2d(out_channels, momentum=1)),
        #('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))

def batchnorm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True,eps=1e-5, momentum=0.1):
    # momentum = 1 restricts stats to the current mini-batch
    # This hack only works when momentum is 1 and avoids needing to track
    # running stats by substituting dummy variables
    size = int(np.prod(np.array(input.data.size()[1])))
    running_mean = torch.zeros(size).cuda()
    running_var = torch.ones(size).cuda()
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

class OmniglotNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(OmniglotNet, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('block1', conv_block(x_dim, hid_dim)),
            ('block2', conv_block(hid_dim, hid_dim)),
            ('block3', conv_block(hid_dim, hid_dim)),
            ('block4', conv_block(hid_dim, z_dim)),
        ]))

    def forward(self, x, weights=None):
        if weights is None:
            x = self.encoder(x)
        else:
            x = F.conv2d(x, weights['encoder.block1.conv.weight'], weights['encoder.block1.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block1.bn.weight'], bias=weights['encoder.block1.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block2.conv.weight'], weights['encoder.block2.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block2.bn.weight'], bias=weights['encoder.block2.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block3.conv.weight'], weights['encoder.block3.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block3.bn.weight'], bias=weights['encoder.block3.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block4.conv.weight'], weights['encoder.block4.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block4.bn.weight'], bias=weights['encoder.block4.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
        return x.view(x.size(0), -1)

class ResBlock(nn.Module):

    def __init__(self, in_channels, filters, pool_padding=0):
        super(ResBlock, self).__init__()
        self.conv1 = conv(in_channels, filters)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = conv(filters, filters)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = conv(filters, filters)
        self.bn3 = nn.BatchNorm2d(filters)
        self.relu3 = nn.LeakyReLU()
        self.conv4 = conv(in_channels, filters, kernel_size=1, padding=0)

        self.maxpool = nn.MaxPool2d(2, padding=pool_padding)
        self.dropout = nn.Dropout(p=0.9)

    def forward(self, x):
        residual = self.conv4(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out += residual
        out = self.maxpool(out)
        out = self.dropout(out)

        return out

class MiniImagenetNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, in_channels=3):
        super(MiniImagenetNet, self).__init__()
        self.block1 = ResBlock(in_channels, 64)
        self.block2 = ResBlock(64, 96)
        self.block3 = ResBlock(96, 128, pool_padding=1)
        self.block4 = ResBlock(128, 256, pool_padding=1)
        self.conv1 = conv(256, 2048, kernel_size=1, padding=0)
        self.maxpool = nn.MaxPool2d(6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.9)
        self.conv2 = conv(2048, 384, kernel_size=1, padding=0)
        
    def forward(self, x, weights=None):
        if weights is None:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.conv2(x)
        else:
            raise ValueError('Not implemented yet')
        return x.view(x.size(0), -1)
