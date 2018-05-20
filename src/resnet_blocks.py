from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, stride=1):
    """convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
