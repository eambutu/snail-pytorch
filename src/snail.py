import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet_blocks import *
from blocks import *

class SnailOmniglot(nn.Module):
    def __init__(self, N, K, use_cuda=True):
        # N-way, K-shot
        super(SnailOmniglot, self).__init__()
        self.omniglot = OmniglotNet()
        num_filters = int(math.ceil(math.log(N * K + 1)))
        num_channels = 64 + N
        self.attention1 = AttentionBlock(num_channels, 64, 32)
        num_channels += 32
        self.tc1 = TCBlock(num_channels, N * K + 1, 128)
        num_channels += num_filters * 128
        self.attention2 = AttentionBlock(num_channels, 256, 128)
        num_channels += 128
        self.tc2 = TCBlock(num_channels, N * K + 1, 128)
        num_channels += num_filters * 128
        self.attention3 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256
        self.fc = nn.Linear(num_channels, N)
        self.N = N
        self.K = K
        self.use_cuda = use_cuda

    def forward(self, input, labels):
        x = self.omniglot(input)
        batch_size = labels.size()[0] / (self.N * self.K + 1)
        last_idxs = [(i + 1) * (self.N * self.K + 1) - 1 for i in range(batch_size)]
        if self.use_cuda:
            labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1]))).cuda()
        else:
            labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1])))
        x = torch.cat((x, labels), 1)
        x = x.view((batch_size, self.N * self.K + 1, -1))
        x = self.attention1(x)
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        x = self.fc(x)
        return x
