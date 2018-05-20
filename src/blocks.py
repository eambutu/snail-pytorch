import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CasualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, dilation=1, groups=1, bias=True):
        super(CasualConv1d, self).__init__()
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                                padding, dilation, groups, bias)

    def forward(self, input):
        # Takes something of shape (N, in_channels, T),
        # returns (N, out_channels, T)
        out = self.conv1d(input)
        return out[:, :, :-1]

class DenseBlock(nn.Module):
    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.casualconv1 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.casualconv2 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)

    def forward(self, input):
        # input is dimensions (N, in_channels, T)
        xf = self.casualconv1(input)
        xg = self.casualconv2(input)
        activations = F.tanh(xf) * F.sigmoid(xg) # shape: (N, filters, T)
        return torch.cat((input, activations), dim=1)
        
class TCBlock(nn.Module):
    def __init__(self, in_channels, seq_length, filters):
        super(TCBlock, self).__init__()
        self.dense_blocks = [DenseBlock(in_channels, 2 ** (i+1), filters)
                             for i in range(math.ceil(math.log(seq_length)))]

    def forward(self, input):
        for block in self.dense_blocks:
            input = block(input)
        return input

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlobk, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        # input is dim (N, in_channels, T)
        mask = np.array([[1 if i>j else 0 for i in range(input.shape[2])] for j in range(input.shape[2])])
        mask = torch.ByteTensor(mask)

        input = torch.transpose(input, 1, 2) # shape: (N, T, in_channels)
        keys = self.linear_keys(input) # shape: (N, T, key_size)
        query = self.linear_query(input) # shape: (N, T, key_size)
        values = self.linear_values(input) # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        temp = temp.data.masked_fill_(mask, -float('inf'))
        temp = F.softmax(temp / self.sqrt_key_size, dim=1) # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = torch.bmm(temp, values) # shape: (N, T, value_size)
        return torch.cat((input, torch.transpose(temp, 1, 2)), dim=1) # shape: (N, in_channels + value_size, T)
