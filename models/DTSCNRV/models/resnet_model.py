import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Conv1dPadSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(Conv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)
        return net


class ResBlock(nn.Module):
    def __init__(self, inp, n_features):
        super(ResBlock, self).__init__()

        self.conv_x1 = Conv1dPadSame(inp, n_features, 8, 1, 1)
        self.conv_x2 = nn.BatchNorm1d(n_features)
        self.conv_x3 = nn.ReLU()

        self.conv_y1 = Conv1dPadSame(n_features, n_features, 5, 1, 1)
        self.conv_y2 = nn.BatchNorm1d(n_features)
        self.conv_y3 = nn.ReLU()

        self.conv_z1 = Conv1dPadSame(n_features, n_features, 3, 1, 1)
        self.conv_z2 = nn.BatchNorm1d(n_features)
        self.conv_z3 = nn.ReLU()

        self.shortcut_y1 = Conv1dPadSame(inp, n_features, 1, 1, 1)
        self.shortcut_y2 = nn.BatchNorm1d(n_features)

        self.output_block_1 = nn.ReLU()


    def forward(self, x):
        c_x = self.conv_x1(x)
        c_x = self.conv_x2(c_x)
        c_x = self.conv_x3(c_x)

        c_y = self.conv_y1(c_x)
        c_y = self.conv_y2(c_y)
        c_y = self.conv_y3(c_y)

        c_z = self.conv_z1(c_y)
        c_z = self.conv_z2(c_z)
        c_z = self.conv_z3(c_z)

        s = self.shortcut_y1(x)
        s = self.shortcut_y2(s)
        
        o = torch.sum(torch.stack([s, c_z]), dim=0)
        o = self.output_block_1(o)

        return o


class AutoEncoder(nn.Module):
    def __init__(self, inp1, inp2, nclusters=None):
        super(AutoEncoder, self).__init__()
        self.nclusters = nclusters
        self.res_block1 = ResBlock(inp1, 64)
        self.res_block2 = ResBlock(64, 128)
        self.res_block3 = ResBlock(128, 50)
        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(p=0.5)

        self.dense1 = nn.Linear(50, inp2)
        self.res_block4 = ResBlock(inp1, 64)
        self.res_block5 = ResBlock(64, 128)
        self.res_block6 = ResBlock(128, 128)

        self.conv = Conv1dPadSame(128, inp1, 3, 1, 1)

        self.cluster_layer = Parameter(torch.Tensor(self.nclusters, 50))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.alpha = 1.0

    def forward(self, x):
        r1 = self.res_block1(x)
        r2 = self.res_block2(r1)
        r3 = self.res_block3(r2)
        a1 = self.avgpool1(r3).squeeze(2).unsqueeze(1)
        d1 = self.dropout1(a1)
        d2 = self.dense1(d1)
        r4 = self.res_block4(d2)
        r5 = self.res_block5(r4)
        r6 = self.res_block6(r5)
        c = self.conv(r6)
    
        q = 1.0 / (1.0 + torch.sum(torch.pow(d1 - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return c, d1, q


'''
import torch

a = torch.rand(256, 1, 100)
model = AutoEncoder(a.shape[1], a.shape[2])
model(a)
'''
