import torch
import random
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepTimeContrasting(nn.Module):
    def __init__(self, window, latent_size, ncluster, kernel, dilations, sampling_factor, dropout=0.25, layers=3):
        super(DeepTimeContrasting, self).__init__()
        self.window = window
        self.n_multiv = 1
        self.n_kernels = 32
        self.d_model = 512
        self.kernel = kernel
        self.sampling_factor = sampling_factor
        self.dilations = dilations
        self.d_inner = 2048
        self.n_head = 8
        self.nclusters = ncluster
        self.latent_size = latent_size
        self.d_k = 64
        self.d_v = 64
        self.channel_sizes = [30]*self.dilations
        self.kernel_size = 7
        self.drop_prob = dropout
        self.n_layers = layers

        self.tcn = TemporalConvNet(
            self.n_multiv, self.channel_sizes, kernel_size=self.kernel_size, dropout=self.drop_prob)
        self.conv1 = nn.Conv1d(self.channel_sizes[-1], self.n_multiv, 1)

        self.linear1 = nn.Linear(self.window, self.d_model)
        self.layer_stack1 = nn.ModuleList([
            EncoderLayer(self.d_model, self.d_inner, self.n_head,
                         self.d_k, self.d_v, dropout=self.drop_prob)
            for _ in range(self.n_layers)])
        self.pool1 = nn.AvgPool1d(
            kernel_size=self.sampling_factor, stride=None)
        self.linear2 = nn.Linear(
            int(self.d_model/self.sampling_factor), self.latent_size)

        self.cl_linear = nn.Linear(self.latent_size, self.latent_size)
        self.cl_relu = nn.ReLU()

        self.upsample1 = nn.Upsample(
            scale_factor=sampling_factor, mode='nearest')
        self.layer_stack2 = nn.ModuleList([
            EncoderLayer(self.latent_size, self.d_inner, self.n_head,
                         self.d_k, self.d_v, dropout=self.drop_prob)
            for _ in range(self.n_layers)])
        self.linear3 = nn.Linear(self.latent_size, self.d_model)
        self.conv2 = nn.Conv1d(self.n_multiv, self.n_multiv, 1)
        self.linear4 = nn.Linear(self.d_model, self.window)

        self.cluster_layer = Parameter(
            torch.Tensor(self.nclusters, self.latent_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.alpha = 1.0

    def forward(self, x):
        sf_output = self.tcn(x)
        sf_output = self.conv1(sf_output)
        sf_output = self.linear1(sf_output)

        for enc_layer in self.layer_stack1:
            sf_output, _ = enc_layer(sf_output)

        sf_output = self.pool1(sf_output)

        latent_vector = nn.ReLU()(self.linear2(sf_output))
        #latent_vector = nn.ReLU()(F.normalize(self.linear2(sf_output), dim=2))
        #latent_vector = latent_vector + torch.randn(latent_vector.shape).cuda()
        #latent_vector = self.linear2(sf_output)
        #latent_vector = F.normalize(self.cl_relu(self.cl_linear(latent_vector)) + torch.randn(latent_vector.shape).cuda(), dim=2)

        sf_output = self.upsample1(latent_vector)
        for enc_layer in self.layer_stack2:
            sf_output, _ = enc_layer(latent_vector)

        sf_output = self.linear3(sf_output)
        sf_output = self.conv2(sf_output)
        sf_output = self.linear4(sf_output)

        q = 1.0 / (1.0 + torch.sum(
            torch.pow(latent_vector - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return sf_output, latent_vector, q


