import torch
from torch import nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, input_channels, out, inp, latent_dim=64, kernel_size=[4, 4], strides=[1, 1], n_channels=[32, 128]):
        super().__init__()
        self.latent_transform1 = nn.Linear(inp, 256)
        self.latent_transform2 = nn.Linear(256, 128)
        self.latent_transform3 = nn.Linear(128, latent_dim)
        '''
        self.conv1 = nn.Conv2d(input_channels, n_channels[0], (1, 4), stride=strides[0])
        self.maxpool1 = nn.MaxPool2d(1, 2)
        self.conv2 = nn.Conv2d(n_channels[0], n_channels[1], (1, 4), stride=strides[1])
        self.maxpool2 = nn.MaxPool2d(1, 2)
        self.latent_transform = nn.Linear(out, latent_dim)
        '''

    def forward(self, x):
        '''
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.shape[0], -1)
        out = self.latent_transform(x)
        return out
        '''
        x = self.latent_transform1(x)
        x = self.latent_transform2(x)
        x = self.latent_transform3(x)
        return x



class ConvDecoder(nn.Module):
    def __init__(self, output_channels, inp, output_size=(28, 28), latent_dim=64, flatten_shape=[4, 4, 128],
                 kernel_size=[4, 3], strides=[1, 1], n_channels=[128, 128]):
        super().__init__()
        '''
        self.latent_transform = nn.Linear(latent_dim, flatten_shape[0]*flatten_shape[1]*flatten_shape[2])
        self.deconv1 = nn.Conv2d(flatten_shape[-1], n_channels[0], (1, 4), stride=strides[0], padding=1)
        self.deconv2 = nn.Conv2d(n_channels[0], n_channels[-1], (1, 3), stride=strides[-1], padding=1)
        self.out_transform = nn.Conv2d(n_channels[-1], output_channels, 1, 1)
        self.flatten_shape = flatten_shape
        self.output_size = output_size
        '''
        self.latent_transform1 = nn.Linear(latent_dim, 128)
        self.latent_transform2 = nn.Linear(128, 256)
        self.latent_transform3 = nn.Linear(256, inp)


    def forward(self, x):
        '''
        x = self.latent_transform(x)
        x = x.view(x.shape[0], self.flatten_shape[2], self.flatten_shape[1], self.flatten_shape[0])
        x = F.interpolate(x, scale_factor=2)
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = F.interpolate(x, size=self.output_size)
        out = torch.sigmoid(self.out_transform(x))
        return out
        '''
        x = self.latent_transform1(x)
        x = self.latent_transform2(x)
        x = self.latent_transform3(x)
        return x 


class SOMVAE(nn.Module):
    def __init__(self, encoder, decoder_q, decoder_e, latent_dim=64, som_dim=[8,8],
                 input_length=28, input_channels=28, beta=1, gamma=1, alpha=1, tau=1):
        super().__init__()
        self.encoder = encoder
        self.decoder_e = decoder_e
        self.decoder_q = decoder_q
        self.input_length = input_length
        self.input_channels = input_channels
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.som_dim = som_dim
        self.latent_dim = latent_dim
        self.embeddings = nn.Parameter(torch.randn(som_dim[0], som_dim[1], latent_dim)*0.05)
        self.mse_loss = nn.MSELoss()
        
        probs_raw = torch.zeros(*(som_dim + som_dim))
        probs_pos = torch.exp(probs_raw)
        probs_sum = torch.sum(probs_pos, dim=[-1, -2], keepdim=True)
        self.probs = nn.Parameter(probs_pos/probs_sum)
        
    def forward(self, x):
        z_e = self.encoder(x)
        z_dist = (z_e.unsqueeze(1).unsqueeze(2) - self.embeddings.unsqueeze(0))**2
        z_dist_sum = torch.sum(z_dist, dim=-1)
        z_dist_flat = z_dist_sum.view(x.shape[0], -1)

        k = torch.argmin(z_dist_flat, dim=-1)

        k_1 = k // self.som_dim[1]
        k_2 = k % self.som_dim[1]
        
        k_stacked = torch.stack([k_1, k_2], dim=1)
        z_q = self._gather_nd(self.embeddings, k_stacked)
        
        k_1 = k // self.som_dim[1]
        k_2 = k % self.som_dim[1]
        k_stacked = torch.stack([k_1, k_2], dim=1)
        
        k1_not_top = k_1 < self.som_dim[0] - 1
        k1_not_bottom = k_1 > 0
        k2_not_right = k_2 < self.som_dim[1] - 1
        k2_not_left = k_2 > 0
        
        k1_up = torch.where(k1_not_top, k_1 + 1, k_1)
        k1_down = torch.where(k1_not_bottom, k_1 - 1, k_1)
        k2_right = torch.where(k2_not_right, k_2 + 1, k_2)
        k2_left = torch.where(k2_not_left, k_2 - 1, k_2)
        
        z_q_up = torch.zeros(x.shape[0], self.latent_dim).cuda()
        z_q_up_ = self._gather_nd(self.embeddings, torch.stack([k1_up, k_2], dim=1))
        z_q_up[k1_not_top == 1] = z_q_up_[k1_not_top == 1]
        
        z_q_down = torch.zeros(x.shape[0], self.latent_dim).cuda()
        z_q_down_ = self._gather_nd(self.embeddings, torch.stack([k1_down, k_2], dim=1))
        z_q_down[k1_not_bottom == 1] = z_q_down_[k1_not_bottom == 1]
        
        z_q_right = torch.zeros(x.shape[0], self.latent_dim).cuda()
        z_q_right_ =  self._gather_nd(self.embeddings, torch.stack([k_1, k2_right], dim=1))
        z_q_right[k2_not_right == 1] == z_q_right_[k2_not_right == 1]
        
        z_q_left = torch.zeros(x.shape[0], self.latent_dim).cuda()
        z_q_left_ = self._gather_nd(self.embeddings, torch.stack([k_1, k2_left], dim=1))
        z_q_left[k2_not_left == 1] = z_q_left_[k2_not_left == 1]
        
        z_q_neighbors = torch.stack([z_q, z_q_up, z_q_down, z_q_right, z_q_left], dim=1)
        
        x_q = self.decoder_q(z_q)
        x_e = self.decoder_e(z_e)
        
        return x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat
        
        
    def _loss_reconstruct(self, x, x_e, x_q):
        l_e = self.mse_loss(x, x_e)
        l_q = self.mse_loss(x, x_q)
        mse_l = l_e + l_q
        return mse_l
    
    def _loss_commit(self, z_e, z_q):
        commit_l = self.mse_loss(z_e, z_q)
        return commit_l
    
    def _loss_som(self, z_e, z_q_neighbors):
        z_e = z_e.detach()
        som_l = torch.mean((z_e.unsqueeze(1) - z_q_neighbors)**2)
        return som_l
    
    def loss_prob(self, k):
        k_1 = k // self.som_dim[1]
        k_2 = k % self.som_dim[1]
        k_1_old = torch.cat([k_1[0:1], k_1[:-1]], dim=0)
        k_2_old = torch.cat([k_2[0:1], k_2[:-1]], dim=0)
        k_stacked = torch.stack([k_1_old, k_2_old, k_1, k_2], dim=1)
        
        probs_raw = self.probs
        probs_pos = torch.exp(probs_raw)
        probs_sum = torch.sum(probs_pos, dim=[-1, -2], keepdim=True)
        self.probs = nn.Parameter(probs_pos/probs_sum)
        
        transitions_all = self._gather_nd(self.probs, k_stacked)
        prob_l = -self.gamma*torch.mean(torch.log(transitions_all))
        return prob_l
    
    def _loss_z_prob(self, k, z_dist_flat):
        k_1 = k // self.som_dim[1]
        k_2 = k % self.som_dim[1]
        k_1_old = torch.cat([k_1[0:1], k_1[:-1]], dim=0)
        k_2_old = torch.cat([k_2[0:1], k_2[:-1]], dim=0)
        k_stacked = torch.stack([k_1_old, k_2_old], dim=1)
        
        probs_raw = self.probs
        probs_pos = torch.exp(probs_raw)
        probs_sum = torch.sum(probs_pos, dim=[-1, -2], keepdim=True)
        self.probs = nn.Parameter(probs_pos/probs_sum)
        
        out_probabilities_old = self._gather_nd(self.probs, k_stacked)
        out_probabilities_flat = out_probabilities_old.view(k.shape[0], -1)
        weighted_z_dist_prob = z_dist_flat*out_probabilities_flat
        prob_z_l = torch.mean(weighted_z_dist_prob)
        return prob_z_l
    
    def loss(self, x, x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat):
        mse_l = self._loss_reconstruct(x, x_e, x_q)
        commit_l = self._loss_commit(z_e, z_q)
        som_l = self._loss_som(z_e, z_q_neighbors)
        prob_l = self.loss_prob(k)
        prob_z_l = self._loss_z_prob(k, z_dist_flat)
        l = mse_l + self.alpha*commit_l + self.beta*som_l + self.gamma*prob_l + self.tau*prob_z_l
        return l
        
    def _gather_nd(self, params, idx):
        idx = idx.long()
        outputs = []
        for i in range(len(idx)):
            outputs.append(params[[idx[i][j] for j in range(idx.shape[1])]])
        outputs = torch.stack(outputs)
        return outputs