import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class autoencoder(nn.Module):
    def __init__(self, inp_dim, nclusters=None):
        super(autoencoder, self).__init__()
        self.nclusters = nclusters
        self.encoder = nn.Sequential(
            nn.Linear(inp_dim, 512),
            nn.ReLU(), 
            nn.Linear(512, 128),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.ReLU(), nn.Linear(128, 64),
            nn.ReLU(), nn.Linear(64, 64), 
            nn.ReLU(), nn.Linear(64, 50)
            )
        self.decoder = nn.Sequential(
            nn.Linear(50, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(), 
            nn.Linear(512, inp_dim), nn.ReLU())

        self.cluster_layer = Parameter(torch.Tensor(self.nclusters, 50))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.alpha = 1.0

    def forward(self, x):
        e = self.encoder(x)
        d = self.decoder(e)

        q = 1.0 / (1.0 + torch.sum(torch.pow(e - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return d, e, q

'''
import torch

x = torch.rand(256, 1, 784)
model = autoencoder()
print(model(x).shape)
'''
