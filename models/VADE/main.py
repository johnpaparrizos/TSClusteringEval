import torch
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data
from .vade import VaDE, lossfun


class TSDataset(Dataset):
    def __init__(self, x, y, nclusters):
        self.x, self.y = x, y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]


def train(model, data_loader, optimizer, device, epoch):
    model.train()

    total_loss = 0
    for x, _ in data_loader:
        x = x.unsqueeze(1).float().to(device)
        recon_x, mu, logvar = model(x)
        loss = lossfun(model, x, recon_x, mu, logvar)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model, total_loss

def test(model, data_loader, device, epoch, nclusters):
    model.eval()

    preds = []
    gain = torch.zeros((nclusters, nclusters), dtype=torch.int, device=device)
    with torch.no_grad():
        for xs, ts in data_loader:
            xs, ts = xs.unsqueeze(1).to(device).float(), ts.to(device)
            ys = model.classify(xs)
            preds.append(ys.detach().cpu().numpy())

    preds = np.concatenate(preds, 0)
    return preds

   
def vade(ts, labels, nclusters):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size=64

    dataset = TSDataset(ts, labels, nclusters)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model = VaDE(nclusters, ts.shape[1], nclusters)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    for epoch in range(1, 100 + 1):
        model, _ = train(model, train_loader, optimizer, device, epoch)
        preds = test(model, train_loader, device, epoch, nclusters)
        #lr_scheduler.step()

    return preds

