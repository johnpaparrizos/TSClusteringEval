import torch
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class autoencoder(nn.Module):
    def __init__(self, n_clus, length):
        super(autoencoder,self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv1d(1,64,kernel_size=3,stride=1,padding=1), 
            nn.BatchNorm1d(64),
            nn.Sigmoid())
        self.encoder2 =  nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3,stride=1,padding=1), 
            nn.BatchNorm1d(64),
            nn.Sigmoid())
        self.encoder3 =  nn.Sequential(
            nn.Conv1d(64,1,kernel_size=3,stride=1,padding=1), 
            nn.BatchNorm1d(1),
            nn.Sigmoid())
        self.encoder4 = nn.Sequential(
            nn.Conv1d(1,1,kernel_size=2,stride=2,padding=0), 
            nn.BatchNorm1d(1),
            nn.Sigmoid())
        if length%2 == 0:
            self.decoder4 = nn.Sequential(
               nn.ConvTranspose1d(1,1,kernel_size=2,stride=2,padding=0), 
                nn.BatchNorm1d(1),
                nn.Sigmoid())
        else:
            self.decoder4 = nn.Sequential(
               nn.ConvTranspose1d(1,1,kernel_size=2,stride=2,padding=0,output_padding=1),
                nn.BatchNorm1d(1),
                nn.Sigmoid())
        self.decoder3 = nn.Sequential(
            nn.Conv1d(1,64,kernel_size=3,stride=1,padding=1), 
            nn.BatchNorm1d(64),
            nn.Sigmoid()
            )
        self.decoder2 =  nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3,stride=1,padding=1), 
            nn.BatchNorm1d(64),
            nn.Sigmoid())
        self.decoder1 = nn.Sequential(
            nn.Conv1d(64,1,kernel_size=3,stride=1,padding=1), 
            nn.BatchNorm1d(1),
            nn.Sigmoid())
        self.n_clus = n_clus
        self.em = nn.Linear(int(length/2), self.n_clus)


    def forward(self,x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        dec4 = self.decoder4(enc4)
        dec3 = self.decoder3(dec4)
        dec2 = self.decoder2(dec3)
        dec1 = self.decoder1(dec2)
        x = enc4.squeeze(1)

        em = self.em(x)
        em = F.softmax(em, dim=1)
        return (enc1,enc2,enc3,enc4,dec4,dec3,dec2,dec1,em)

    def obtain_batch_loss(self,x,beta):
        enc1,enc2,enc3,enc4,dec4,dec3,dec2,dec1,em=self.forward(x)
        #print(enc1.shape,enc2.shape,enc3.shape,x.shape)
        #print(dec2.shape,dec3.shape,dec4.shape,dec1.shape)
        loss1 = nn.MSELoss()(dec1,x)
        loss2 = nn.MSELoss()(dec2,enc1)
        loss3 = nn.MSELoss()(dec3,enc2)
        loss4 = nn.MSELoss()(dec4,enc3)
        r_sumoveri = torch.sqrt(torch.sum(em,dim=0))
        num = em/r_sumoveri   
        q = num/(torch.sum(num,dim=0))
        clus_loss = -1*torch.mean(q*torch.log(em))/self.n_clus
        recons_loss = loss1+beta*(loss2+loss3+loss4)
        return (clus_loss, recons_loss)

    def encode(self,x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        return enc4


class TSDataset(Dataset):
    def __init__(self, x, y, nclusters):
        self.x, self.y = x, y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]



def depict(ts, labels, nclusters):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size=64

    dataset = TSDataset(ts, labels, nclusters)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    autoenc = autoencoder(nclusters, ts.shape[1]).to(device)
    optim = torch.optim.Adam(autoenc.parameters(), lr=1e-3)

    epochs = 100
    for epoch in range(epochs):
        running_clus_loss=0
        running_recons_loss=0
        num_images=0
        for i,(img, _) in enumerate(train_loader):
            img = img.unsqueeze(1).float().to(device)
            #img = img.to(device)
            optim.zero_grad()
            clus_loss, recons_loss = autoenc.obtain_batch_loss(img, 0.8)
            loss = recons_loss + clus_loss
            loss.backward()
            optim.step()
            running_clus_loss = running_clus_loss + clus_loss.item()*len(img)
            running_recons_loss = running_recons_loss + recons_loss.item()*len(img)
            num_images= num_images+len(img)
        
        print('epoch: '+str(epoch)+' clus_loss: '+str(running_clus_loss/num_images)+' recons_loss: '+str(running_recons_loss/num_images))


    encd=np.array([])
    autoenc.eval()
    for i, (img, label) in enumerate(train_loader):
        img = img.unsqueeze(1).float().to(device)
        encoded = autoenc.encode(img.to(device)).cpu().detach()
        if(i==0):
            encd=np.array(encoded)
        else:
            encd = np.concatenate([encd,np.array(encoded)])

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=nclusters, n_init=10, random_state=9)
    y_pred = kmeans.fit_predict(encd.squeeze(1))

    return y_pred

