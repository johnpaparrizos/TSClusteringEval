import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .contrastive_dataloader_v1 import TSDataset
from .contrastive_loss import SimpleContrastiveLoss


device = "cuda"
def dtscnrv(ts, labels, nclusters, clustering_loss='idec'):
    metric = 'euclidean'

    dataset = TSDataset(ts, labels, nclusters)
    contrastive_loss = SimpleContrastiveLoss()
    batch_size = 128
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = 'resnet'

    if model=='rnn':
        from .models.rnn_model import RNNModel
        model = RNNModel(ts.shape[1], 250, 50, 3, nclusters=nclusters)
    elif model=='fcn':
        from .models.fcn_model import autoencoder
        model = autoencoder(ts.shape[1], nclusters=nclusters)
    elif model=='resnet':
        ts = np.expand_dims(ts, axis=1)
        from .models.resnet_model import AutoEncoder
        model = AutoEncoder(ts.shape[1], ts.shape[2], nclusters=nclusters)
    elif model=='tcn_attention':
        from .models.tcn_attention_model import DeepTimeContrasting, target_distribution 
        model = DeepTimeContrasting(window=ts.shape[1], latent_size=50, ncluster=nclusters, kernel=5, dilations=7, sampling_factor=8, dropout=0.25, layers=1)

    model.to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(300):
        total_loss = 0
        for batch_idx, (x, x_close, x_far, _) in enumerate(tqdm(train_loader)):
            x = x.unsqueeze(1).float().to(device)

            _, a, _  = model(x)
            _, close, _  = model(x_close.reshape(x_close.shape[0]*1, 1, -1).float().to(device))
            _, far, _  = model(x_far.reshape(x_far.shape[0]*1, 1, -1).float().to(device))

            loss = contrastive_loss(a.squeeze(1), close.squeeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()

        print("epoch {} loss={:.4f}".format(epoch, total_loss/(batch_idx+1)))
        
    model.eval()
    hidden = torch.zeros(ts.shape[0], 1, 50)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_idx, (x, _, _, _) in enumerate(train_loader):
        with torch.no_grad():
            x = x.unsqueeze(1).float().to(device)
            _, z, _ = model(x)
            hidden[batch_idx*batch_size:(batch_idx+1)*batch_size, :, :] = z

    kmeans = KMeans(n_clusters=nclusters, n_init=10, random_state=9)
    y_pred = kmeans.fit_predict(hidden.squeeze(1).data.cpu().numpy())

    if clustering_loss == 'idec':
        print('IDEC Training Started .........')

        hidden = None
        y_pred_last = y_pred
        model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

        def target_distribution(q):
            weight = q**2 / q.sum(0)
            return (weight.t() / weight.sum(1)).t()

        model.train()
        for epoch in range(100):
            if epoch % 5 == 0:
                q_vec = torch.zeros(ts.shape[0], nclusters)
                train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                for batch_idx, (x, _, _, _) in enumerate(train_loader):
                    with torch.no_grad():
                        x = x.unsqueeze(1).float().to(device)
                        _, _, q = model(x)
                        q_vec[batch_idx*batch_size:(batch_idx+1)*batch_size, :] = q

                q_vec = q_vec.data
                p = target_distribution(q_vec)

                y_pred = q_vec.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred

            total_loss = 0
            for batch_idx, (x, x_close, x_far, idx) in enumerate(tqdm(train_loader)):
                x = x.unsqueeze(1).float().to(device)

                x_bar, a, q = model(x)
                _, close, _  = model(x_close.reshape(x_close.shape[0]*1, 1, -1).float().to(device))
                _, far, _  = model(x_far.reshape(x_far.shape[0]*1, 1, -1).float().to(device))

                loss = contrastive_loss(a.squeeze(1), close.squeeze(1))

                reconstr_loss = F.mse_loss(x_bar, x)
                kl_loss = F.kl_div(q.log().to(device), p[idx].to(device))
                idec_loss = 0.1 * kl_loss + loss + reconstr_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss = total_loss + loss.item()
            print("epoch {} loss={:.4f}".format(epoch, total_loss/(batch_idx+1)))


        model.eval()
        hidden = torch.zeros(ts.shape[0], 1, 50)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch_idx, (x, _, _, _) in enumerate(train_loader):
            with torch.no_grad():
                x = x.unsqueeze(1).float().to(device)
                _, z, _ = model(x)
                hidden[batch_idx*batch_size:(batch_idx+1)*batch_size, :, :] = z

        kmeans = KMeans(n_clusters=nclusters, n_init=10, random_state=9)
        y_pred = kmeans.fit_predict(hidden.squeeze(1).data.cpu().numpy())

    return y_pred

