import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from einops import rearrange
from sklearn.cluster import KMeans
from transformers import GPT2Model, GPT2Config


class GPT4TS(nn.Module):
    def __init__(self, configs, slen, device):
        super(GPT4TS, self).__init__()
        self.patch_size = 1
        self.stride = 1
        self.seq_len = slen
        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        self.in_layer = nn.Linear(self.patch_size, 768)

        # Initialize GPT-2 with the new configuration
        #custom_config = GPT2Config()
        #custom_config.n_positions = slen+1  # Change this to your desired sequence length
        #self.gpt2 = GPT2Model(custom_config)
        #print(custom_config)

        custom_config = GPT2Config.from_pretrained('gpt2')
        custom_config.n_positions = slen + 1  # Change this to your desired sequence length
        self.gpt2 = GPT2Model(custom_config)

        # Load pre-trained weights
        pretrained_model = GPT2Model.from_pretrained('gpt2')

        # Copy weights from the pre-trained model to the custom model
        self.gpt2.wte.weight.data = pretrained_model.wte.weight.data

        # Adjust the position embeddings
        if custom_config.n_positions > 1024:
            new_wpe = nn.Parameter(torch.zeros(custom_config.n_positions, self.gpt2.wpe.embedding_dim))
            new_wpe.data[:1024] = pretrained_model.wpe.weight.data

            # Interpolate positional embeddings
            interp = nn.functional.interpolate(pretrained_model.wpe.weight.data.unsqueeze(0).permute(0, 2, 1), 
                                               size=custom_config.n_positions, 
                                               mode='linear')
            new_wpe.data = interp.squeeze(0).permute(1, 0)
        else:
            new_wpe = nn.Parameter(pretrained_model.wpe.weight.data[:custom_config.n_positions])
        
        self.gpt2.wpe.weight = new_wpe

        for i, layer in enumerate(pretrained_model.h):
            if i >= custom_config.n_layer:
                break
            self.gpt2.h[i].load_state_dict(layer.state_dict())

        self.hidden_layer = nn.Linear(768 * self.patch_num, 200)

        self.out_layer = nn.Linear(200, slen)

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if "ln" in name or "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.to(device)
        self.train()

    def forward(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev
        x = rearrange(x, "b l m -> b m l")

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, "b m n p -> (b m) n p")

        x = self.in_layer(x)

        outputs = self.gpt2(inputs_embeds=x).last_hidden_state

        hidden = self.hidden_layer(outputs.reshape(B * M, -1))

        outputs = self.out_layer(hidden)
        outputs = rearrange(outputs, "(b m) l -> b l m", b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return hidden, outputs


def ofa(ts, labels, nclust):
    ts, labels, nclust = dl.load(d)
    slen = int(ts.shape[1])

    start = time.time()
    gpt4_model = GPT4TS(GPT2Config, slen, 'cuda')
    mse_loss = nn.MSELoss()
    optim = torch.optim.Adam(gpt4_model.parameters(), lr=1e-3)

    gpt4_model.train()
    for t in tqdm(ts):
        t = torch.tensor(t).unsqueeze(0).unsqueeze(2).to(torch.float32).cuda()

        optim.zero_grad()

        hidden, output = gpt4_model(t)
        loss = mse_loss(output, t)

        loss.backward()
        optim.step()


    gpt4_model.eval()
    rep = []
    for t in tqdm(ts):
        t = torch.tensor(t).unsqueeze(0).unsqueeze(2).to(torch.float32).cuda()

        hidden, output = gpt4_model(t)
        
        rep.append(hidden.squeeze(dim=0).detach().cpu().numpy())

    rep = np.array(rep)
    print(rep.shape)

    kmeans = KMeans(n_clusters=nclust, init='random', n_init=1).fit(rep)
    pred = kmeans.labels_

    return pred
