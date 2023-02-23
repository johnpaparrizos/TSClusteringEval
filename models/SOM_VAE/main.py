import torch
import torch as nn
from .som_vae import *
from tqdm import tqdm

def som_vae(x, y, ncluster):
    learning_rate = 0.0005
    alpha = 1.0
    beta = 0.9
    gamma = 1.8
    tau = 1.4
    decay_factor = 0.9
    interactive = True
    save_model = False
    time_series = True

    if x.shape[1] % 2 ==0:
        o = (x.shape[1]//4)-2
    else:
        o = (x.shape[1]//4)-1

    model = SOMVAE(ConvEncoder(1, out=o, inp=x.shape[1], n_channels=[1, 1], kernel_size=[1, 4]), ConvDecoder(1, inp=x.shape[1], output_size=(1, x.shape[1]), kernel_size=[1, 4], n_channels=[128, 128]), ConvDecoder(1, inp=x.shape[1], output_size=(1, x.shape[1]), n_channels=[128, 128]), alpha=alpha, beta=beta, gamma=gamma, tau=tau, som_dim=[1, ncluster]).cuda()
    model.train()

    model_param_list = nn.ParameterList()
    for p in model.named_parameters():
        if p[0] != 'probs':
            model_param_list.append(p[1])
            
    probs_param_list = nn.ParameterList()
    for p in model.named_parameters():
        if p[0] == 'probs':
            probs_param_list.append(p[1])


    opt_model = torch.optim.Adam(model_param_list, lr=learning_rate)
    opt_probs = torch.optim.Adam(probs_param_list, lr=learning_rate)

    sc_opt_model = torch.optim.lr_scheduler.StepLR(opt_model, 1000, decay_factor)
    sc_opt_probs = torch.optim.lr_scheduler.StepLR(opt_probs, 1000, decay_factor)

    #x = torch.from_numpy(x).float().unsqueeze(1).unsqueeze(1).to("cuda")
    x = torch.from_numpy(x).float().to("cuda") 

    for e in tqdm(range(500)):
        opt_model.zero_grad()
        opt_probs.zero_grad()
        
        x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat = model(x)
        l = model.loss(x, x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat)
        l_prob = model.loss_prob(k)
        l.backward()
        opt_model.step()
        l_prob.backward()
        opt_probs.step()
        sc_opt_model.step()
        sc_opt_probs.step()
        '''
        if e%10 == 0:
            with torch.no_grad():
                x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat = model(x)
                l = model.loss(x, x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat)
                #print("Loss: ", l.item())
        '''

    #model.eval()
    x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat = model(x)
    #print(k.detach().cpu().numpy().tolist())
    return k.detach().cpu().numpy().tolist()
