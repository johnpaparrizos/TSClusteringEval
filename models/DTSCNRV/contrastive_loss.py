import torch
import torch.nn as nn
import torch.nn.functional as F
 

class SimpleContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05, verbose=True):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
            
        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * emb_i.shape[0], )).to(emb_i.device).scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
                )    

            loss_ij = -torch.log(numerator / denominator)
                
            return loss_ij.squeeze(0)

        N = emb_i.shape[0]
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss


