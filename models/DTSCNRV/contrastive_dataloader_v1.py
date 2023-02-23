import sys
import torch
import random
import numpy as np
from torch.utils.data import Dataset


def calculate_euclidean(ts1, ts2):
    return np.sqrt(np.sum((ts1 - ts2)**2))


def euclidean(ts):
    dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
    for i in range(ts.shape[0]):
        for j in range(i+1, ts.shape[0]):
            dist_mat[i, j] = calculate_euclidean(ts[i], ts[j])
            dist_mat[j, i] = dist_mat[i, j]
    return dist_mat


class TSDataset(Dataset):
    def __init__(self, x, y, nclusters):
        self.x, self.y = x, y
        self.dist_mat = np.nan_to_num(euclidean(self.x))

    def __len__(self):
        return self.x.shape[0]

    def removeOutliers(self, x, outlierConstant):
        a = np.array(x)
        upper_quartile = np.percentile(a, 75)
        lower_quartile = np.percentile(a, 25)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

        indexList, resultList = [], []
        for i, y in enumerate(a.tolist()):
            if y >= quartileSet[0] and y <= quartileSet[1]:
                resultList.append(y)
                indexList.append(i)        
        return np.array(resultList), np.array(indexList)

    def __getitem__(self, idx):
        filtered_dist, filtered_idxs = self.removeOutliers(self.dist_mat[idx], 1.5)

        closest_idx = torch.Tensor(filtered_idxs)[torch.topk(torch.Tensor(filtered_dist), 11, largest=False, sorted=True, dim=0)[1][1:]]
        farthest_idx = torch.Tensor(filtered_idxs)[torch.topk(torch.Tensor(filtered_dist), 10, largest=True, sorted=True, dim=0)[1]]

        cl_ix = np.random.choice(10, 1)
        fr_ix = np.random.choice(10, 1)

        closest = closest_idx[cl_ix][0]
        farthest = farthest_idx[fr_ix][0]
        
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(np.array(self.x[int(closest)])), torch.from_numpy(np.array(self.x[int(farthest)])), idx

