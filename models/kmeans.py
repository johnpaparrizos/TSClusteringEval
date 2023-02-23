import numpy as np
from tqdm import tqdm
import multiprocessing
from dtaidistance import dtw
from numpy.random import randint


def kMeans(X, k):
    m = X.shape[0]
    idx = randint(0, k, size=m)
    cent = np.zeros((k, X.shape[1]))
    D = np.zeros((m, k))

    for it in range(300):
        old_idx = idx

        pool = multiprocessing.Pool(k)
        result = pool.map(kmeans_centroid, [[idx, X, j, cent[j, :]] for j in range(k)])
        for j, c in enumerate(result):
            cent[j, :] = c
        pool.close()

        pool = multiprocessing.Pool()
        args = []
        for p in range(m):
            for q in range(k):
                args.append([X[p, :], cent[q, :]])
        result = pool.map(euclidean, args)
        pool.close()
        
        r = 0
        for p in range(m):
            for q in range(k):
                D[p, q] = result[r]
                r = r + 1
        idx = D.argmin(1)

        if np.array_equal(old_idx, idx):
            break

    return idx, cent


def kmeans_centroid(data):
    a = []
    idx, X, j, cur_center = data[0], data[1], data[2], data[3]
    for i in range(len(idx)):
        if idx[i] == j:
            opt_x = X[i]
            a.append(opt_x)
    a = np.array(a)
    
    if len(a) == 0:
        return np.zeros((1, X.shape[1]))

    return np.mean(a, axis=0)


def euclidean(data):
    x, y = data[0], data[1]
    return np.sqrt(np.sum((x-y)**2))

