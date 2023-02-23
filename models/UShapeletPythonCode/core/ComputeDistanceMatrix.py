import numpy as np
from .findNN import findNN


def ComputeDistanceMatrix(shapelets, data):
    n = data.shape[0]
    F = shapelets.shape[0]
    dis = np.zeros((n, F))
    locations = np.zeros((n, F))

    for j in range(n):
        for i in range(F):
            locations[j, i], dis[j, i] = findNN(data[j,:], shapelets[i,:])
    
    return dis, locations
