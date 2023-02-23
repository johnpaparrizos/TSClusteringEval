import numpy as np
from .ComputeDistanceMatrix import *
from .ComputeGap import *

def ComputeGapRaw(shapelets, dataset):
    dis, locations = ComputeDistanceMatrix(shapelets, dataset) 
    dis = np.real(dis)
    maxGap, dt, _ = ComputeGap(dis)

    return maxGap, dt, dis, locations