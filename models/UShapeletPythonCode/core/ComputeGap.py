import numpy as np
from .ComputeGapOne import *


def ComputeGap(dis):
    shapeletsNumber = dis.shape[1]
    maxGap = np.zeros((shapeletsNumber, 1))
    dt = np.zeros((shapeletsNumber, 1))
    ratios = np.zeros((shapeletsNumber, 1))

    for i in range(shapeletsNumber):        
        disSorted = np.sort(dis)
        disSize = len(dis)
        startPoint = int(np.ceil(disSize * 0.167))
        endPoint = int(np.floor(disSize * 0.833))

        for j in range(startPoint, endPoint):
            d = disSorted[j]
            gap, _, _, _, _, ratio = ComputeGapOne(dis[:, i], d)
            if (gap > maxGap[i]):
                maxGap[i] = gap
                dt[i] = d
                ratios[i] = ratio
    
    return maxGap, dt, ratios