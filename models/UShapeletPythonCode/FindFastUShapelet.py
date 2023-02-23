import time
import numpy as np
from tqdm import tqdm
from .SortUshapelets import *
from .GetActualGap import *


def FindFastUShapelet(data, classLabels, dataFileName, sLen):
    _, bestShapelets, _ = SortUshapelets(data, sLen)

    tsNumber, dataWidth = data.shape
    sPosSize = dataWidth - sLen + 1
    sNum = tsNumber*sPosSize
    maxGapCurrent = 0
    bestShIndex = 0
    clsNum = len(set(classLabels))
    
    if (sNum > 100):
        onePercentData = np.round(sNum*0.001)
        riNew = np.zeros((bestShapelets.shape[0], 1))
        i = 0
        clsNum = len(set(classLabels))  

        for i in range(int(onePercentData)):
            gap, ri, _ = GetActualGap(sLen, bestShapelets, i, data, classLabels, clsNum)
            bestShapelets[i, 2] = gap
            riNew[i] = ri
            if (gap > maxGapCurrent):
                bestShIndex = i
                maxGapCurrent = gap

        computationStop = i
    
    return bestShIndex, bestShapelets, sLen, clsNum, _, _, _, _
