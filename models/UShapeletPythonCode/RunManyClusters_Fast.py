#from core import *
import numpy as np
from sklearn import metrics
from .FindFastUShapelet import *


def UShapelet(data, classLabels):
    #sLen = int(data.shape[1]/2)
    #sLen = int((data.shape[1]*7)/20) 
    sLen = int((data.shape[1]*3)/20)

    allClassLabels = classLabels.copy()
    minGap = 0
    dataSize = data.shape[0]
    uShapelets = []
    gaps = []
    remainingInd = np.array([idx for idx in range(dataSize)])
    labelsResult = np.zeros((dataSize, 1))
    currentClusterNum = 1
    totalTime = 0

    while (len(remainingInd) > 3):
        bestShIndex, bestShapelets, sLen, clsNum, _, _, _, _ = FindFastUShapelet(data, classLabels, None, sLen)

        maxGapCurrent = bestShapelets[bestShIndex, 2]
        _, _, newIDX = GetActualGap(sLen, bestShapelets, bestShIndex, data, classLabels, clsNum)

        ts = remainingInd[int(bestShapelets[bestShIndex, 0])]
        loc = bestShapelets[bestShIndex, 1]

        bsfUsh = [ts, loc, sLen, maxGapCurrent]
        bsfCurrentIDX = newIDX

        gaps.append(maxGapCurrent)
        if (minGap == 0):
            if (maxGapCurrent > 0):
                minGap = maxGapCurrent
            else:
                break
        else:
            if (minGap / 2 > maxGapCurrent):
                break
        
        indToDelete = np.argwhere(bsfCurrentIDX)

        data = np.delete(data, np.concatenate(indToDelete), axis=0)
        uShapelets.append(bsfUsh)
        
        if (len(classLabels) > 0):
            classLabels = np.delete(classLabels, np.concatenate(indToDelete), axis=0)

        labelsResult[remainingInd[indToDelete]] = currentClusterNum
        remainingInd = np.delete(remainingInd, np.concatenate(indToDelete), axis=0) 
        currentClusterNum = currentClusterNum + 1

    return labelsResult[:, 0]
