import numpy as np
from tqdm import tqdm
from .GetRandomProjectionsMatrix import *

def SortUshapelets(data, sLen):
    tsNumber, dataWidth = data.shape

    lb = max(0.1*tsNumber, 2)
    ub = tsNumber * 0.9

    sPosSize = dataWidth - sLen + 1
    sNum = tsNumber * sPosSize

    zScoredUshapelets = np.zeros((sNum, sLen + 2))
    SAX_shapelets_TS = np.zeros((sNum, 3))
    curRow = 0

    for i in range(tsNumber):
        ts = data[i, :]
        for j in range(sPosSize):
            SAX_shapelets_TS[curRow, 0] = i
            SAX_shapelets_TS[curRow, 1] = j
            sub_section = ts[j:j+sLen]
            zScoredUshapelets[curRow, 0] = i
            zScoredUshapelets[curRow, 1] = j
            if (sub_section.std(ddof=1) > 0.0000001):
                zScoredUshapelets[curRow, 2:] = (sub_section - sub_section.mean())/(sub_section.std(ddof=1))
            curRow = curRow + 1
    
    uShTS = GetRandomProjectionsMatrix(zScoredUshapelets, sLen, sNum)
    SAX_shapelets_TS = np.concatenate((SAX_shapelets_TS, uShTS), axis=1)
    SAX_shapelets_TS_backup = SAX_shapelets_TS
    hashesTotal = SAX_shapelets_TS.shape[1] - 3

    rowsToDelete = np.ones((sNum))

    for i in range(sNum):
        #print(i, ub, lb, len(np.argwhere((SAX_shapelets_TS[i, 3:] > ub) | (SAX_shapelets_TS[i, 3:] < lb))), hashesTotal*0.5)
        if (len(np.argwhere((SAX_shapelets_TS[i, 3:] > ub) | (SAX_shapelets_TS[i, 3:] < lb))) > hashesTotal*0.5):
            rowsToDelete[i] = 0
    
    SAX_shapelets_TS = SAX_shapelets_TS[np.where(rowsToDelete==1)] 
    #SAX_shapelets_TS = SAX_shapelets_TS[np.concatenate(np.argwhere(rowsToDelete==1))]

    '''
    rowsToDelete = np.zeros((sNum), dtype=np.int32)

    for i in range(sNum):
        if (len(np.argwhere((SAX_shapelets_TS[i, 3:] > ub) | (SAX_shapelets_TS[i, 3:] < lb))) > hashesTotal*0.5):
            rowsToDelete[i] = 1

    print(SAX_shapelets_TS.shape)
    SAX_shapelets_TS = np.delete(SAX_shapelets_TS, rowsToDelete, axis=0) 
    print(SAX_shapelets_TS.shape)
    '''

    stds = np.std(SAX_shapelets_TS[:, 3:], ddof=1, axis=1)
    uShapeletsOrder = np.argsort(stds)
    stds = np.sort(stds)

    if len(stds) > 0:
        medianStd = stds[round(len(stds) / 2)]
        smallStds = stds[stds <= medianStd]
        uShapeletsOrder[stds <= medianStd] = uShapeletsOrder[np.random.permutation(len(smallStds))]
        SAX_shapelets_TS = SAX_shapelets_TS[uShapeletsOrder, :]

    otherInd = np.argwhere(rowsToDelete==0)
    #otherInd = np.argwhere(rowsToDelete) 
    otherInd = otherInd[np.random.permutation(len(otherInd))]
    otherInd = np.concatenate(otherInd)
    SAX_shapelets_TS = np.concatenate((SAX_shapelets_TS, SAX_shapelets_TS_backup[otherInd]), axis=0)

    return uShapeletsOrder, SAX_shapelets_TS, stds
