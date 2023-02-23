from .core.EvaluateShapelet import *
from .core.ComputeGapRaw import *

def GetActualGap(sLen, bestShapelets, i, data, classLabels, clsNum):
    clsLabels = classLabels
    ts = int(bestShapelets[i, 0])
    loc = int(bestShapelets[i, 1])

    actualShapelet = data[ts, loc:loc+sLen]
    actualShapelet = actualShapelet.reshape((1, actualShapelet.shape[0]))

    #np.save('./original.npy', data[ts])
    #np.save('./shapelet.npy', data[ts, loc:loc+sLen])

    if clsNum > 2:
        sClass = classLabels[ts]
        clsLabels[clsLabels!=sClass] = sClass - 1
    
    maxGapActual, dt, dis_raw1, _ = ComputeGapRaw(actualShapelet, data)
    ri, newIDX = EvaluateShapelet(dis_raw1, dt, clsLabels)

    return maxGapActual, ri, newIDX
