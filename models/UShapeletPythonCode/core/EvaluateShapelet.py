import numpy as np
from sklearn import metrics


def EvaluateShapelet(distanceMatrix, dt, dataLabels):
    IDX = np.zeros((dataLabels.shape[0]))
    IDX[np.concatenate(distanceMatrix) <= dt[0]] = 1
    RI = metrics.rand_score(dataLabels, IDX)

    return RI, IDX