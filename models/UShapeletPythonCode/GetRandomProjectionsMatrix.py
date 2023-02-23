from numpy import concatenate
from .GetSaxHash import *
from collections import OrderedDict


def ismember_rows(a, b):
    return np.nonzero(np.all(b == a[:,np.newaxis], axis=2))[1]


def GetRandomProjectionsMatrix(data, sLen, sNum):
    dimensionality = 16
    cardinality = 4

    shapeletsHash, shapeletsStorage = GetSaxHash(data, cardinality, sLen, dimensionality)

    R = 10
    maskedNum = 3
    masks = np.zeros((R, maskedNum))
    A = data[:, 0]
    x = list(set(A))
    tsNumber = len(x)
    tsLengths = np.zeros((tsNumber, 1))

    for k in range(tsNumber):
        tsLengths[k, 0] = np.sum(A==x[k])

    shapeletsHash = OrderedDict(sorted(shapeletsHash.items()))
    shapeletsStorage = OrderedDict(sorted(shapeletsStorage.items()))

    shapelets = shapeletsHash.keys()
    shapelets = list(shapelets)
    shValues = shapeletsHash.values()
    shStorageValues = shapeletsStorage.values()
    shStorageVals = [len(x) for x in shStorageValues]
    shHashNum = [len(x) for x in shValues]

    shCount = len(shapelets)
    tsPerS = np.ones((sNum, 1))
    vals = np.zeros((sNum, 1))
    
    firstInd = 0
    for i in range(shCount):
        vals[firstInd:firstInd+shStorageVals[i]] = shHashNum[i]
        firstInd = firstInd + shStorageVals[i]

    shValues = np.concatenate(np.array(list(shValues)), axis=0).astype(np.int32)
    shStorageValues = np.concatenate(np.array(list(shStorageValues)), axis=0).astype(np.int32)
    tsPerS[shStorageValues] = vals
    tsPerS = np.tile(tsPerS, (1, R))

    for i in range(R):
        maskedPlaces = np.sort(np.random.randint(16, size=(1, maskedNum))) + 1

        while (not (np.all(np.argwhere(ismember_rows(masks, maskedPlaces)))==1)) or (np.argwhere(np.diff(maskedPlaces)).shape[0] < maskedNum -1):
            maskedPlaces = np.sort(np.random.randint(16, size=(1, maskedNum))) + 1

        masks[i, :] = maskedPlaces
        newKeys = shapelets
        for j in range(maskedNum):
            wCard = maskedPlaces[0, j]
            newKeys = ((np.array(newKeys) // (10**wCard))*(10**wCard)) + (np.array(newKeys) % 10**(wCard-1))

        x = list(set(newKeys))
        N = len(x)
        
        count = np.zeros((N,1))
        for k in range(N):
          count[k, 0] = np.sum(newKeys==x[k])
        
        repInd = np.argwhere(count>1)
        repIndLen = repInd.shape[0]

        for k in range(repIndLen):
            idxs = (newKeys == x[repInd[k][0]]).astype(int)
            allTS = shValues[np.argwhere(idxs==1)]
            allShPositions = shStorageValues[np.argwhere(idxs==1)]
            tss = np.concatenate(allTS)
            shPositions = list(allShPositions)
            uniqTSnum = len(list(set(tss)))
            tsPerS[shPositions, i] = uniqTSnum
    
    return tsPerS
