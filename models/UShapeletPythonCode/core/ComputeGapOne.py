import numpy as np


def ComputeGapOne(dis, d):
    ma, mb, sa, sb = 0, 0, 0, 0
    Da = np.argwhere(dis <= d)
    Db = np.argwhere(dis > d)

    gap = 0
    r = 0
    if ((Da.shape[0]<2) | (Db.shape[0]<2)):
       return 0, None, None, None, None, 0

    r = len(np.argwhere(Da))/len(np.argwhere(Db))
    
    if ((0.2 < r) & (r < 5)):
        ma = np.mean(dis[Da])
        mb = np.mean(dis[Db])
        sa = np.std(dis[Da])
        sb = np.std(dis[Db])
        gap = mb - sb - (ma + sa)

    return gap, ma, mb, sa, sb, r
