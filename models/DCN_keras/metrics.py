import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


#def acc(y_true, y_pred):
#    """
#    Calculate clustering accuracy. Require scikit-learn installed
#
#    # Arguments
#        y: true labels, numpy.array with shape `(n_samples,)`
#        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#
#    # Return
#        accuracy, in [0,1]
#    """
#    y_true = y_true.astype(np.int64)
#    assert y_pred.size == y_true.size
#    D = max(y_pred.max(), y_true.max()) + 1
#    w = np.zeros((D, D), dtype=np.int64)
#    for i in range(y_pred.size):
#        w[y_pred[i], y_true[i]] += 1
#    ind = linear_sum_assignment(w.max() - w)
#    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def acc(y,ypred):
    """
    Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.
    
    ypred: 1-D numpy vector, predicted labels
    y: 1-D numpy vector, ground truth

    The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
    This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.
    
    """
    assert len(y) > 0
#    assert len(np.unique(ypred)) == len(np.unique(y))
    
    s = np.unique(ypred)
    t = np.unique(y)
    
    N = len(np.unique(ypred))
    C = np.zeros((N, N), dtype = np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(ypred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)
    
    # convert the C matrix to the 'true' cost
    Cmax = np.amax(C)
    C = Cmax - C
    # 
#    indices = linear_sum_assignment(C)
#    row = indices[:][:, 0]
#    col = indices[:][:, 1]
    row,col = linear_sum_assignment(C)
    # calculating the accuracy according to the optimal assignment
    count = 0
    for i in range(N):
        idx = np.logical_and(ypred == s[row[i]], y == t[col[i]] )
        count += np.count_nonzero(idx)
    
    return 1.0*count/len(y)
