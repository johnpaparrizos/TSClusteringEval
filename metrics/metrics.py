from sklearn import metrics


class ClusterMetrics:
    def __init__(self, gt, pred):
        self.gt = gt
        self.pred = pred
        
    def rand_score(self):
        return metrics.rand_score(self.gt, self.pred)

    def adjusted_rand_score(self):
        return metrics.adjusted_rand_score(self.gt, self.pred)

    def normalized_mutual_information(self):
        return metrics.normalized_mutual_info_score(self.gt, self.pred)
