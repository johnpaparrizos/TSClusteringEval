from sklearn import metrics


class ClusterMetrics:
    """
    Class for computing clustering evaluation metrics.
    Args:
        gt (array-like): Ground truth labels.
        pred (array-like): Predicted labels.
    """
    def __init__(self, gt, pred):
        self.gt = gt
        self.pred = pred
        
    def rand_score(self):
        """
        Computes the Rand index score.

        Returns:
            float: The Rand index score.
        """
        return metrics.rand_score(self.gt, self.pred)

    def adjusted_rand_score(self):
        """
        Computes the adjusted Rand index score.

        Returns:
            float: The adjusted Rand index score.
        """
        return metrics.adjusted_rand_score(self.gt, self.pred)

    def normalized_mutual_information(self):
        """
        Computes the normalized mutual information score.

        Returns:
            float: The normalized mutual information score.
        """
        return metrics.normalized_mutual_info_score(self.gt, self.pred)
