import math
import numpy as np


class DensityPeakCluster(object):
    """
    Density Peak Cluster.

    This class implements the Density Peak clustering algorithm.

    Methods:
        fit: Fit the model to the data.
        plot: Plot the clustering.

    Attributes:
        n_id: The number of data rows.
        distance: The distance between each pair of data points.
        dc: The threshold of density cut off.
        rho: The density of each data point.
        nneigh: The minimum upper density nearest neighbor for each data point.
        delta: The minimum upper density nearest neighbor distance for each data point.
        labels_: The cluster labels.
        center: The center data point.
    """
    def __init__(self,
                 dc=None,
                 silence=True,
                 gauss_cutoff=True,
                 threshold_metric='kneepoint',
                 density_threshold=None,
                 distance_threshold=None,
                 anormal=True):
        """
        Initialize the parameters for Density Peak clustering.

        Parameters:
        :param dc: The local density threshold. Use None for default select, 'auto' for auto select.
        :param silence: Whether to print process log.
        :param gauss_cutoff: The neighbor cutoff function. True for Gaussian, False for hard cutoff.
        :param threshold_metric: The metric used to determine the density and distance thresholds.
        :param density_threshold: The density threshold to filter center. Use None for auto.
        :param distance_threshold: The distance threshold to filter center. Use None for auto.
        :param anormal: Whether to assign anormal points to -1.
        """

        self.dc = dc
        self.silence = silence
        self.gauss_cutoff = gauss_cutoff
        self.threshold_metric = threshold_metric
        self.density_threshold = density_threshold
        self.distance_threshold = distance_threshold
        self.anormal = anormal
        self.labels_ = None

    def build_distance(self):
        """
        Calculate the distance between each pair of data points.

        Returns:
        distance: The distance dictionary.
        max_dis: The maximum distance.
        min_dis: The minimum distance.
        """
        # Calculate distance matrix
        distance_matrix = self.data
        triangle_upper = np.triu_indices(self.data.shape[0], 1)
        triangle_upper = distance_matrix[triangle_upper]

        distance = {}
        for i in range(self.n_id):
            for j in range(i + 1, self.n_id):
                distance[(i, j)] = distance_matrix[i, j]
                distance[(j, i)] = distance_matrix[i, j]

        max_dis, min_dis = np.max(triangle_upper), np.min(triangle_upper)
        return distance, max_dis, min_dis

    def auto_select_dc(self):
        """
        Auto select the local density threshold that ensures the average neighbor is 1-2 percent of all nodes.

        Returns:
        dc: The local density threshold.
        """
        max_dis, min_dis = self.max_dis, self.min_dis
        dc = (max_dis + min_dis) / 2

        while True:
            nneighs = sum([1 for v in self.distances.values() if v < dc]) / self.n_id ** 2
            if 0.01 <= nneighs <= 0.002:
                break
            # Binary search
            if nneighs < 0.01:
                min_dis = dc
            else:
                max_dis = dc
            dc = (max_dis + min_dis) / 2
            if max_dis - min_dis < 0.0001:
                break
        return dc

    def select_dc(self):
        """
        Select the local density threshold.

        Returns:
        dc: The local density threshold.
        """
        if self.dc == 'auto':
            dc = self.auto_select_dc()
        else:
            percent = 2.0
            position = int(self.n_id * (self.n_id + 1) / 2 * percent / 100)
            dc = np.sort(list(self.distances.values()))[position * 2 + self.n_id]

        return dc

    def local_density(self):
        """
        Compute the local density for all data points.

        Returns:
        rho: The local density vector.
        """
        guass_func = lambda dij, dc: math.exp(- (dij / dc) ** 2)
        cutoff_func = lambda dij, dc: 1 if dij < dc else 0
        func = guass_func if self.gauss_cutoff else cutoff_func
        rho = [0] * self.n_id
        for i in range(self.n_id):
            for j in range(i + 1, self.n_id):
                temp = func(self.distances[(i, j)], self.dc)
                rho[i] += temp
                rho[j] += temp
        return np.array(rho, np.float32)

    def min_neighbor_and_distance(self):
        """
        Compute the minimum distance to the higher local density point (which is the nearest neighbor) for all data points.

        Returns:
        delta: The distance vector.
        nneigh: The nearest neighbor vector.
        """
        sort_rho_idx = np.argsort(-self.rho)
        delta, nneigh = [float(self.max_dis)] * (self.n_id), [0] * self.n_id
        delta[sort_rho_idx[0]] = -1.
        for i in range(self.n_id):
            for j in range(0, i):
                old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]
                if self.distances[(old_i, old_j)] < delta[old_i]:
                    delta[old_i] = self.distances[(old_i, old_j)]
                    nneigh[old_i] = old_j
        delta[sort_rho_idx[0]] = max(delta)
        return np.array(delta, np.float32), np.array(nneigh, np.float32)

    def collapse(self):
        """
        Perform clustering.

        Returns:
        labels_: The cluster labels.
        center: The center data point.
        """
        cluster, center = {}, {}  # cl/icl in cluster_dp.m

        if self.threshold_metric == 'median':
            if self.density_threshold is None:
                self.density_threshold = np.median(self.rho)
            if self.distance_threshold is None:
                self.distance_threshold = np.median(self.delta)
        
        elif self.threshold_metric == 'kneepoint':
            from kneed import KneeLocator
            try:
                self.density_threshold = KneeLocator(np.sort(self.rho)[np.sort(self.rho)>0], range(1, len(np.sort(self.rho)[np.sort(self.rho)>0])+1), curve='convex', direction='decreasing').knee
                if self.density_threshold == None:
                    self.density_threshold = np.median(self.rho)
            except:
                self.density_threshold = np.median(self.rho)
                
            try:
                self.distance_threshold = KneeLocator(np.sort(self.delta)[np.sort(self.delta)>=0], range(1, len(np.sort(self.delta)[np.sort(self.delta)>=0])+1), curve='convex', direction='decreasing').knee
                if self.distance_threshold == None:
                    self.distance_threshold = np.median(self.delta)  
            except:
                self.density_threshold = np.median(self.delta) 

        # Initialize center and other points
        for idx, (ldensity, mdistance, nneigh_item) in enumerate(zip(self.rho, self.delta, self.nneigh)):
            if ldensity >= self.density_threshold and mdistance >= self.distance_threshold:
                center[idx] = idx
                cluster[idx] = idx
            else:
                cluster[idx] = -1

        # Assignation
        ordrho = np.argsort(-self.rho)
        for i in range(ordrho.shape[0]):
            if cluster[ordrho[i]] == -1:
                cluster[ordrho[i]] = cluster[self.nneigh[ordrho[i]]]

        # Halo
        halo, bord_rho = {}, {}
        for i in range(ordrho.shape[0]):
            halo[i] = cluster[i]
        if len(center) > 0:
            for idx in center.keys():
                bord_rho[idx] = 0.0
            for i in range(self.rho.shape[0]):
                for j in range(i + 1, self.rho.shape[0]):
                    if cluster[i] != cluster[j] and self.distances[i, j] <= self.dc:
                        rho_aver = (self.rho[i] + self.rho[j]) / 2.0
                        if rho_aver > bord_rho[cluster[i]]:
                            bord_rho[cluster[i]] = rho_aver
                        if rho_aver > bord_rho[cluster[j]]:
                            bord_rho[cluster[j]] = rho_aver
            for i in range(self.rho.shape[0]):
                if self.rho[i] < bord_rho[cluster[i]]:
                    halo[i] = 0

        if self.anormal:
            for i in range(self.rho.shape[0]):
                if halo[i] == 0:
                    cluster[i] = - 1

        labels_ = np.zeros(self.n_id).astype(int)
        for k in cluster:
            labels_[k] = cluster[k]

        return labels_, list(center.values())

    def fit(self, data):
        """
        Fit the model to the data.

        Parameters:
        data: The data for clustering.

        Returns:
        None
        """
        if isinstance(data, np.ndarray): data = np.array(data)

        self.data = data
        self.n_id = data.shape[0]

        # Calculate distance
        self.distances, self.max_dis, self.min_dis = self.build_distance()

        # Select dc
        self.dc = self.select_dc()
        # calculate local density
        self.rho = self.local_density()

        # calculate nearest neighbor and delta
        self.delta, self.nneigh = self.min_neighbor_and_distance()

        # clustering
        self.labels_, self.center = self.collapse()

        return self.labels_

    def plot(self, mode="all", title="", **kwargs):
        """
        Plot cluster.
        :param mode: "decision" to plot decision graph,
            "label" to plot label graph, "all" to plot decision and label graph.
        :param title: title to plot
        :return: None
        """
        import matplotlib.pyplot as plt
        from dpca.plot_utils import plot_scatter

        if mode not in {"decision", "label", "all"}:
            raise ValueError("mode must be one of [decision, label, all].")

        if mode in ["decision", "all"]:
            title_ = "decision graph" if title == "" else title + " decision graph"
            plot_scatter(np.vstack((self.rho, self.delta)).T,
                         self.labels_, x_label='density', y_label='distance', title=title_, **kwargs)

        if mode in ["label", "all"]:
            title_ = "label graph" if title == "" else title + " label graph"
            plot_scatter(self.data, self.labels_, title=title_, **kwargs)

        plt.show()
