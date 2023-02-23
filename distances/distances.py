import math
import time
import numpy as np
from sklearn.metrics import pairwise_distances


class DistanceMatrix:
    def __init__(self, gamma):
        self.gamma = gamma

    def euclidean(self, ts):
        self.dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
        for i in range(ts.shape[0]):
            for j in range(i+1, ts.shape[0]):
                self.dist_mat[i, j] = self.calculate_euclidean_distance(ts[i], ts[j])
                self.dist_mat[j, i] = self.dist_mat[i, j]
        return self.dist_mat

    def calculate_euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x-y)**2))

    
    def msm_dist(self, new, x, y, c):
        if ((x <= new) and (new <= y)) or ((y <= new) and (new <= x)):
            dist = c
        else:
            dist = c + min(abs(new - x), abs(new - y))
        return dist


    def msm_scb(self, x, y, c=0.5, w=10):
        xlen = len(x)
        ylen = len(y)
        cost = np.full((xlen, ylen), np.inf)

        cost[0][0] = abs(x[0] - y[0])

        for i in range(1,len(x)):
            cost[i][0] = cost[i-1][0] + self.msm_dist(x[i], x[i-1], y[0], c)

        for i in range(1,len(y)):
            cost[0][i] = cost[0][i-1] + self.msm_dist(y[i], x[0], y[i-1], c)

        for i in range(1,xlen):
            for j in range(max(0, int(i-w)), min(ylen, int(i+w))):
                cost[i][j] = min(cost[i-1][j-1] + abs(x[i] - y[j]),
                                cost[i-1][j] + self.msm_dist(x[i], x[i -1], y[j], c),
                                cost[i][j-1] + self.msm_dist(y[j], x[i], y[j-1], c))

        return cost[xlen-1][ylen-1]


    def msm(self, ts):
        self.dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
        for i in range(ts.shape[0]):
            for j in range(i+1, ts.shape[0]):
                self.dist_mat[i, j] = self.msm_scb(ts[i], ts[j], c=0.5, w=int(ts.shape[1]/5))
                self.dist_mat[j, i] = self.dist_mat[i, j]
        return self.dist_mat


    def twed_scb(self, x, timesx, y, timesy, lamb=1, nu=0.0001, w=10):
        xlen = len(x)
        ylen = len(y)
        cur = np.full(ylen, np.inf)
        prev = np.full(ylen, np.inf)

        for i in range(0, xlen):
            prev = cur
            cur = np.full(ylen, np.inf)
            minw = max(0, i - w)
            maxw = min(ylen-1, i + w)
            for j in range(minw, maxw+1):
                if i + j == 0:
                    cur[j] = (x[i] - y[j]) **2
                elif i == 0:
                    c1 = (
                        cur[j - 1]
                        + (y[j - 1] - y[j]) **2
                        + nu * (timesy[j] - timesy[j - 1])
                        + lamb
                    )
                    cur[j] = c1
                elif j == 0:
                    c1 = (
                        prev[j]
                        + (x[i - 1] - x[i]) **2
                        + nu * (timesx[i] - timesx[i - 1])
                        + lamb
                    )
                    cur[j] = c1
                else:
                    c1 = (
                        prev[j]
                        +(x[i - 1] - x[i]) **2
                        + nu * (timesx[i] - timesx[i - 1])
                        + lamb
                    )
                    c2 = (
                        cur[j - 1]
                        + (y[j - 1] - y[j])**2
                        + nu * (timesy[j] - timesy[j - 1])
                        + lamb
                    )
                    c3 = (
                        prev[j - 1]
                        + (x[i] - y[j]) ** 2
                        + (x[i - 1]- y[j - 1]) ** 2
                        + nu
                        * (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1]))
                    )
                    cur[j] = min(c1, c2, c3)

        return cur[ylen - 1]


    def twed(self, ts):
        self.dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
        for i in range(ts.shape[0]):
            for j in range(i+1, ts.shape[0]):
                self.dist_mat[i, j] = self.twed_scb(ts[i], np.array(list(range(ts[i].shape[0]))), ts[j], np.array(list(range(ts[j].shape[0]))), w=int(ts.shape[1]))
                self.dist_mat[j, i] = self.dist_mat[i, j]
        return self.dist_mat


    def kdtw_sim(self, x, y, gamma):
        x_copy = np.zeros((x.shape[0]+1))
        y_copy = np.zeros((y.shape[0]+1))

        x_copy[1:] = x
        y_copy[1:] = y

        x_len, y_len = x_copy.shape[0], y_copy.shape[0]

        DP = np.zeros((x_len, y_len))
        DP1 = np.zeros((x_len, y_len))
        DP2 = np.zeros((max(x_len, y_len)))

        l = min(x_len, y_len)
        DP2[0] = 1
        for i in range(1, l):
            DP2[i] = self.Dlpr(x_copy, y_copy, gamma)

        DP[0, 0] = 1
        DP1[0, 0] = 1
        n, m = len(x_copy), len(y_copy)

        for i in range(1, l):
            DP[i,0] = DP[i-1,1]*self.Dlpr(x_copy[i], y_copy[1], gamma)
            DP1[i,0] = DP1[i-1,1]*DP2[i]

        for j in range(1, l):
            DP[0,j] = DP[1,j-1]*self.Dlpr(x_copy[1], y_copy[j], gamma)
            DP1[0,j] = DP1[1,j-1]*DP2[j]

        for i in range(1, n):
            for j in range(1, m):
                lcost=self.Dlpr(x_copy[i], y_copy[j], gamma)
                DP[i,j]=(DP[i-1,j] + DP[i,j-1] + DP[i-1,j-1])*lcost
                if (i == j):
                    DP1[i,j] = DP1[i-1,j-1]*lcost + DP1[i-1,j]*DP2[i] + DP1[i,j-1]*DP2[j]
                else:
                    DP1[i,j] = DP1[i-1,j]*DP2[i] + DP1[i,j-1]*DP2[j]
        DP=DP+DP1
        similarity = DP[n-1,m-1]

        return similarity


    def Dlpr(self, x, y, gamma):
        factor=1/3
        minprob=1e-20
        cost = factor*(np.exp(-gamma*np.sum((x - y)**2))+minprob)
        return cost


    def kdtw_norm(self, x, y, gamma):
        sim = self.kdtw_sim(x, y, gamma)/np.sqrt(self.kdtw_sim(x, x, gamma) * self.kdtw_sim(y, y, gamma))
        return sim


    def kdtw(self, ts):
        dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
        for i in range(ts.shape[0]):
            for j in range(i+1, ts.shape[0]):
                dist_mat[i, j] = self.kdtw_norm(ts[i], ts[j], gamma=0.1)
                dist_mat[j, i] = dist_mat[i, j]
        return dist_mat


    def lorentzian(self, ts):
        self.dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
        for i in range(ts.shape[0]):
            for j in range(i+1, ts.shape[0]):
                self.dist_mat[i, j] = self.calculate_lorentzian_distance(ts[i], ts[j])
                self.dist_mat[j, i] = self.dist_mat[i, j]
        return self.dist_mat

    def calculate_lorentzian_distance(self, x, y):
        return np.sum(np.log(1 + np.absolute(x-y)))

    def SBD(self, ts):
        self.dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
        for i in range(ts.shape[0]):
            for j in range(i+1, ts.shape[0]):
                self.dist_mat[i, j] = self.calculate_SBD_distance(ts[i], ts[j])
                self.dist_mat[j, i] = self.dist_mat[i, j]
        return self.dist_mat

    def _next_pow_2(self, x):
        return 1<<(x-1).bit_length()

    def NCCc_pairwise(self, x, y):
        l = self._next_pow_2(2 * len(x) - 1)
        cc = np.fft.ifft(np.fft.fft(x, n=l) * np.conjugate(np.fft.fft(y, n=l)))
        return cc / (np.linalg.norm(x) * np.linalg.norm(y))

    def calculate_SBD_distance(self, x, y):
        cc = self.NCCc_pairwise(x, y)
        maxidx = np.argmax(cc)
        dist = 1 - cc[maxidx]
        return dist

    def sum_exp_NCCc(self, x, y, gamma):
        sim = sum(np.exp(gamma*self.NCCc_pairwise(x,y)))
        return sim

    def calculate_sink_distance(self, x, y, sum_exp_NCCc_xx, sum_exp_NCCc_yy, gamma):
        sim = self.sum_exp_NCCc(x, y, gamma) / (np.sqrt(sum_exp_NCCc_xx * sum_exp_NCCc_yy))
        return sim

    def store_sum_exp_NCCc(self, ts, gamma):
        sum_exp_NCCc_list = []
        for i in range(ts.shape[0]):
            sum_exp_NCCc_list.append(self.sum_exp_NCCc(ts[i], ts[i], gamma))
        return sum_exp_NCCc_list

    def sink(self, ts, gamma=None):
        if gamma:
            self.gamma = gamma
        
        sum_exp_NCCc_list = self.store_sum_exp_NCCc(ts, self.gamma)
        self.dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
        for i in range(ts.shape[0]):
            for j in range(i, ts.shape[0]):
                self.dist_mat[i, j] = self.calculate_sink_distance(ts[i], ts[j], sum_exp_NCCc_list[i], sum_exp_NCCc_list[j], self.gamma)
                self.dist_mat[j, i] = self.dist_mat[i, j]
        return self.dist_mat


if __name__ == '__main__':
    import os
    import sys
    import json
    from tqdm import tqdm
    sys.path.append('.')
    from datasets.ucr_uea import ClusterDataLoader

    time_dict = {'timings': []}
    dataloader = ClusterDataLoader('ucr_uea', './data/UCR2018/')
    for sub_dataset_name in tqdm(sorted(os.listdir('./data/UCR2018/'), key=str.lower)[0:128]):
        print(sub_dataset_name)
        ts, labels, nclusters = dataloader.load(sub_dataset_name)

        os.makedirs('./distances/stored_distances/'+sub_dataset_name, exist_ok=True)
        metrics_dict = {'sub_dataset_name': sub_dataset_name, 'euclidean': None, 'lorentzian': None, 'SBD': None, 'sink': []}
        dm = DistanceMatrix(None)

        '''
        # euclidean distance
        dist_time_start = time.time()
        dist_mat = dm.euclidean(ts)
        dist_timing = time.time() - dist_time_start
        metrics_dict['euclidean'] = dist_timing
        np.savetxt('./distances/stored_distances/'+sub_dataset_name+'/'+'euclidean.txt', dist_mat)
        print('Euclidean', dist_timing)

        # lorentzian distance
        dist_time_start = time.time()
        dist_mat = dm.lorentzian(ts)
        dist_timing = time.time() - dist_time_start
        metrics_dict['lorentzian'] = dist_timing
        np.savetxt('./distances/stored_distances/'+sub_dataset_name+'/'+'lorentzian.txt', dist_mat)
        print('Lorentzian', dist_timing)
        '''

        # SBD distance
        dist_time_start = time.time()
        dist_mat = dm.SBD(ts)
        dist_timing = time.time() - dist_time_start
        metrics_dict['SBD'] = dist_timing
        np.savetxt('./distances/stored_distances/'+sub_dataset_name+'/'+'sbd.txt', dist_mat)
        print('SBD', dist_timing)

        '''
        # SINK distance
        for gamma in range(19, 21):
            dm = DistanceMatrix(ts, gamma)
            dist_time_start = time.time()
            dist_mat = dm.sink(ts)
            dist_timing = time.time() - dist_time_start
            metrics_dict['sink'].append([gamma, dist_timing])
            np.savetxt('./distances/stored_distances/'+sub_dataset_name+'/'+'sink_'+str(gamma)+'.txt', dist_mat)
            print('SINK', dist_timing)
        '''

        time_dict['timings'].append(metrics_dict)

    with open('./distances/timings/timing_sbd.json', 'w') as f:
        json.dump(time_dict, f)
