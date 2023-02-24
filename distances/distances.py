import math
import time
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist


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
        xlen = len(x)
        ylen = len(y)
        xp = np.zeros(xlen+1)
        yp = np.zeros(ylen+1)
        for i in range(1, xlen+1):
            xp[i] = x[i-1]
        for i in range(1, ylen+1):
            yp[i] = y[i-1]
        xlen = xlen + 1
        ylen = ylen + 1
        x = xp
        y = yp
        length = max(xlen, ylen)
        dp = np.zeros((length, length))
        dp1 = np.zeros((length, length))
        dp2 = np.zeros(length)
        dp2[0] = 1
        for i in range(1, min(xlen, ylen)):
            dp2[i] = self.Dlpr(x[i], y[i], gamma)
        dp[0][0] = 1
        dp1[0][0] = 1
        for i in range(1, xlen):
            dp[i][0] = dp[i - 1][0] * self.Dlpr(x[i], y[1], gamma)
            dp1[i][0] = dp1[i - 1][0] * dp2[i]
        for i in range(1, ylen):
            dp[0][i] = dp[0][i - 1] * self.Dlpr(x[1], y[i], gamma)
            dp1[0][i] = dp1[0][i - 1] * dp2[i]
        for i in range(1, xlen):
            for j in range(1, ylen):
                lcost = self.Dlpr(x[i], y[j], gamma)
                dp[i][j] = (dp[i - 1][j] + dp[i][j - 1] + dp[i - 1][j - 1]) * lcost
                if i ==j:
                    dp1[i][j] = dp1[i - 1][j - 1] * lcost + dp1[i - 1][j] * dp2[i] + dp1[i][j - 1] * dp2[j]
                else:
                    dp1[i][j] = dp1[i - 1][j] * dp2[i] + dp1[i][j - 1] * dp2[j]

        for i in range(0, xlen):
            for j in range(0, ylen):
                dp[i][j] += dp1[i][j]
    
        ans = dp[xlen - 1][ylen - 1]

        return ans

            
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
                dist_mat[i, j] = self.kdtw_norm(ts[i], ts[j], gamma=0.125)
                dist_mat[j, i] = dist_mat[i, j]
        return dist_mat


    def gak_dist(self, x, y, gamma): 
        x, y = np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
        K = np.exp(-(cdist(x, y, "sqeuclidean") / (2 * gamma ** 2) + np.log(2 - np.exp(cdist(x, y, "sqeuclidean") / (2 * gamma ** 2)))))

        csum = np.zeros((len(x)+1, len(y)+1))
        csum[0][0] = 1
        for i in range(len(x)):
            for j in range(len(y)):
                csum[i+1][j+1] = (csum[i, j + 1] + csum[i + 1, j] + csum[i, j]) * K[i][j]

        return csum[len(x)][len(y)]


    def gak(self, ts):
        dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
        for i in range(ts.shape[0]):
            for j in range(i+1, ts.shape[0]):
                dist_mat[i, j] = self.gak_dist(ts[i], ts[j], gamma=0.1)
                dist_mat[j, i] = dist_mat[i, j]
        return dist_mat


    def swale_dist(self, x, y, epsilon):
        cur = np.zeros(len(y))
        prev = np.zeros(len(y))
        for i in range(len(x)):
            prev = cur
            cur = np.zeros(len(y))
            minw = 0
            maxw = len(y)-1
            for j in range(int(minw),int(maxw)+1):
                if i + j == 0:
                    cur[j] = 0
                elif i == 0:
                    cur[j] = j * 5
                elif j == minw:
                    cur[j] = i * 5
                else:
                    if (abs(x[i] - y[i]) <= epsilon):
                        cur[j] = prev[j-1] + 1
                    else:
                        cur[j] = min(prev[j], cur[j-1]) + 5
        return cur[len(y)-1]


    def swale(self, ts):
        dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
        for i in range(ts.shape[0]):
            for j in range(i+1, ts.shape[0]):
                dist_mat[i, j] = self.swale_dist(ts[i], ts[j], epsilon=0.2)
                dist_mat[j, i] = dist_mat[i, j]
        return dist_mat
   
    
    def erp_dist(self, x, y):
        lenx = len(x)
        leny = len(y)

        acc_cost_mat = np.full((lenx, leny), np.inf)

        for i in range(lenx):
            m = 0
            minw = 0
            maxw = leny-1

            for j in range(minw, maxw+1):
                if i + j == 0:
                    acc_cost_mat[i, j] = 0
                elif i == 0:
                    acc_cost_mat[i, j] = acc_cost_mat[i, j-1] + (y[j]-m)**2
                elif j == 0:
                    acc_cost_mat[i, j] = acc_cost_mat[i-1, j] + (x[i]-m)**2
                else:
                    acc_cost_mat[i, j] = min(acc_cost_mat[i-1, j-1] + (x[i] - y[j])**2,
                                         acc_cost_mat[i, j-1] + (y[j] - m)**2,
                                         acc_cost_mat[i-1, j] + (x[i]-m)**2)

        return math.sqrt(acc_cost_mat[lenx-1, leny-1])


    def erp(self, ts):
        dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
        for i in range(ts.shape[0]):
            for j in range(i+1, ts.shape[0]):
                dist_mat[i, j] = self.erp_dist(ts[i], ts[j])
                dist_mat[j, i] = dist_mat[i, j]
        return dist_mat


    def lcss_dist(self, x, y, w):
        lenx = len(x)
        leny = len(y)
        epsilon = 0.2
        if w == None:
            w = max(lenx, leny)
        D = np.zeros((lenx, leny))
        for i in range(lenx):
            wmin = max(0, i-w)
            wmax = min(leny-2, i+w)
            for j in range(wmin, wmax+1):
                if i + j == 0:
                    if abs(x[i]-y[j]) <= epsilon:
                        D[i][j] = 1
                    else:
                        D[i][j] = 0
                elif i == 0:
                    if abs(x[i]-y[j]) <= epsilon:
                        D[i][j] = 1
                    else:
                        D[i][j] =  D[i][j-1]
                elif j ==0:
                    if abs(x[i]-y[j]) <= epsilon:
                        D[i][j] = 1
                    else:
                        D[i][j] =  D[i-1][j]
                else:
                    if abs(x[i]-y[j]) <= epsilon:
                        D[i][j] = max(D[i-1][j-1]+1,
                                      D[i-1][j],
                                      D[i][j+1])
                    else:
                        D[i][j] = max(D[i-1][j-1],
                                      D[i-1][j],
                                      D[i][j+1])
        result = D[lenx-1, leny -1]
        return 1 - result/min(len(x),len(y))


    def lcss(self, ts):
        dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
        for i in range(ts.shape[0]):
            for j in range(i+1, ts.shape[0]):
                dist_mat[i, j] = self.lcss_dist(ts[i], ts[j], w=int(len(ts[i])/20))
                print(dist_mat[i, j])
                dist_mat[j, i] = dist_mat[i, j]
        return dist_mat


    def edr_dist(self, x, y):
        cur = np.full((1, len(y)), -np.inf)
        prev = np.full((1, len(y)), -np.inf)

        for i in range(len(x)):
            m = 0.1
            minw = 0
            maxw = len(y)-1
            prev = cur
            cur = np.full((len(y)), -np.inf)

            for j in range(int(minw), int(maxw)+1):
                if i + j == 0:
                    cur[j] = 0
                elif i == 0:
                    cur[j] = -j
                elif j == 0:
                    cur[j] = -i
                else:
                    if abs(x[i] - y[j]) <= m:
                        s1 = 0
                    else:
                        s1 = -1
                    cur[j] = max(prev[j - 1] + s1, prev[j] - 1, cur[j - 1] - 1)

        return 0 - cur[len(y) - 1]


    def edr(self, ts):
        dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
        for i in range(ts.shape[0]):
            for j in range(i+1, ts.shape[0]):
                dist_mat[i, j] = self.edr_dist(ts[i], ts[j])
                dist_mat[j, i] = dist_mat[i, j]
        return dist_mat


    def dtw_dist(self, x, y, w):
        N = len(x)
        M = len(y)
        if w == None:
            w = max(N, M)

        D = np.full((N+1, M+1), np.inf)
        D[0, 0] = 0
        
        for i in range(1, N+1):
            for j in range(max(1, i-w), min(i+w, M)+1):
                cost = (x[i-1] - y[j-1])**2
                D[i, j] = cost + min(D[i-1,j],D[i-1,j-1],D[i,j-1])

        Dist = math.sqrt(D[N, M])

        return Dist


    def dtw(self, ts):
        dist_mat = np.zeros((ts.shape[0], ts.shape[0]))
        for i in range(ts.shape[0]):
            for j in range(i+1, ts.shape[0]):
                dist_mat[i, j] = self.dtw_dist(ts[i], ts[j], w=int(len(ts[i])/10))
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
