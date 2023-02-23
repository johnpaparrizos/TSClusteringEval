import numpy as np

def findNN(x,y):
    x = np.concatenate(([0], x))
    n = len(x)

    y = (y-np.mean(y))/np.std(y)
    m = len(y)

    x = np.concatenate((x, [0 for _ in range(n, 2*n)]))
    y = y[::-1]                             
    y = np.concatenate((y, [0 for _ in range(m, 2*n)]))

    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    Z = X*Y
    z = np.fft.ifft(Z)

    sumy = np.sum(y)
    sumy2 = np.sum(y**2)
    
    cum_sumx = np.cumsum(x)
    cum_sumx2 =  np.cumsum(x**2)
    sumx2 = cum_sumx2[m:n+1]-cum_sumx2[0:n-m+1]
    sumx = cum_sumx[m:n+1]-cum_sumx[0:n-m+1]
    meanx = sumx/m
    sigmax2 = (sumx2/m)-(meanx**2)
    sigmax = np.sqrt(sigmax2)

    dist = (sumx2 - 2*sumx*meanx + m*(meanx**2))/sigmax2 - 2*(z[m:n+1] - sumy*meanx)/sigmax + sumy2
    dist = np.sqrt(dist)
    
    loc = np.argmin(dist)
    bsf = np.min(dist)
    bsf = bsf/np.sqrt(m)

    return loc, np.real(bsf)
