import numpy as np

def PartitioningAroundMedoids(nclusters, D):
    medoids = np.random.randint(D.shape[0], size=nclusters)
    n = D.shape[0]
    k = len(medoids)
    maxit = 100

    labels = np.argmin(D[medoids, :], axis=0)
    costs = np.min(D[medoids, :], axis=0)


    cost = np.sum(costs)
    last = 0
    it = 0
    while ((last!=medoids) & (it < maxit)).any():
        best_so_far_medoids = medoids
        for i in range(k):
            medoids_aux = medoids
            for j in range(n):
                medoids_aux[i] = j
                
                labels_aux = np.argmin(D[medoids_aux, :], axis=0)
                costs_aux = np.min(D[medoids, :], axis=0)

                cost_aux = np.sum(costs_aux)
                if cost_aux < cost:
                    best_so_far_medoids = medoids_aux
                    cost = cost_aux
                    labels = labels_aux

        last = medoids
        medoids = best_so_far_medoids
        it = it + 1

    return labels
