import numpy
import sys
import os
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from .utils import load_dataset, n_classes
from .mimic_models import MimicBetaInitModelL2, MimicBetaInitModelConvL2


def ldps(data, n_clusters):
    oar_job_id = "-1"
    model_class = MimicBetaInitModelConvL2
    nkiter = 9
    ratio_n_shapelets = 10
    n_ts, ts_len = data.shape[:2]

    shapelet_lengths = {}
    for sz in [int(p * ts_len) for p in [.15, .3, .45]]:
        n_shapelets = int(numpy.log(ts_len - sz) * ratio_n_shapelets)  # 2, 5, 8, 10
        shapelet_lengths[sz] = n_shapelets

    m = model_class(shapelet_lengths, d=data.shape[2], print_loss_every=1000, ada_grad=True, niter=1000,
                print_approx_loss=True)
    m.fit(data)
    for ikiter in range(nkiter):
        m.partial_fit(data, 1000, (ikiter + 1) * 1000)

    data_shtr = numpy.empty((data.shape[0], sum(shapelet_lengths.values())))
    for i in range(data.shape[0]):
        data_shtr[i] = m._shapelet_transform(data[i])

    km = KMeans(n_clusters=n_clusters)
    pred_labels = km.fit_predict(data_shtr)
    return pred_labels


