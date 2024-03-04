import os
import sys
import json
import argparse
import numpy as np
from models.models import ClusterModel
from metrics.metrics import ClusterMetrics
from datasets.ucr_uea import ClusterDataLoader
from distances.distances import DistanceMatrix

sys.path.append('.')

parser = argparse.ArgumentParser(description='Clustering')
parser.add_argument('--dataset', type=str, default='ucr_uea', help='dataset name')
parser.add_argument('--start', type=int, default=1, help='start number of dataset')
parser.add_argument('--end', type=int, default=128, help='end number of dataset')
parser.add_argument('--data_path', type=str, default='./data/UCR2018/', help='path to the dataset')
parser.add_argument('--results_path', type=str, default=None, help='path to the results')
parser.add_argument('--model', type=str, default='agglomerative', help='name of the model')
parser.add_argument('--linkage', type=str, default=None, help='linkage in hierarchical clustering')
parser.add_argument('--ar_coeff_transforms', type=str, default='lpcc', help='ar coefficient transform')
parser.add_argument('--gamma', type=int, default=None, help='gamma value for SINK distance')
parser.add_argument('--threshold_metric', type=str, default='kneepoint', help='threshold metric for DensityPeaks')
parser.add_argument('--distance', type=str, default=None, help='distance measure')
parser.add_argument('--precomputed', type=str, default='False', help='use precomputed distances')
parser.add_argument('--param_file', type=str, default=None, help='path to extra parameter file')
args = parser.parse_args()


def main():
    dataloader = ClusterDataLoader(args.dataset, args.data_path)
    results = {'clustering': []}

    if args.param_file:
        params = np.load(args.param_file, allow_pickle=True)

    for i, sub_dataset_name in enumerate(sorted(os.listdir(args.data_path), key=str.lower)[args.start-1:args.end]):
        ts, labels, nclusters = dataloader.load(sub_dataset_name)
        print(sub_dataset_name)
        dm = DistanceMatrix(args.gamma)

        if args.param_file:
            param = params[i]
        else:
            param = None

        if args.precomputed == 'False':
            model = ClusterModel(ts, labels, dm, nclusters, args.linkage, args.threshold_metric,
                                 param, args.ar_coeff_transforms, sub_dataset_name=sub_dataset_name)
            predictions, dist_timing, cluster_timing = getattr(model, args.model)(args.distance)
        else:
            precomputed_dist_path = os.path.join('../../TSCluster/distances/stored_distances', sub_dataset_name)
            model = ClusterModel(ts, labels, dm, nclusters, args.linkage, args.threshold_metric, param,
                                 args.ar_coeff_transforms, args.precomputed, precomputed_dist_path, sub_dataset_name=sub_dataset_name)
            predictions, dist_timing, cluster_timing = getattr(model, args.model)(args.distance)

        metrics = ClusterMetrics(labels, predictions)
        metrics_dict = {'rand_score': None, 'adjusted_rand_score': None, 'normalized_mutual_information': None}
        for metric in metrics_dict.keys():
            metrics_dict[metric] = getattr(metrics, metric)()

        metrics_dict['sub_dataset_name'] = sub_dataset_name
        metrics_dict['distance_timing'] = dist_timing
        metrics_dict['cluster_timing'] = cluster_timing

        if args.model == "DEC" or args.model == "IDEC" or args.model == "DCN" or args.model == "DTCR":
            metrics_dict['estimator'] = model.estimator
            metrics_dict["inertia"] = model.inertia            

        print('Metrics:')
        print(metrics_dict)
        results['clustering'].append(metrics_dict)

        with open(args.results_path, 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    main()
