import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def view_results(path):
    """
    View the results of a clustering experiment.

    Args:
        path (str): Path to the directory containing the results.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='View Results')
    parser.add_argument('--path', type=str, default=None, help='path to results')
    args = parser.parse_args()

    result_list = []
    names = []
    for file in os.listdir(args.path):
        with open(os.path.join(args.path, file)) as f:
            data = json.load(f)
        for sub_data in data['clustering']:
            result_list.append(sub_data)
            names.append(sub_data['sub_dataset_name'])

    res_lis = {'name': [], 'rand_score': [], 'adjusted_rand_score': [], 'normalized_mutual_information': [], 'distance_timing': [], 'cluster_timing': []}
    for name in tqdm(names):
        rand_list, adj_rand, nmi_list, dist_list, cluster_list = [], [], [], [], []
        for result in result_list:
            if result['sub_dataset_name'] == name:
                rand_list.append(result['rand_score'])
                adj_rand.append(result['adjusted_rand_score'])
                nmi_list.append(result['normalized_mutual_information'])
                dist_list.append(result['distance_timing'])
                cluster_list.append(result['cluster_timing'])

        res_lis['name'].append(name)
        res_lis['rand_score'].append(sum(rand_list) / len(rand_list))
        res_lis['adjusted_rand_score'].append(sum(adj_rand) / len(adj_rand))
        res_lis['normalized_mutual_information'].append(sum(nmi_list) / len(nmi_list))

        if dist_list[0] is None:
            res_lis['distance_timing'].append(None)
        else:
            res_lis['distance_timing'].append(sum(dist_list) / len(dist_list))

        res_lis['cluster_timing'].append(sum(cluster_list) / len(cluster_list))

    df = pd.DataFrame(data=res_lis)
    df.to_csv(os.path.join(args.path, 'experiment.csv'))
