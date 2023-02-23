import os
import csv
import numpy as np


class ClusterDataLoader:
    def __init__(self, dataset_name, dataset_path):
        self.name = dataset_name
        self.path = dataset_path

    def load(self, sub_dataset_name):
        ts, labels = [], []
        for mode in ['_TRAIN', '_TEST']:
            with open(os.path.join(self.path, sub_dataset_name, sub_dataset_name + mode)) as csv_file:
                lines = csv.reader(csv_file, delimiter=',')
                for line in lines:
                    ts.append([float(x) for x in line[1:]])
                    labels.append(int(line[0])-1)
                
        if min(labels) == 1:
            labels = labels - 1
        if min(labels) == -1:
            labels = labels + 1

        return np.array(ts), np.array(labels), int(len(set(labels)))
