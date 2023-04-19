import os.path as osp
import os

import numpy as np
import time
from torch_geometric.datasets import TUDataset

SCRIPT_DIR = osp.abspath(osp.join(osp.dirname(__file__)))
PREV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
# Return classes as a numpy array.
def read_classes(ds_name):
    # Classes
    # print(PREV_DIR)
    with open(PREV_DIR + "/datasets/" + ds_name + "/" + ds_name + "/raw/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return np.array(classes)


def read_targets(ds_name):
    # Classes
    with open(PREV_DIR + "/datasets/" + ds_name + "/" + ds_name + "/raw/" + ds_name + "_graph_attributes.txt", "r") as f:
        classes = [float(i) for i in list(f)]
    f.closed

    return np.array(classes)


def read_multi_targets(ds_name):
    # Classes
    with open("datasets/" + ds_name + "/" + ds_name + "/raw/" + ds_name + "_graph_attributes.txt", "r") as f:
        classes = [[float(j) for j in i.split(",")] for i in list(f)]
    f.closed

    return np.array(classes)


# Download dataset, regression problem=False, multi-target regression=False.
def get_dataset(dataset, regression=False, multi_target_regression=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', dataset)
    download_start = time.time()
    TUDataset(path, name=dataset)
    download_time = time.time() - download_start
    print('download_time: {:.2f}'.format(download_time))

    if multi_target_regression:
        return read_multi_targets(dataset)
    read_start = time.time()
    if not regression:
        graph = read_classes(dataset)
    else:
        graph = read_targets(dataset)
    read_time = time.time() - read_start
    print("read graph time: {:.2f}".format(read_time))
    return graph