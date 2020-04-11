from __future__ import division
from auxiliarymethods.evaluation import kernel_svm_evaluation, linear_svm_evaluation, sgd_regressor_evaluation, ridge_regressor_evaluation, kernel_ridge_regressor_evaluation
from auxiliarymethods.evaluation import gnn_evaluation
from gnn_baselines.gin import GIN0
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mse
import kernel_baselines as kb
import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
from sklearn.preprocessing import StandardScaler
import numpy as np


def main():
    dp.get_dataset("ZINC_train", regression=True)
    dp.get_dataset("ZINC_val", regression=True)
    dp.get_dataset("ZINC_test", regression=True)

    indices_train = []
    indices_val = []
    indices_test = []

    infile = open("datasets/test.index.txt", "r")
    for line in infile:
        indices_test = line.split(",")
        indices_test = [int(i) for i in indices_test]

    infile = open("datasets/val.index.txt", "r")
    for line in infile:
        indices_val = line.split(",")
        indices_val = [int(i) for i in indices_val]

    infile = open("datasets/train.index.txt", "r")
    for line in infile:
        indices_train = line.split(",")
        indices_train = [int(i) for i in indices_train]

    targets = kb.read_targets("ZINC_train", indices_train)
    targets.extend(kb.read_targets("ZINC_val", indices_val))
    targets.extend(kb.read_targets("ZINC_test", indices_test))
    targets = np.array(targets)
    print(len(targets))

    print("###")

    all_matrices = kb.compute_wl_1_sparse_ZINC(True, True, indices_train, indices_val, indices_test)

    print("####")
    indices_train = list(range(0,10000))
    indices_val = list(range(10000,11000))
    indices_test = list(range(11000,12000))

    #all_matrices = [all_matrices[4]]

    # p = sgd_regressor_evaluation(all_matrices, targets, indices_train, indices_val, indices_test, num_repetitions=1,
    #                              alpha=[0.0001])
    # print(p)
    p = ridge_regressor_evaluation(all_matrices, targets, indices_train, indices_val, indices_test, num_repetitions=1,
                                   alpha=[1.0])
    print(p)

    p = sgd_regressor_evaluation(all_matrices, targets, indices_train, indices_val, indices_test, num_repetitions=1,
                                   alpha=[0.0001])
    print(p)

    #
    #
    # dp.get_dataset("ZINC_train", regression=True)
    # dp.get_dataset("ZINC_val", regression=True)
    # dp.get_dataset("ZINC_test", regression=True)
    #
    # indices_train = []
    # indices_val = []
    # indices_test = []
    #
    # infile = open("datasets/test_50.index.txt", "r")
    # for line in infile:
    #     indices_test = line.split(",")
    #     indices_test = [int(i) for i in indices_test]
    #
    # infile = open("datasets/val_50.index.txt", "r")
    # for line in infile:
    #     indices_val = line.split(",")
    #     indices_val = [int(i) for i in indices_val]
    #
    #
    # indices_train = list(range(0,5000))
    #
    # targets = kb.read_targets("ZINC_train", indices_train)
    # targets.extend(kb.read_targets("ZINC_val", indices_val))
    # targets.extend(kb.read_targets("ZINC_test", indices_test))
    # targets = np.array(targets)
    # print(len(targets))
    #
    # print("###")
    #
    # all_matrices = kb.compute_wl_1_sparse_ZINC(True, True, indices_train, indices_val, indices_test)
    # print("####")
    #
    # indices_train = list(range(0,50000))
    # indices_val = list(range(50000,55000))
    # indices_test = list(range(55000,60000))
    #
    # #all_matrices = [all_matrices[4]]
    #
    # # p = sgd_regressor_evaluation(all_matrices, targets, indices_train, indices_val, indices_test, num_repetitions=1,
    # #                              alpha=[0.0001])
    # # print(p)
    # p = ridge_regressor_evaluation(all_matrices, targets, indices_train, indices_val, indices_test, num_repetitions=5,
    #                                alpha=[0.001, 0.01, 0.1, 1.0, 10.0])
    # print(p)
    #
    # p = sgd_regressor_evaluation(all_matrices, targets, indices_train, indices_val, indices_test, num_repetitions=5,
    #                                alpha=[0.001, 0.0001,0.00001])
    # print(p)



    #
    # print("###")
    # all_matrices = []
    # for i in range(0, 6):
    #     all_matrices.append(kb.compute_wl_1_sparse_ZINC(i, True, True, indices_train, indices_val, indices_test))
    #
    # indices_train = list(range(10000))
    # indices_val = list(range(1000))
    # indices_test = list(range(1000))
    # p = sgd_regressor_evaluation(all_matrices, targets, indices_train, indices_val, indices_test)
    # print(p)
    # p = ridge_regressor_evaluation(all_matrices, targets, indices_train, indices_val, indices_test)
    # print(p)


    #
    #
    #
    # print("####################################################################")
    #
    # indices_train = []
    # indices_val = []
    # indices_test = []
    #
    # infile = open("datasets/train_50.index.txt", "r")
    # for line in infile:
    #     indices_train = line.split(",")
    #     indices_train = [int(i) for i in indices_train]
    #
    # infile = open("datasets/val_50.index.txt", "r")
    # for line in infile:
    #     indices_val = line.split(",")
    #     indices_val = [int(i) for i in indices_val]
    #
    # indices_test = list(range(0, 5000))
    #
    # targets = kb.read_targets("ZINC_train", indices_train)
    # targets.extend(kb.read_targets("ZINC_val", indices_val))
    # targets.extend(kb.read_targets("ZINC_test", indices_test))
    # targets = np.array(targets)
    # print(len(targets))
    #
    # print("###")
    # all_matrices = []
    # for i in range(0, 6):
    #     all_matrices.append(kb.compute_wl_1_sparse_ZINC(i, True, True, indices_train, indices_val, indices_test))
    #
    # indices_train = list(range(0,50000))
    # indices_val = list(range(50000,55000))
    # indices_test = list(range(55000,60000))
    # p = sgd_regressor_evaluation(all_matrices, targets, indices_train, indices_val, indices_test)
    # print(p)
    # p = ridge_regressor_evaluation(all_matrices, targets, indices_train, indices_val, indices_test)
    # print(p)
    #
    #
    # datataset = [["TRIANGLES", False, False],
    #              ["github_stargazers", False, False],
    #              ["reddit_threads", False, False]]
    # for d, use_labels, use_edge_labels in datataset:
    #     dataset = d
    #     classes = dp.get_dataset(dataset)
    #
    #     print("WL1")
    #     all_matrices = []
    #     for i in range(1, 6):
    #         gm = kb.compute_wl_1_sparse(dataset, i, use_labels, use_edge_labels)
    #         gm_n = aux.normalize_feature_vector(gm)
    #         all_matrices.append(gm_n)
    #     print("###")
    #     print(linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=True))
    #
    #     print("GR")
    #     all_matrices = []
    #     gm = kb.compute_graphlet_sparse(dataset, use_labels, use_edge_labels)
    #     gm_n = aux.normalize_feature_vector(gm)
    #     all_matrices.append(gm_n)
    #     print("###")
    #     print(linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=True))
    #
    #     print("SP")
    #     all_matrices = []
    #     gm = kb.compute_shortestpath_sparse(dataset, use_labels)
    #     gm_n = aux.normalize_feature_vector(gm)
    #     all_matrices.append(gm_n)
    #     print("###")
    #     print(linear_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True, primal=True))
    #
    #

    # datataset = [
    #                  # ["ENZYMES", True],
    #                  # ["IMDB-BINARY", False], ["IMDB-MULTI", False],
    #                  # ["NCI1", True],
    #                  # ["PROTEINS", True],
    #                  # ["REDDIT-BINARY", False],
    #                  ["deezer_ego_nets", False]]
    # for d, use_labels in datataset:
    #
    #     dataset = d
    #     classes = dp.get_dataset(dataset)
    #
    #     # print("WL1")
    #     # all_matrices = []
    #     # for i in range(1, 6):
    #     #     gm = kb.compute_wl_1_dense(dataset, i, use_labels, False)
    #     #     gm_n = aux.normalize_gram_matrix(gm)
    #     #     all_matrices.append(gm_n)
    #     # print("###")
    #     # print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))
    #
    #     print("WLOA")
    #     all_matrices = []
    #     for i in range(1, 6):
    #         gm = kb.compute_wloa_dense(dataset, i, use_labels, False)
    #         gm_n = aux.normalize_gram_matrix(gm)
    #         all_matrices.append(gm_n)
    #     print("###")
    #     print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))
    #
    #     print("GR")
    #     all_matrices = []
    #     gm = kb.compute_graphlet_dense(dataset, use_labels, False)
    #     gm_n = aux.normalize_gram_matrix(gm)
    #     all_matrices.append(gm_n)
    #     print("###")
    #     print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))
    #
    #     print("SP")
    #     all_matrices = []
    #     gm = kb.compute_shortestpath_dense(dataset, use_labels)
    #     gm_n = aux.normalize_gram_matrix(gm)
    #     all_matrices.append(gm_n)
    #     print("###")
    #     print(kernel_svm_evaluation(all_matrices, classes, num_repetitions=10, all_std=True))


if __name__ == "__main__":
    main()
