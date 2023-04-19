import sys, os, time

import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
import kernel_baselines as kb
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation, linear_svm_evaluation

def main():
    dataset_map_labels = {"DHFR": True, "FRANKENSTEIN":False, "NCI1": True, "DD": True, "PROTEINS": True, \
                    "Fingerprint": False, "Letter-med": False, "MSRC_21": True, \
                    "COLLAB": False, "deezer_ego_nets": False, "github_stargazers": False, "REDDIT-MULTI-5K": False, \
                    "COLORS-3": False}
    dataset = sys.argv[1]
    use_labels = False
    if dataset in dataset_map_labels:
        use_labels = dataset_map_labels[dataset]
    else:
        print("unsupported dataset, try again!")
        return
    iter_comp = 1
    num_eval = 1
    use_edge_labels = False
    # download dataset
    classes = dp.get_dataset(dataset)
    all_matrices = []
    feature_vectors = []
    comp_start = time.time()
    for i in range(iter_comp):
        # gm = kb.compute_wl_1_dense(dataset, i+1, use_labels, use_edge_labels)
        # gm_n = aux.normalize_gram_matrix(gm)
        # wl_1_finish = time.time()
        # print("wl_1 takes: {:.2f}".format(wl_1_finish - comp_start))

        # gm = kb.compute_graphlet_dense(dataset, use_labels, use_edge_labels)
        # gm_n = aux.normalize_gram_matrix(gm)
        # graphlet_finish = time.time()
        # print("graphlet takes: {:.2f}".format(graphlet_finish - wl_1_finish))

        # gm = kb.compute_shortestpath_dense(dataset, use_labels)
        # gm_n = aux.normalize_gram_matrix(gm)
        # shortestpath_finish = time.time()
        # print("shortestpath takes: {:.2f}".format(shortestpath_finish - graphlet_finish))

        gm = kb.compute_wloa_dense(dataset, i+1, use_labels, use_edge_labels)
        gm_n = aux.normalize_gram_matrix(gm)
        wloa_finish = time.time()
        print("wloa takes: {:.2f}".format(wloa_finish - comp_start))
        # all_matrices.append(gm_n)
    comp_dense_finish = time.time()
    # print("compute dense: {:.2f}".format(comp_dense_finish - comp_start))
    # for i in range(iter_comp):
    #     fv = kb.compute_wl_1_sparse(dataset, i+1, use_labels, False)
    #     fv = aux.normalize_feature_vector(fv)
    #     feature_vectors.append(fv)
    comp_sparse_finish = time.time()
    # print("compute sparse {:.2f} for {}"
    #       .format(comp_sparse_finish - comp_dense_finish, dataset))
    
    #eval
    # acc, s_1, s_2 = kernel_svm_evaluation(all_matrices, classes, num_repetitions=num_eval, all_std=True)
    # kernel_svm_evaluation_finish = time.time()
    # print(acc, s_1, s_2)
    # print("kernel_svm_evaluation time: {:.2f}".format(kernel_svm_evaluation_finish - comp_sparse_finish))
    # acc2, std_10, std_100 = linear_svm_evaluation(feature_vectors, classes, num_repetitions=num_reps, all_std=True)
    # linear_svm_evaluation_finish = time.time()
    # print("linear_svm_evaluation time {:.2f}"
        #   .format( linear_svm_evaluation_finish - kernel_svm_evaluation_finish, dataset))

if __name__ == "__main__":
    main()