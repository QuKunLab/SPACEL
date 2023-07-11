import numpy as np
from scipy.sparse import coo_matrix, block_diag
import torch
import torch.nn.functional as F
from .pygcn_utils import *

def get_graph_inputs(celltype_ad_list):
    print('Generating GNN inputs...')
    A_list = []
    X_list = []
    for celltype_ad in celltype_ad_list:
        X_tmp = np.matrix(celltype_ad.X,dtype='float32')
        X_list.append(X_tmp)
        A_list.append(coo_matrix(celltype_ad.obsp['spatial_distances'],dtype='float32'))

    X_raw = np.concatenate(X_list)
    class_index = 0
    slice_class = []
    for A_tmp in A_list:
        slice_class = slice_class + [class_index]*A_tmp.shape[0]
        class_index += 1
    A = block_diag(A_list)
    nb_mask = np.argwhere(A > 0).T
    slice_class_onehot = F.one_hot(torch.tensor(slice_class)).float()
    return X_raw,A,nb_mask,slice_class_onehot

def get_graph_kernel(features,adj,k=2):
    features_scaled = (features-features.mean(0))/features.std(0)
    features_scaled = torch.tensor(features_scaled)
    SYM_NORM = False  # symmetric (True) vs. left-only (False) normalization
    L = normalized_laplacian(adj, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, k)
    support = k + 1
    graph = [features_scaled]+T_k
    for _i in range(len(graph))[1:]:
        graph[_i] = sparse_mx_to_torch_sparse_tensor(graph[_i])
    return features_scaled, graph, support

def split_train_test_idx(X,train_prop):
    rand_idx = np.random.permutation(X.shape[0])
    train_idx = rand_idx[:int(len(rand_idx)*train_prop)]
    test_idx = rand_idx[int(len(rand_idx)*train_prop):]
    return train_idx, test_idx