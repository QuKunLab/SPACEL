import numpy as np
import pandas as pd
import random
import torch
import squidpy as sq
from .graph import get_graph_inputs,get_graph_kernel,split_train_test_idx
from .utils import generate_celltype_ad_list,cal_celltype_weight
from .base_model import SplaneModel

def init_model(
    expr_ad_list:list,
    n_clusters:int,
    k:int=2,
    use_weight=True,
    train_prop:float=0.5,
    n_neighbors=6,
    min_prop=0.01,
    lr:float=3e-3,
    l1:float=0.01,
    l2:float=0.01,
    latent_dim:int=16,
    hidden_dims:int=64,
    gnn_dropout:float=0.8,
    simi_neighbor=1,
    use_gpu=None,
    seed=42
)->SplaneModel:
    """Initialize Splane model.
    
    Build the model then set the data and paratemters. 
    
    Args:
        expr_ad_list: A list of AnnData object of spatial transcriptomic data as the model input.
        n_clusters: The number of cluster of the model ouput. 
        k: The order of neighbors of a spot for graph construction.
        use_weight: If True, the cell type proportion of Moran was used as the weight of the loss.
        train_prop: The proportion of training set.
        n_neighbors: The number of neighbors for graph construction.
        lr: Learning rate of training.
        latent_dim: The dimension of latent features. It equal to the number of nodes of bottleneck layer.
        hidden_dims: The number of nodes of hidden layers.
        gnn_dropout: The dropout rate of GNN model.
        simi_neighbor: The order of neighbors used for similarity loss. If is None, It equal to the order used for constructed graph.
        seed: Random number seed.
    Returns:
        A ``DataFrame`` contained deconvoluted results. Each row representing a spot, and each column representing a cell type.
    """
    
    print('Setting global seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if use_gpu is None:
        if torch.cuda.is_available():
            use_gpu = True
        else:
            use_gpu = False
    
    for expr_ad in expr_ad_list:
        if 'spatial_connectivities' not in expr_ad.obsp.keys():
            sq.gr.spatial_neighbors(expr_ad,coord_type='grid',n_neighs=n_neighbors)
    celltype_ad_list = generate_celltype_ad_list(expr_ad_list,min_prop=min_prop)
    celltype_weights,morans_mean = cal_celltype_weight(celltype_ad_list)
    kept_ind = celltype_weights > 0
    if not use_weight:
        celltype_weights = np.ones(len(celltype_weights))/len(celltype_weights)
    X,A,nb_mask,slice_class_onehot = get_graph_inputs(celltype_ad_list)
    X_filtered, graph, support = get_graph_kernel(X[:,kept_ind],A,k=k)
    celltype_weights = celltype_weights[kept_ind]
    morans_mean = morans_mean[kept_ind]
    if simi_neighbor == 1:
        nb_mask = nb_mask
    elif simi_neighbor == None:
        nb_mask = np.array(np.where(graph[-1].to_dense())!=0)
        nb_mask = nb_mask[:,nb_mask[0] != nb_mask[1]]
    else:
        raise ValueError('simi_neighbor must be 1 or None.')
    train_idx,test_idx = split_train_test_idx(X,train_prop=0.5)
    if use_gpu:
        for i in range(len(graph)):
            graph[i] = graph[i].cuda()
        slice_class_onehot = slice_class_onehot.cuda()
    return SplaneModel(
        expr_ad_list,
        n_clusters,
        X_filtered,
        graph,
        support,
        slice_class_onehot,
        nb_mask,
        train_idx,
        test_idx,
        celltype_weights,
        morans_mean,
        lr,
        l1,
        l2,
        latent_dim,
        hidden_dims,
        gnn_dropout,
        use_gpu
    )
