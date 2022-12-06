import os 
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
from copy import deepcopy
import time
from functools import partial

def rotate_loc(x,y,x0,y0,angle):
    angle = angle/(180/np.pi)
    x1=np.cos(angle)*(x-x0)-np.sin(angle)*(y-y0) + x0
    y1=np.cos(angle)*(y-y0)+np.sin(angle)*(x-x0) + y0
    return x1,y1

# Deprecated
def neighbor_simi(source_loc,target_loc, source_cluster, target_cluster, target_p_dis,k,knn_exclude_cutoff):
    dis_knn_exclude_cutoff = np.median(np.partition(target_p_dis,kth=k+knn_exclude_cutoff,axis=1)[:,k+knn_exclude_cutoff])
    p_dis = pairwise_distances(source_loc,target_loc)
    top5_neighbors = np.argpartition(p_dis,kth=k,axis=1)[:,:k]
    n_simi = []
    for i, sc in enumerate(source_cluster):
        tmp_neighbors = list(top5_neighbors[i])
        tmp_neighbors_reversed = tmp_neighbors[::-1]
        for j in tmp_neighbors_reversed:
            if p_dis[i,j] > dis_knn_exclude_cutoff:
                tmp_neighbors.remove(j)
            else:
                break
        if len(tmp_neighbors) != 0: 
            n_simi.append((target_cluster[tmp_neighbors] == sc).sum()/len(tmp_neighbors))
    return np.mean(n_simi) - (len(n_simi)/i-1)**2

def neighbor_simi_fast(source_loc,target_loc, source_cluster, target_cluster,k,knn_exclude_cutoff,p=2,a=1):
    target_tree = BallTree(target_loc, leaf_size=15, metric='minkowski')
    t_dis, _ = target_tree.query(target_loc, k=k+knn_exclude_cutoff)
    dis_knn_exclude_cutoff = np.median(t_dis[:,k+knn_exclude_cutoff-1])
    s_dis, nearest_neighbors = target_tree.query(source_loc, k=k)
    nearest_neighbors[s_dis > dis_knn_exclude_cutoff] = -1
    kept_ind = (nearest_neighbors != -1).sum(1)>0
    if kept_ind.sum() > 0:
        nearest_neighbors = nearest_neighbors[kept_ind]
        neighbors_cluster = deepcopy(nearest_neighbors)
        neighbors_cluster[nearest_neighbors!=-1] = target_cluster[nearest_neighbors[nearest_neighbors!=-1]]
        simi = (((neighbors_cluster - source_cluster[kept_ind].reshape(-1,1)) == 0).sum(1)/(neighbors_cluster!=-1).sum(1)).mean()
    else:
        simi = 0
    overlap = kept_ind.sum()/len(source_cluster)
    neighbors_simi = simi - a*((1-overlap)**p)
    return neighbors_simi

def neighbor_simi_new(source_loc,target_loc, source_cluster, target_cluster,k,knn_exclude_cutoff,p=2,a=1):
    target_tree = BallTree(target_loc, leaf_size=15, metric='minkowski')
    source_tree = BallTree(source_loc, leaf_size=15, metric='minkowski')
    t_dis, _ = target_tree.query(target_loc, k=k+knn_exclude_cutoff)
    dis_knn_exclude_cutoff = np.median(t_dis[:,k+knn_exclude_cutoff-1])
    s_dis, target_nearest_neighbors = target_tree.query(source_loc, k=k)
    _, source_nearest_neighbors = source_tree.query(source_loc, k=k)
    target_nearest_neighbors[s_dis > dis_knn_exclude_cutoff] = -1
    kept_ind = (target_nearest_neighbors != -1).sum(1)>0
    if kept_ind.sum() > 0:
        target_nearest_neighbors = target_nearest_neighbors[kept_ind]
        target_neighbors_cluster = deepcopy(target_nearest_neighbors)
        target_neighbors_cluster[target_nearest_neighbors!=-1] = target_cluster[target_nearest_neighbors[target_nearest_neighbors!=-1]]
        source_neighbors_cluster = source_cluster[source_nearest_neighbors][kept_ind]
        cluster_num = max(np.unique(np.concatenate([target_cluster,source_cluster])))+1
        cluster_onehot = np.concatenate([np.eye(cluster_num),np.zeros((1,cluster_num))])
        # simi = (((neighbors_cluster - source_cluster[kept_ind].reshape(-1,1)) == 0).sum(1)/(neighbors_cluster!=-1).sum(1)).mean()
        target_cluster_count = cluster_onehot[target_neighbors_cluster].sum(1)
        source_cluster_count = cluster_onehot[source_neighbors_cluster].sum(1)
        target_cluster_prop = target_cluster_count/target_cluster_count.sum(1,keepdims=True)
        source_cluster_prop = source_cluster_count/source_cluster_count.sum(1,keepdims=True)
        simi = ((source_cluster_prop - target_cluster_prop)**2).sum(1).mean()/2
    else:
        simi = 1
    overlap = kept_ind.sum()/len(source_cluster)
    neighbors_simi = simi + a*((1-overlap)**p)
    # print(simi,overlap)
    return -neighbors_simi

def score(warp_param, target_loc, source_loc, target_cluster, source_cluster,k,knn_exclude_cutoff,p,a):
    new_source_loc = []
    x0 = 0
    y0 = 0
    for x,y in source_loc:
        new_source_loc.append(rotate_loc(x,y,x0,y0,warp_param[0]))
    new_source_loc = np.array(new_source_loc)
    new_source_loc[:,0] += warp_param[1]  #x方向平移
    new_source_loc[:,1] += warp_param[2]  #y方向平移
    return -neighbor_simi_fast(new_source_loc, target_loc, source_cluster, target_cluster, k, knn_exclude_cutoff,p,a)

def optimize(target_loc, source_loc, target_cluster, source_cluster, n_neighbors, knn_exclude_cutoff ,p,a, bound_alpha,n_threads,*args,**kwargs): 

    source_loc_flip = deepcopy(source_loc)
    source_loc_flip[:,0] = -source_loc_flip[:,0]
    func1 = partial(score, target_loc=target_loc, source_loc=source_loc, target_cluster=target_cluster, source_cluster=source_cluster,k=n_neighbors,knn_exclude_cutoff=knn_exclude_cutoff,p=p,a=a)
    opm1 = differential_evolution(func1, bounds=((0, 360), (target_loc.min(0)[0]*bound_alpha, target_loc.max(0)[0]*bound_alpha), (target_loc.min(0)[1]*bound_alpha, target_loc.max(0)[1]*bound_alpha)),workers=n_threads,*args,**kwargs)
    score1 = -opm1.fun
    result1 = opm1.x
    
    func2 = partial(score, target_loc=target_loc, source_loc=source_loc_flip, target_cluster=target_cluster, source_cluster=source_cluster,k=n_neighbors,knn_exclude_cutoff=knn_exclude_cutoff,p=p,a=a)
    opm2 = differential_evolution(func2,bounds=((0, 360), (target_loc.min(0)[0]*bound_alpha, target_loc.max(0)[0]*bound_alpha), (target_loc.min(0)[1]*bound_alpha, target_loc.max(0)[1]*bound_alpha)),workers=n_threads,*args,**kwargs)
    score2 = -opm2.fun
    result2 = opm2.x

    print(f'Optimal score: flip = {score1}, not flip = {score2}')

    if score1 > score2:
        return 0, result1, score1, 1, result2, score2
    else:
        return 1, result2, score2, 0, result1, score1

def align_pairwise(param, n_neighbors, knn_exclude_cutoff,p,a,bound_alpha,n_threads,*args,**kwargs):
    # ???i
    i, target_loc,source_loc,target_cluster,source_cluster = param
    flip, result, score, alter_flip, alter_result,alter_score  = optimize(target_loc, source_loc, target_cluster, source_cluster, n_neighbors, knn_exclude_cutoff, p,a,bound_alpha,n_threads,*args,**kwargs)
    return [i,flip,result[0],result[1],result[2],score,alter_flip,alter_result[0],alter_result[1],alter_result[2],alter_score]

def align(
    ad_list,
    cluster_key='spatial_domain',
    output_path=None,
    raw_loc_key='spatial',
    aligned_loc_key='spatial_aligned',
    n_neighbors=5,
    knn_exclude_cutoff=1,
    p=2,
    a=1,
    bound_alpha=1,
    write_loc_path=None,
    n_threads=1,
    seed=42,
    subset_prop=None,
    *args,
    **kwargs
):
    if subset_prop is not None:
        for i in range(len(ad_list)):
            ad_list[i] = ad_list[i][np.random.permutation(ad_list[i].obs_names)[:int(ad_list[i].shape[0]*subset_prop)]].copy()
    if output_path is None:
        output_path = 'Scube_outputs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # centering X, Y coordinate
    for i in range(len(ad_list)):
        raw_loc = np.asarray(ad_list[i].obsm[raw_loc_key], dtype=np.float32)
        raw_loc[:,:2] = raw_loc[:,:2] - np.median(raw_loc[:,:2],axis=0)
        ad_list[i].obsm['spatial_pair'] = raw_loc
    
    start = time.time()
    print('Start alignment...')
    res = []
    for i in range(1,len(ad_list)):
        print(f'Alignment slice {i} to {i-1}')
        target_ind = i-1
        source_ind = i
        target_xy = ad_list[target_ind].obsm['spatial_pair'][:,:2]
        source_xy = ad_list[source_ind].obsm['spatial_pair'][:,:2]
        target_cluster = np.asarray(ad_list[target_ind].obs[cluster_key])
        source_cluster = np.asarray(ad_list[source_ind].obs[cluster_key])
        param = [i, target_xy, source_xy, target_cluster, source_cluster]
        r = align_pairwise(param, n_neighbors=n_neighbors, knn_exclude_cutoff=knn_exclude_cutoff,p=p,a=a,bound_alpha=bound_alpha, n_threads=n_threads,seed=seed)
        res.append(r)
    
    warp_info=np.array(res,dtype=np.float32)
    np.save(os.path.join(output_path,'warp_info.npy'),warp_info)
    
    warp_info = warp_info[:,:6]
    score = warp_info[:,5]
    print('All score:', str(score))
    print('Runtime: ' + str(time.time() - start),'s')

    ad_list[0].obsm[aligned_loc_key] = ad_list[0].obsm['spatial_pair']
    for i in range(1,len(ad_list)):
        target_ind = i-1
        source_ind = i
        target_loc = ad_list[target_ind].obsm['spatial_pair']
        source_loc = ad_list[source_ind].obsm['spatial_pair']
        target_cluster = np.asarray(ad_list[target_ind].obs[cluster_key])
        source_cluster = np.asarray(ad_list[source_ind].obs[cluster_key])
        old_source_loc = deepcopy(source_loc)
        for r in warp_info[:source_ind][::-1]:
            if r[1] == 1:
                loc_flip = deepcopy(old_source_loc)
                loc_flip[:,0] = -loc_flip[:,0]
                old_source_loc = loc_flip
            new_source_xy = []
            for _x,_y in old_source_loc[:,:2]:
                new_source_xy.append(rotate_loc(_x,_y,0,0,r[2]))
            new_source_xy = np.array(new_source_xy)
            new_source_loc = deepcopy(source_loc)
            new_source_loc[:,:2] = new_source_xy
            # translate loc
            new_source_loc[:,0] += r[3]
            new_source_loc[:,1] += r[4]
            old_source_loc = deepcopy(new_source_loc)
        ad_list[source_ind].obsm[aligned_loc_key] = new_source_loc

    # !!!why dataframe?
    for i in range(len(ad_list)):
        if isinstance (ad_list[i].obsm[raw_loc_key],pd.DataFrame):
            columns = ad_list[i].obsm[raw_loc_key].columns
        elif ad_list[i].obsm[raw_loc_key].shape[1] == 3:
            columns = ['X','Y','Z']
        else:
            columns = ['X','Y']
        aligned_loc = pd.DataFrame(ad_list[i].obsm[aligned_loc_key], columns=columns, index=ad_list[i].obs.index)
        ad_list[i].obsm[aligned_loc_key] = aligned_loc
    
    if write_loc_path is not None:
        coo = pd.DataFrame()
        for i in range(len(ad_list)):
            loc = ad_list[i].obsm[aligned_loc_key]
            coo = pd.concat([coo,loc],axis=0)
        coo.to_csv(write_loc_path)
