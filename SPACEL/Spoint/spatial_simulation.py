import numpy as np
import pandas as pd
import numba as nb
from numba import jit
import collections
from .data_downsample import downsample_cell,downsample_matrix_by_cell
from .data_augmentation import random_augment,random_augmentation_cell
import logging
# logging.basicConfig(level=print,
#                     format='%(asctime)s %(levelname)s %(message)s',
#                     datefmt='%m-%d %H:%M')
# logging.getLogger().setLevel(print)

# 汇总每个spot的细胞数，统计细胞数的分布
def count_cell_counts(cell_counts):
    cell_counts = np.array(cell_counts.values,dtype=int).reshape(-1)
    counts_list = np.array(np.histogram(cell_counts,range=[0,np.max(cell_counts)+1],bins=np.max(cell_counts)+1)[0],dtype=int)
    counts_index = np.array((np.histogram(cell_counts,range=[0,np.max(cell_counts)+1],bins=np.max(cell_counts)+1)[1][:-1]),dtype=int)
    counts_df = pd.DataFrame(counts_list,index=counts_index,columns=['count'],dtype=np.int32)
    counts_df = counts_df[(counts_df['count'] != 0) & (counts_df.index != 0)]
    count_sum = 0
    for i in np.arange(len(counts_df)):
        count_sum += counts_df.iloc[i].values
        if count_sum > counts_df.values.sum()*0.99:
            counts_df_filtered = counts_df.iloc[:i+1,:]
            break
    return counts_df_filtered

# 对某个axis调用numpy函数(numba版本)
@nb.njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

# 对某个axis计算均值(numba版本)
@nb.njit
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)

# 对某个axis计算加和(numba版本)
@nb.njit
def np_sum(array, axis):
    return np_apply_along_axis(np.sum, axis, array)

# 根据参数采样单细胞数据，生成模拟spot(numba版本)
@jit(nopython=True,parallel=True)
def sample_cell(param_list,cluster_p,clusters,cluster_id,sample_exp,sample_cluster,cell_p_balanced,downsample_fraction=None,data_augmentation=True,max_rate=0.8,max_val=0.8,kth=0.2):
    exp = np.empty((len(param_list), sample_exp.shape[1]),dtype=np.float32)
    density = np.empty((len(param_list), sample_cluster.shape[1]),dtype=np.float32)

    for i in nb.prange(len(param_list)):
        params = param_list[i]
        num_cell = params[0]
        num_cluster = params[1]
        used_clusters = clusters[np.searchsorted(np.cumsum(cluster_p), np.random.rand(num_cluster), side="right")]
        cluster_mask=np.array([False]*len(cluster_id))
        for c in used_clusters:
            cluster_mask = (cluster_id==c)|(cluster_mask)
        used_cell_ind = np.where(cluster_mask)[0]
        used_cell_p = cell_p_balanced[cluster_mask]
        used_cell_p = used_cell_p/used_cell_p.sum()
        sampled_cells = used_cell_ind[np.searchsorted(np.cumsum(used_cell_p), np.random.rand(num_cell), side="right")]
        combined_exp = np_sum(sample_exp[sampled_cells,:],axis=0).astype(np.float32)
        if data_augmentation:
            combined_exp = random_augmentation_cell(combined_exp,max_rate=max_rate,max_val=max_val,kth=kth)
        if downsample_fraction is not None:
            combined_exp = downsample_cell(combined_exp, downsample_fraction)
        combined_clusters = np_sum(sample_cluster[cluster_id[sampled_cells]],axis=0).astype(np.float32)
        exp[i,:] = combined_exp
        density[i,:] = combined_clusters
    return exp,density

def init_sample_prob(sc_ad,celltype_key):
    print('### Initializing sample probability')
    sc_ad.uns['celltype2num'] = pd.DataFrame(
        np.arange(len(sc_ad.obs[celltype_key].value_counts())).T,
        index=sc_ad.obs[celltype_key].value_counts().index.values,
        columns=['celltype_num']
    )
    sc_ad.obs['celltype_num'] = [sc_ad.uns['celltype2num'].loc[c,'celltype_num'] for c in sc_ad.obs[celltype_key]]
    cluster_p_unbalance = sc_ad.obs['celltype_num'].value_counts()/sc_ad.obs['celltype_num'].value_counts().sum()
    cluster_p_sqrt = np.sqrt(sc_ad.obs['celltype_num'].value_counts())/np.sqrt(sc_ad.obs['celltype_num'].value_counts()).sum()
    cluster_p_balance = pd.Series(
        np.ones(len(sc_ad.obs['celltype_num'].value_counts()))/len(sc_ad.obs['celltype_num'].value_counts()), 
        index=sc_ad.obs['celltype_num'].value_counts().index
    )
#     cluster_p_balance = np.ones(len(sc_ad.obs['celltype_num'].value_counts()))/len(sc_ad.obs['celltype_num'].value_counts())
    cell_p_balanced = [1/cluster_p_unbalance[c] for c in sc_ad.obs['celltype_num']]
    cell_p_balanced = np.array(cell_p_balanced)/np.array(cell_p_balanced).sum()
    sc_ad.obs['cell_p_balanced'] = cell_p_balanced
    sc_ad.uns['cluster_p_balance'] = cluster_p_balance
    sc_ad.uns['cluster_p_sqrt'] = cluster_p_sqrt
    sc_ad.uns['cluster_p_unbalance'] = cluster_p_unbalance
    return sc_ad

# 将表达矩阵转化成array
def generate_sample_array(sc_ad, used_genes):
    if used_genes is not None:
        sc_df = sc_ad.to_df().loc[:,used_genes]
    else:
        sc_df = sc_ad.to_df()
    return sc_df.values

# 从均匀分布中获取每个spot采样的细胞数和细胞类型数
def get_param_from_uniform(num_sample,cells_min=None,cells_max=None,clusters_min=None,clusters_max=None):

    cell_count = np.asarray(np.ceil(np.random.uniform(int(cells_min),int(cells_max),size=num_sample)),dtype=int)
    cluster_count = np.asarray(np.ceil(np.clip(np.random.uniform(clusters_min,clusters_max,size=num_sample),1,cell_count)),dtype=int)
    return cell_count, cluster_count

# 从高斯分布中获取每个spot采样的细胞数和细胞类型数
def get_param_from_gaussian(num_sample,cells_min=None,cells_max=None,cells_mean=None,cells_std=None,clusters_mean=None,clusters_std=None):

    cell_count = np.asarray(np.ceil(np.clip(np.random.normal(cells_mean,cells_std,size=num_sample),int(cells_min),int(cells_max))),dtype=int)
    cluster_count = np.asarray(np.ceil(np.clip(np.random.normal(clusters_mean,clusters_std,size=num_sample),1,cell_count)),dtype=int)
    return cell_count,cluster_count

# 从用空间数据估计的cell counts中获取每个spot采样的细胞数和细胞类型数
def get_param_from_cell_counts(
    num_sample,
    cell_counts,
    cluster_sample_mode='gaussian',
    clusters_mean=None,clusters_std=None,
    clusters_min=None,clusters_max=None
):
    if cluster_sample_mode == 'gaussian':
        cluster_count = np.asarray(np.ceil(np.clip(np.random.normal(clusters_mean,clusters_std,size=num_sample),1,cell_counts)),dtype=int)
    elif cluster_sample_mode == 'uniform':
        cluster_count = np.asarray(np.ceil(np.clip(np.random.uniform(clusters_min,clusters_max,size=num_sample),1,cell_counts)),dtype=int)
    else:
        raise TypeError('Not correct sample method.')
    return cell_counts,cluster_count

# 获取每个cluster的采样概率
def get_cluster_sample_prob(sc_ad,mode):
    if mode == 'unbalance':
        cluster_p = sc_ad.uns['cluster_p_unbalance'].values
    elif mode == 'balance':
        cluster_p = sc_ad.uns['cluster_p_balance'].values
    elif mode == 'sqrt':
        cluster_p = sc_ad.uns['cluster_p_sqrt'].values
    else:
        raise TypeError('Balance argument must be one of [ None, banlance, sqrt ].')
    return cluster_p

def cal_downsample_fraction(sc_ad,st_ad,celltype_key=None):
    st_counts_median = np.median(st_ad.X.sum(axis=1))
    simulated_st_data, simulated_st_labels = generate_simulation_data(sc_ad,num_sample=10000,celltype_key=celltype_key,balance_mode=['unbalance'])
    simulated_st_counts_median = np.median(simulated_st_data.sum(axis=1))
    if st_counts_median < simulated_st_counts_median:
        fraction = st_counts_median / simulated_st_counts_median
        print(f'### Simulated data downsample fraction: {fraction}')
        return fraction
    else:
        return None

# 生成模拟数据
def generate_simulation_data(
    sc_ad,
    celltype_key,
    num_sample: int, 
    used_genes=None,
    balance_mode=['unbalance','sqrt','balance'],
    cell_sample_method='gaussian',
    cluster_sample_method='gaussian',
    cell_counts=None,
    downsample_fraction=None,
    data_augmentation=True,
    max_rate=0.8,max_val=0.8,kth=0.2,
    cells_min=1,cells_max=20,
    cells_mean=10,cells_std=5,
    clusters_mean=None,clusters_std=None,
    clusters_min=None,clusters_max=None,
    n_cpus=None
):
    if not 'cluster_p_unbalance' in sc_ad.uns:
        sc_ad = init_sample_prob(sc_ad,celltype_key)
    num_sample_per_mode = num_sample//len(balance_mode)
    cluster_ordered = np.array(sc_ad.obs['celltype_num'].value_counts().index)
#     print(cluster_ordered)
    cluster_num = len(cluster_ordered)
#     print(cluster_num)
    cluster_id = sc_ad.obs['celltype_num'].values
#     print(cluster_id)
    cluster_mask = np.eye(cluster_num)
#     print(cluster_mask)
    
    if cell_counts is not None:
        cells_mean = np.mean(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)])
        cells_std = np.std(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)])
        cells_min = int(np.min(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)]))
        cells_max = int(np.max(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)]))
    if clusters_mean is None:
        clusters_mean = cells_mean/2
    if clusters_std is None:
        clusters_std = cells_std/2
    if clusters_min is None:
        clusters_min = cells_min
    if clusters_max is None:
        clusters_max = np.min((cells_max//2,cluster_num))
            
    if cell_counts is not None:
        cell_counts, cluster_count = get_param_from_cell_counts(num_sample_per_mode,cell_counts,cluster_sample_method,cells_mean=cells_mean,cells_std=cells_std,cells_max=cells_max,cells_min=cells_min,clusters_mean=clusters_mean,clusters_std=clusters_std,clusters_min=clusters_min,clusters_max=clusters_max)
    elif cell_sample_method == 'gaussian':
        cell_counts, cluster_count = get_param_from_gaussian(num_sample_per_mode,cells_mean=cells_mean,cells_std=cells_std,cells_max=cells_max,cells_min=cells_min,clusters_mean=clusters_mean,clusters_std=clusters_std)
    elif cell_sample_method == 'uniform':
        cell_counts, cluster_count = get_param_from_uniform(num_sample_per_mode,cells_max=cells_max,cells_min=cells_min,clusters_min=clusters_min,clusters_max=clusters_max)
    else:
        raise TypeError('Not correct sample method.')
    params = np.array(list(zip(cell_counts, cluster_count)))

    sample_data_list = []
    sample_labels_list = []
    for b in balance_mode:
        print(f'### Genetating simulated spatial data using scRNA data with mode: {b}')
        cluster_p = get_cluster_sample_prob(sc_ad,b)
        if downsample_fraction is not None:
            if downsample_fraction > 0.035:
                sample_data,sample_labels = sample_cell(
                    param_list=params,
                    cluster_p=cluster_p,
                    clusters=cluster_ordered,
                    cluster_id=cluster_id,
                    sample_exp=generate_sample_array(sc_ad,used_genes),
                    sample_cluster=cluster_mask,
                    cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                    downsample_fraction=downsample_fraction,
                    data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                )
            else:
                sample_data,sample_labels = sample_cell(
                    param_list=params,
                    cluster_p=cluster_p,
                    clusters=cluster_ordered,
                    cluster_id=cluster_id,
                    sample_exp=generate_sample_array(sc_ad,used_genes),
                    sample_cluster=cluster_mask,
                    cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                    data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
                )
                # logging.warning('### Downsample data with python backend')
                sample_data = downsample_matrix_by_cell(sample_data, downsample_fraction, n_cpus=n_cpus, numba_end=False)
        else:
            sample_data,sample_labels = sample_cell(
                param_list=params,
                cluster_p=cluster_p,
                clusters=cluster_ordered,
                cluster_id=cluster_id,
                sample_exp=generate_sample_array(sc_ad,used_genes),
                sample_cluster=cluster_mask,
                cell_p_balanced=sc_ad.obs['cell_p_balanced'].values,
                data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
            )
#         if data_augmentation:
#             sample_data = random_augment(sample_data)
        sample_data_list.append(sample_data)
        sample_labels_list.append(sample_labels)
    return np.concatenate(sample_data_list), np.concatenate(sample_labels_list)

@jit(nopython=True,parallel=True)
def sample_cell_exp(cell_counts,sample_exp,cell_p,downsample_fraction=None,data_augmentation=True,max_rate=0.8,max_val=0.8,kth=0.2):
    exp = np.empty((len(cell_counts), sample_exp.shape[1]),dtype=np.float32)
    ind = np.zeros((len(cell_counts), np.max(cell_counts)),dtype=np.int32)
    cell_ind = np.arange(sample_exp.shape[0])
    for i in nb.prange(len(cell_counts)):
        num_cell = cell_counts[i]
        sampled_cells=cell_ind[np.searchsorted(np.cumsum(cell_p), np.random.rand(num_cell), side="right")]
        combined_exp=np_sum(sample_exp[sampled_cells,:],axis=0).astype(np.float64)
#         print(combined_exp.dtype)
        if downsample_fraction is not None:
            combined_exp = downsample_cell(combined_exp, downsample_fraction)
        if data_augmentation:
            combined_exp = random_augmentation_cell(combined_exp,max_rate=max_rate,max_val=max_val,kth=kth)
        exp[i,:] = combined_exp
        ind[i,:cell_counts[i]] = sampled_cells + 1
    return exp,ind

def generate_simulation_st_data(
    st_ad,
    num_sample: int, 
    used_genes=None,
    balance_mode=['unbalance'],
    cell_sample_method='gaussian',
    cell_counts=None,
    downsample_fraction=None,
    data_augmentation=True,
    max_rate=0.8,max_val=0.8,kth=0.2,
    cells_min=1,cells_max=10,
    cells_mean=5,cells_std=3,
):
    print('### Genetating simulated data using spatial data')
    cell_p = np.ones(len(st_ad))/len(st_ad)
    if cell_counts is not None:
        cells_mean = np.mean(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)])
        cells_std = np.std(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)])
        cells_min = int(np.min(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)]))
        cells_max = int(np.max(np.sort(cell_counts)[int(len(cell_counts)*0.05):int(len(cell_counts)*0.95)]))
    elif cell_sample_method == 'gaussian':
        cell_counts = np.asarray(np.ceil(np.clip(np.random.normal(cells_mean,cells_std,size=num_sample),int(cells_min),int(cells_max))),dtype=int)
    elif cell_sample_method == 'uniform':
        cell_counts = np.asarray(np.ceil(np.random.uniform(int(cells_min),int(cells_max),size=num_sample)),dtype=int)
    else:
        raise TypeError('Not correct sample method.')

    sample_data,sample_ind = sample_cell_exp(
        cell_counts=cell_counts,
        sample_exp=generate_sample_array(st_ad,used_genes),
        cell_p=cell_p,
        downsample_fraction=downsample_fraction,
        data_augmentation=data_augmentation,max_rate=max_rate,max_val=max_val,kth=kth,
    )
#     if data_augmentation:
#         sample_data = random_augment(sample_data)
    return sample_data,sample_ind
