import random
import anndata
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import davies_bouldin_score

def one_hot_encode(labels, unique_labels=None):
    if unique_labels is None:
        unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    encoded = np.eye(num_classes)[np.array([label_map[label] for label in labels])]
    return encoded, unique_labels

def add_cell_type_composition(ad, prop_df=None, celltype_anno=None, all_celltypes=None):
    """Add cell type composition.
    
    Adding cell type compostion to AnnData object of spatial transcriptomic data as Splane input. 
    
    Args:
        ad: A AnnData object of spatial transcriptomic data as Splane input.
        prop_df: A DataFrame of cell type composition used for spot-based spatial transcriptomic data.
        celltype_anno: A list containing the cell type annotations for each cell in the single-cell resolution spatial transcriptomic data. This parameter is not used if `prof_ad` is provided.
        all_celltypes: A list of all cell types present in all slices. This parameter is used when a single slice does not cover all cell types in the dataset.

    Returns:
        ``None``
    """
    if prop_df is not None:
        if all_celltypes is not None:
            prop_df.loc[:,np.setdiff1d(all_celltypes, prop_df.columns)] = 0
        ad.obs[prop_df.columns] = prop_df.values
        ad.uns['celltypes'] = prop_df.columns
    elif celltype_anno is not None:
        encoded, unique_celltypes = one_hot_encode(celltype_anno, all_celltypes)
        ad.obs[unique_celltypes] = encoded
        ad.uns['celltypes'] = unique_celltypes
    else:
        raise ValueError("prop_df and celltype_anno can not both be None.")
    
# From scanpy
def _morans_i_mtx(
    g_data: np.ndarray,
    g_indices: np.ndarray,
    g_indptr: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    M, N = X.shape
    assert N == len(g_indptr) - 1
    W = g_data.sum()
    out = np.zeros(M, dtype=np.float_)
    for k in range(M):
        x = X[k, :]
        out[k] = _morans_i_vec_W(g_data, g_indices, g_indptr, x, W)
    return out

# From scanpy
def _morans_i_vec_W(
    g_data: np.ndarray,
    g_indices: np.ndarray,
    g_indptr: np.ndarray,
    x: np.ndarray,
    W: np.float_,
) -> float:
    z = x - x.mean()
    z2ss = (z * z).sum()
    N = len(x)
    inum = 0.0

    for i in range(N):
        s = slice(g_indptr[i], g_indptr[i + 1])
        i_indices = g_indices[s]
        i_data = g_data[s]
        inum += (i_data * z[i_indices]).sum() * z[i]

    return len(x) / W * inum / z2ss

def fill_low_prop(ad,min_prop):
    mtx = ad.X
    mtx[mtx < min_prop] = 0
    ad.X = mtx
    return ad

def cal_celltype_moran(ad):
    moran_vals = _morans_i_mtx(
        ad.obsp['spatial_connectivities'].data,
        ad.obsp['spatial_connectivities'].indices,
        ad.obsp['spatial_connectivities'].indptr,
        ad.X.T
    )
    ad.uns['moran_vals'] = np.nan_to_num(moran_vals)
    
def cal_celltype_weight(ad_list):
    print('Calculating cell type weights...')
    for ad in ad_list:
        cal_celltype_moran(ad)
    moran_min=-1
    morans = ad_list[0].uns['moran_vals'].copy()
    for i, ad in enumerate(ad_list[1:]):
        morans += ad.uns['moran_vals'].copy()
    morans_mean = morans/len(ad_list)
    celltype_weights = morans_mean/morans_mean.sum()
    return celltype_weights, morans_mean

def generate_celltype_ad_list(expr_ad_list,min_prop):
    celltype_ad_list = []
    for expr_ad in expr_ad_list:
        celltype_ad = anndata.AnnData(expr_ad.obs[[c for c in expr_ad.uns['celltypes']]])
        celltype_ad.obs = expr_ad.obs
        celltype_ad.obsm =expr_ad.obsm
        celltype_ad.obsp = expr_ad.obsp
        celltype_ad = fill_low_prop(celltype_ad,min_prop)
        celltype_ad_list.append(celltype_ad)
    return celltype_ad_list

def clustering(Cluster, feature):
    predict_labels = Cluster.fit_predict(feature)
    db = davies_bouldin_score(feature, predict_labels)
    return db
    
def split_ad(ad,by):
    ad_list = []
    for s in np.unique(ad.obs[by]):
        ad_split = ad[ad.obs[by] == s].copy()
        ad_list.append(ad_split)
    return ad_list

