from . import data_utils
from . import base_model
from . import data_downsample
from . import data_augmentation
from . import spatial_simulation
import numpy as np
import torch
from scipy.sparse import csr_matrix
import numba
import logging
import random
# logging.basicConfig(level=print,
#                     format='%(asctime)s %(levelname)s %(message)s',
#                     datefmt='%m-%d %H:%M')
# logging.getLogger().setLevel(print)

def init_model(
    sc_ad,
    st_ad,
    celltype_key,
    sc_genes=None,
    st_genes=None,
    used_genes=None,
    deg_method:str='wilcoxon',
    n_top_markers:int=200,
    n_top_hvg:int=None,
    log2fc_min=0.5,
    pval_cutoff=0.01,
    pct_diff=None, 
    pct_min=0.1,
    use_rep='scvi',
    st_batch_key=None,
    sm_size:int=500000,
    cell_counts=None,
    clusters_mean=None,
    cells_mean=10,
    cells_min=1,
    cells_max=20,
    cell_sample_counts=None,
    cluster_sample_counts=None,
    ncell_sample_list=None,
    cluster_sample_list=None,
    scvi_layers=2,
    scvi_latent=128,
    scvi_gene_likelihood='zinb',
    scvi_dispersion='gene-batch',
    latent_dims=128, 
    hidden_dims=512,
    infer_losses=['kl','cos'],
    n_threads=4,
    seed=42,
    use_gpu=None
):
    """Initialize Spoint model.
    
    Given specific data and parameters to initialize Spoint model.

    Args:
        sc_ad: An AnnData object representing single cell reference.
        st_ad: An AnnData object representing spatial transcriptomic data.
        celltype_key: A string representing cell types annotation columns in obs of single cell reference.
        sc_genes: A sequence of strings containing genes of single cell reference used in Spoint model. Only used when ``used_genes`` is None.
        st_genes: A sequence of strings containing genes of spatial transcriptomic data used in Spoint model. Only used when ``used_genes`` is None.
        used_genes: A sequence of strings containing genes used in Spoint model.
        deg_method: A string passed to method parameter of scanpy.tl.rank_genes_groups.
        n_top_markers: The number of differential expressed genes in each cell type of single cell reference used in Spoint model.
        n_top_hvg: The number of highly variable genes of spatial transcriptomic data used in Spoint model.
        log2fc_min: The threshold of log2 fold-change used for filtering differential expressed genes of single cell reference.
        pval_cutoff: The threshold of p-value used for filtering differential expressed genes of single cell reference.
        pct_min: The threshold of precentage of expressed cells used for filtering differential expressed genes of single cell reference.
        st_batch_key: A column name in obs of spatial transcriptomic data representing batch groups of spatial transcriptomic data.
        sm_size: The number of simulated spots.
        hiddem_dims: The number of nodes of hidden layers in Spoint model.
        n_threads: The number of cpu core used for parallel.
    
    Returns:
        A ``SpointModel`` object.
    """
    
    print('Setting global seed:', seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    spatial_simulation.numba_set_seed(seed)
    numba.set_num_threads(n_threads)

    sc_ad = data_utils.normalize_adata(sc_ad,target_sum=1e4)
    st_ad = data_utils.normalize_adata(st_ad,target_sum=1e4)
    sc_ad, st_ad = data_utils.filter_model_genes(
        sc_ad,
        st_ad,
        celltype_key=celltype_key,
        deg_method=deg_method,
        n_top_markers=n_top_markers,
        n_top_hvg=n_top_hvg,
        used_genes=used_genes,
        sc_genes=sc_genes,
        st_genes=st_genes,
        log2fc_min=log2fc_min, 
        pval_cutoff=pval_cutoff, 
        pct_diff=pct_diff, 
        pct_min=pct_min
    )
    sm_ad = data_utils.generate_sm_adata(sc_ad,num_sample=sm_size,celltype_key=celltype_key,n_threads=n_threads,cell_counts=cell_counts,clusters_mean=clusters_mean,cells_mean=cells_mean,cells_min=cells_min,cells_max=cells_max,cell_sample_counts=cell_sample_counts,cluster_sample_counts=cluster_sample_counts,ncell_sample_list=ncell_sample_list,cluster_sample_list=cluster_sample_list)
    data_utils.downsample_sm_spot_counts(sm_ad,st_ad,n_threads=n_threads)

    model = base_model.SpointModel(
        st_ad,
        sm_ad,
        clusters = np.array(sm_ad.obsm['label'].columns),
        spot_names = np.array(st_ad.obs_names),
        used_genes = np.array(st_ad.var_names),
        use_rep=use_rep,
        st_batch_key=st_batch_key,
        scvi_layers=scvi_layers,
        scvi_latent=scvi_latent,
        scvi_gene_likelihood=scvi_gene_likelihood,
        scvi_dispersion=scvi_dispersion,
        latent_dims=latent_dims, 
        hidden_dims=hidden_dims,
        infer_losses=infer_losses,
        use_gpu=use_gpu,
        seed=seed
    )
    return model