from . import data_utils
from . import base_model
from . import data_downsample
from . import data_augmentation
from . import spatial_simulation
import numpy as np
import tensorflow as tf
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
    st_batch_key=None,
    sm_size:int=500000,
    downsample=False,
    downsample_fraction=None,
    data_aug=True,
    max_rate=0.8,max_val=0.8,kth=0.2,
    hiddem_dims=512,
    n_threads=4,
    always_batch_norm=False,
    rec_loss_axis=0,
    seed=42
):
    print('Setting global seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Settings for dynamic allocation of gpu memory
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    numba.set_num_threads(n_threads)

    sc_ad = data_utils.normalize_adata(sc_ad,target_sum=1e4)
    st_ad = data_utils.normalize_adata(st_ad,target_sum=1e4)
    sc_ad, st_ad = data_utils.filter_model_genes(sc_ad,st_ad,celltype_key=celltype_key,deg_method=deg_method,n_top_markers=n_top_markers)
 
    sm_ad = data_utils.generate_sm_adata(sc_ad,num_sample=sm_size,celltype_key=celltype_key,n_threads=n_threads)
    data_utils.downsample_sm_spot_counts(sm_ad,st_ad,n_threads=n_threads)

    model = base_model.SpointModel(
        st_ad,
        sm_ad,
        clusters = np.array(sm_ad.obsm['label'].columns),
        spot_names = np.array(st_ad.obs_names),
        used_genes = np.array(st_ad.var_names),
        st_batch_key=st_batch_key,
        hidden_dims=hiddem_dims,
        always_batch_norm=always_batch_norm,
        rec_loss_axis=rec_loss_axis
    )
    return model
