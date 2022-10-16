import numba
import numpy as np

@numba.njit
def random_dropout(cell_expr,max_rate):
    non_zero_mask = np.where(cell_expr!=0)[0]
    zero_mask = np.random.choice(non_zero_mask,int(len(non_zero_mask)*np.float32(np.random.uniform(0,max_rate))))
    cell_expr[zero_mask] = 0
    return cell_expr

@numba.njit
def random_scale(cell_expr,max_val):
    scale_factor = np.float32(1+np.random.uniform(-max_val,max_val))
    cell_expr = cell_expr*scale_factor
    return cell_expr

@numba.njit
def random_shift(cell_expr,kth):
    shift_value = np.random.choice(np.array([1,0,-1]),1)[0]*np.unique(cell_expr)[int(np.random.uniform(0,kth)*len(np.unique(cell_expr)))]
    cell_expr[cell_expr != 0] = cell_expr[cell_expr != 0]+shift_value
    cell_expr[cell_expr < 0] = 0
    return cell_expr

@numba.njit(parallel=True)
def random_augment(mtx,max_rate=0.8,max_val=0.8,kth=0.2):
    for i in numba.prange(mtx.shape[0]):
        random_dropout(mtx[i,:],max_rate=max_rate)
        random_scale(mtx[i,:],max_val=max_val)
        random_shift(mtx[i,:],kth=kth)
    return mtx

@numba.njit
def random_augmentation_cell(cell_expr,max_rate=0.8,max_val=0.8,kth=0.2):
    cell_expr = random_dropout(cell_expr,max_rate=max_rate)
    cell_expr = random_scale(cell_expr,max_val=max_val)
    cell_expr = random_shift(cell_expr,kth=kth)
    return cell_expr