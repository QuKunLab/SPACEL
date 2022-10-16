import subprocess
import os
import numpy as np

def auto_cuda_device():
    out = subprocess.getoutput('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free')
    if len(out.split('\n')) > 1:
        memory_available = [int(x.split()[2]) for x in out.split('\n')]
        max_idx = np.where(memory_available == np.max(memory_available))[0]
        os.environ['CUDA_VISIBLE_DEVICES']=str(np.random.permutation(max_idx)[0])
        print('Using GPU:',os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        raise ValueError('Invalid output from nvidia-smi.')

def set_environ_seed(seed=42):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED']=str(seed)
    print('Setting environment seed:',seed)
    
