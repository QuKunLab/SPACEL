# Installation

## Requirements

To install `SPACEL`, you need install [TensorFlow](https://www.tensorflow.org/) with GPU version first. 
- Create conda environment for `SPACEL`:
```
conda env create -f environment.yml
```
or
```
conda create -n SPACEL -c conda-forge -c default cudnn=7.6 cudatoolkit=10.1 python=3.8 r-base r-fitdistrplus
```
you can choose correct `tensorflow`, `cudnn` and `cudatoolkit` version dependent on your graphic driver version. If you don't need GPU acceleration, you can just skip the installation for `cudnn` and `cudatoolkit`.
- Test if [TensorFlow](https://www.tensorflow.org/) for GPU available:
```
python
>>> import tensorflow as tf
>>> tf.test.gpu_device_name()
```
If these command line have not return available gpu names, please check your gpu driver version. For more detail, look at [CUDA Toolkit Major Component Versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions) and [TensorFlow Build Configurations](https://www.tensorflow.org/install/source#tested_build_configurations)

## Installation
- Install `SPACEL`:
```
pip install -r requirements.txt
python setup.py install
