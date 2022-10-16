# SPACEL: characterizing spatial transcriptome architectures by deep-learning
![Overview](docs/figures/figure1.png "Overview")
SPACEL (**SP**atial **A**rchitecture **C**haracterization by d**E**ep **L**earning) is a Python package of deep-learning-based methods for ST data analysis. SPACEL consists of three modules: 
* Spoint embedded a multiple-layer perceptron with a probabilistic model to deconvolute cell type composition for each spot on single ST slice.
* Splane employs a graph convolutional network approach and an adversarial learning algorithm to identify uniform spatial domains that are transcriptomically and spatially coherent across multiple ST slices.
* Scube automatically transforms the spatial coordinate systems of consecutive slices and stacks them together to construct a three-dimensional (3D) alignment of the tissue.

## Overview
* [Requirements](#Requirements)
* [Installation](#Installation)
* Tutorial
    * [Spoint tutorial: Deconvolution of cell types distribution of spatial transcriptomics](tutorial/deconvolution_of_cell_types_distribution.ipynb)
    * [Splane tutorial: Identify uniform spatial domain in multiple slices](tutorial/identification_of_uniform_spatial_domain.ipynb)
    * [Scube tutorial: Alignment of multiple spatial transcriptomic slices](tutorial/alignment_of_multiple_slices.ipynb)
    * [Scube tutorial: 3D expression modeling with gaussian process regression](tutorial/3D_expression_modeling.ipynb)
    
## Requirements

To install `SPACEL`, you need install [TensorFlow](https://www.tensorflow.org/) with GPU version first. 
* Create conda environment for `SPACEL`:
```
conda env create -f environment.yml
```
or
```
conda create -n SPACEL -c conda-forge -c default cudnn=7.6 cudatoolkit=10.1 tensorflow==2.3.1 python==3.8 r-base r-fitdistrplus
```
you can choose correct `tensorflow`, `cudnn` and `cudatoolkit` version dependent on your graphic driver version. If you don't need GPU acceleration, you can just skip the installation for `cudnn` and `cudatoolkit`.
* Test if [TensorFlow](https://www.tensorflow.org/) for GPU available:
```
python
>>> import tensorflow as tf
>>> tf.test.gpu_device_name()
```
If these command line have not return available gpu names, please check your gpu driver version. For more detail, look at [CUDA Toolkit Major Component Versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions) and [TensorFlow Build Configurations](https://www.tensorflow.org/install/source#tested_build_configurations)

## Installation
* Install `SPACEL`:
```
pip install -r requirements.txt
python setup.py install
```
