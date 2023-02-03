[![Documentation Status](https://readthedocs.org/projects/spacel/badge/?version=latest)](https://spacel.readthedocs.io/en/latest/?badge=latest)

# SPACEL: characterizing spatial transcriptome architectures by deep-learning

![Overview](docs/_static/img/figure1.png "Overview")
SPACEL (**SP**atial **A**rchitecture **C**haracterization by d**E**ep **L**earning) is a Python package of deep-learning-based methods for ST data analysis. SPACEL consists of three modules: 
* Spoint embedded a multiple-layer perceptron with a probabilistic model to deconvolute cell type composition for each spot on single ST slice.
* Splane employs a graph convolutional network approach and an adversarial learning algorithm to identify uniform spatial domains that are transcriptomically and spatially coherent across multiple ST slices.
* Scube automatically transforms the spatial coordinate systems of consecutive slices and stacks them together to construct a three-dimensional (3D) alignment of the tissue.

## Getting started
* [Requirements](#Requirements)
* [Installation](#Installation)
* Tutorials
    * [Spoint tutorial: Deconvolution of cell types distribution of spatial transcriptomics](docs/tutorials/deconvolution_of_cell_types_distribution.ipynb)
    * [Splane tutorial: Identify uniform spatial domain in multiple slices](docs/tutorials/identification_of_uniform_spatial_domain.ipynb)
    * [Scube tutorial: Alignment of multiple spatial transcriptomic slices](docs/tutorials/alignment_of_multiple_slices.ipynb)
    * [Scube tutorial: 3D expression modeling with gaussian process regression](docs/tutorials/3D_expression_modeling.ipynb)

Read the [documentation](https://spacel.readthedocs.io) for more information.
    
## Requirements

To install `SPACEL`, you need to install [TensorFlow](https://www.tensorflow.org/) with GPU support first. If you don't need GPU acceleration, you can just skip the installation for `cudnn` and `cudatoolkit`.
* Create conda environment for `SPACEL`:
```
conda env create -f environment.yml
```
or
```
conda create -n SPACEL -c conda-forge -c default cudnn=7.6 cudatoolkit=10.1 python=3.8 r-base r-fitdistrplus
```
You must choose correct `tensorflow`, `cudnn` and `cudatoolkit` version dependent on your graphic driver version. 
* Test if [TensorFlow](https://www.tensorflow.org/) for GPU available:
```
python
>>> import tensorflow as tf
>>> tf.test.gpu_device_name()
```
If these command line have not return available gpu names, please check your gpu driver version. For more detail, look at [CUDA Toolkit Major Component Versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions) and [TensorFlow Build Configurations](https://www.tensorflow.org/install/source#tested_build_configurations).

Note: If you want to run 3D expression GPR model in Scube, you need to install the [Open3D](http://www.open3d.org/docs/release/) python library first.

## Installation
* Install `SPACEL`:
```
pip install -r requirements.txt
python setup.py install
```
