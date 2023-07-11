# Installation

## Requirements

To install `SPACEL`, you need to install [PyTorch](https://pytorch.org) with GPU support first. If you don't need GPU acceleration, you can just skip the installation for `cudnn` and `cudatoolkit`.
* Create conda environment for `SPACEL`:
```
conda env create -f environment.yml
```
or
```
conda create -n SPACEL -c conda-forge -c default cudatoolkit=10.2 python=3.8 r-base r-fitdistrplus
```
You must choose correct `PyTorch`, `cudnn` and `cudatoolkit` version dependent on your graphic driver version. 
* Test if [PyTorch](https://pytorch.org) for GPU available:
```
python
>>> import torch
>>> torch.cuda.is_available()
```
If these command line have not return `True`, please check your gpu driver version and `cudatoolkit` version. For more detail, look at [CUDA Toolkit Major Component Versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions).

Note: If you want to run 3D expression GPR model in Scube, you need to install the [Open3D](http://www.open3d.org/docs/release/) python library first.

## Installation
* Install `SPACEL`:
```
pip install SPACEL
```
