[![Documentation Status](https://readthedocs.org/projects/spacel/badge/?version=latest)](https://spacel.readthedocs.io/en/latest/?badge=latest)
![PyPI](https://img.shields.io/pypi/v/SPACEL)

# SPACEL: characterizing spatial transcriptome architectures by deep-learning

```{image} _static/img/figure1.png
:width: 900px
```
SPACEL (**SP**atial **A**rchitecture **C**haracterization by d**E**ep **L**earning) is a Python package of deep-learning-based methods for ST data analysis. SPACEL consists of three modules: 

- Spoint embedded a multiple-layer perceptron with a probabilistic model to deconvolute cell type composition for each spot on single ST slice.
- Splane employs a graph convolutional network approach and an adversarial learning algorithm to identify uniform spatial domains that are transcriptomically and spatially coherent across multiple ST slices.
- Scube automatically transforms the spatial coordinate systems of consecutive slices and stacks them together to construct a three-dimensional (3D) alignment of the tissue.

## Content
* {doc}`Installation <installation>`
* {doc}`Tutorials <tutorials>`
    * {doc}`Spoint tutorial: Deconvolution of cell types compostion on human brain Visium dataset <tutorials/Visium_human_DLPFC_Spoint>`
    * {doc}`Splane tutorial: Identify uniform spatial domain on human breast cancer Visium dataset <tutorials/Visium_human_breast_cancer_Splane>`
    * {doc}`Splane&Scube tutorial (1/2): Identify uniform spatial domain on human brain MERFISH dataset <tutorials/MERFISH_mouse_brain_Splane>`
    * {doc}`Splane&Scube tutorial (1/2): Alignment of consecutive ST slices on human brain MERFISH dataset <tutorials/MERFISH_mouse_brain_Scube>`
    * {doc}`Scube tutorial: Alignment of consecutive ST slices on mouse embryo Stereo-seq dataset <tutorials/Stereo-seq_Scube>`
    * {doc}`Scube tutorial: 3D expression modeling with gaussian process regression <tutorials/STARmap_mouse_brain_GPR>`
    * {doc}`SPACEL workflow (1/3): Deconvolution by Spoint on mouse brain ST dataset <tutorials/ST_mouse_brain_Spoint>`
    * {doc}`SPACEL workflow (2/3): Identification of spatial domain by Splane on mouse brain ST dataset <tutorials/ST_mouse_brain_Splane>`
    * {doc}`SPACEL workflow (3/3): Alignment 3D tissue by Scube on mouse brain ST dataset <tutorials/ST_mouse_brain_Scube>`
* {doc}`API <api>`

## Latest updates
### Version 1.1.7 2024-01-16
#### Fixed Bugs
- Fixed a variable reference error in function `identify_spatial_domain`. Thanks to @tobias-zehnde for the contribution.

### Version 1.1.6 2023-07-27
#### Fixed Bugs
- Fixed a bug regarding the similarity loss weight hyperparameter `simi_l`, which in the previous version did not affect the loss value.

### Version 1.1.5 2023-07-26
#### Fixed Bugs
- Fixed a bug in the similarity loss of Splane, where it minimized the cosine similarity of the latent vectors of spots with their neighbors.
#### Features
- Optimized the time and memory consumption of the Splane training process for large datasets.

```{toctree}
:hidden: true
:maxdepth: 1

installation
tutorials
api
```
