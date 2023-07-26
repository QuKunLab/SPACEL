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
### Version 1.1.5 2023-07-26
#### Fixed Bugs
- Fixed a bug in the similarity loss of Splane, where it minimized the cosine similarity of the latent vectors of spots with their neighbors.
#### Features
- Optimized the time and memory consumption of the Splane training process for large datasets.

### Version 1.1.2 2023-07-12
#### Fixed Bugs
- Removed `rpy2` from the pypi dependency of SPACEL. It now needs to be pre-installed when creating the environment through conda.
- Fixed a bug in Scube where the `best_model_state` was not referenced before being used.
#### Features
- Added function documentations for Scube related to the GPR model.

### Version 1.1.1 2023-07-11
#### Features
* All code based on `Tensorflow` have been mirated to `PyTorch`, it does not have `Tensorflow` as dependency anymore.
* The `Splane.utils.add_cell_type_composition` function has been implemented to facilitate the cell type composition predicted by deconvolution methods into Splane.
* Spoint and Splane now support tqdm type output for improved progress tracking.

```{toctree}
:hidden: true
:maxdepth: 1

installation
tutorials
api
```
