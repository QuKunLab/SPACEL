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
    * {doc}`Spoint tutorial: Deconvolution of cell types distribution of spatial transcriptomics <tutorials/deconvolution_of_cell_types_distribution>`
    * {doc}`Splane tutorial: Identify uniform spatial domain in multiple slices <tutorials/identification_of_uniform_spatial_domain>`
    * {doc}`Scube tutorial: Alignment of multiple spatial transcriptomic slices <tutorials/alignment_of_multiple_slices>`
    * {doc}`Scube tutorial: 3D expression modeling with gaussian process regression <tutorials/3D_expression_modeling>`
* {doc}`API <api>`

```{toctree}
:hidden: true
:maxdepth: 1

installation
tutorials
api
```
