# BundleMethod.jl
![Run tests](https://github.com/kibaekkim/BundleMethod.jl/workflows/Run%20tests/badge.svg)
[![codecov](https://codecov.io/gh/kibaekkim/BundleMethod.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kibaekkim/BundleMethod.jl)
[![DOI](https://zenodo.org/badge/170216265.svg)](https://zenodo.org/badge/latestdoi/170216265)

This package provides a template for implementing bundle methods.
The template is generic and does not assume any particular structure.
Using the template, this pacakge implements 

- a proximal bundle method with linear constraints
- a trust region method with linear constraints

## Installation

```
] add BundleMethod
```

## Examples

Please see examples in `./examples`.

## Bibtex

```
@misc{DualDecomposition.jl.0.2.1,
  author       = {Kim, Kibaek and Zhang, Weiqi and Nakao, Hideaki and Schanen, Michel},
  title        = {{BundleMethod.jl: Implementation of Bundle Methods in Julia}},
  month        = Mar,
  year         = 2021,
  doi          = {10.5281/zenodo.4574897},
  version      = {0.3.2},
  publisher    = {Zenodo},
  url          = {https://doi.org/10.5281/zenodo.4574897}
}
```

## Acknowledgements
This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
