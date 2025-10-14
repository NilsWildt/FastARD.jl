```@meta
CurrentModule = FastARD
```

# FastARD

Documentation for [FastARD](https://github.com/NilsWildt/FastARD.jl).

FastARD.jl provides a Julia implementation of Fast Automatic Relevance Determination (ARD) for sparse Bayesian regression with uncertainty quantification.

## Overview

FastARD automatically identifies relevant features in high-dimensional regression problems while providing well-calibrated uncertainty estimates. The package is particularly well-suited for:

- **Polynomial chaos expansion** with automatic basis selection
- **Function approximation** with sparse representations  
- **Uncertainty quantification** in engineering and scientific applications
- **Method comparison** against traditional numerical approaches

## Key Features

- Automatic sparsity detection in overcomplete bases
- Bayesian uncertainty quantification with calibrated confidence intervals
- Polynomial chaos expansion support for nonlinear function approximation
- Performance optimization with convergence monitoring
- Comprehensive comparison against advanced numerical methods

## Tested Applications

The package includes thorough testing on the **Ishigami function**, a standard benchmark for sensitivity analysis and uncertainty quantification that demonstrates:

- Automatic selection of relevant polynomial terms from 35 basis functions
- Well-calibrated uncertainty estimates with proper coverage probabilities
- Superior performance compared to pseudoinverse and other advanced methods
- Efficient compression achieving ~80% sparsity while maintaining accuracy

See the [Examples](03-examples.md) page for the complete tested implementation.

## Contributors

```@raw html
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
```
