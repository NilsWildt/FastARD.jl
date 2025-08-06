# API Reference

This page provides complete documentation for all exported functions and types in FastARD.jl.

## Contents

```@contents
Pages = ["95-reference.md"]
Depth = 3
```

## Index

```@index
Pages = ["95-reference.md"]
```

## Core Types

### FastARDRegressor

```@docs
FastARDRegressor
FastARDRegressor(::Type{<:Real})
```

## Main Functions

### Model Fitting

```@docs
fit!
```

### Prediction Functions

```@docs
predict
predict_with_uncertainty
get_active_coefficients
```

## Utility Functions

```@docs
only_finite
```

## Internal Types

For advanced users and developers:

```@docs
PrecomputedCache
```

## Method Overview

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `FastARDRegressor()` | Create model | Parameters | Model instance |
| `fit!(model, X, y)` | Train model | Data matrices | Fitted model |
| `predict(model, X)` | Make predictions | Test data | Predictions |
| `predict_with_uncertainty(model, X)` | Predictions + uncertainty | Test data | Predictions + std |
| `get_active_coefficients(model)` | Get selected features | Fitted model | Indices + coefficients |

## Model Properties

After fitting, the `FastARDRegressor` object contains:

### Estimated Parameters
- **`coef`**: Coefficient vector for all features
- **`alpha`**: Precision parameters (inverse variance) for each feature  
- **`beta`**: Noise precision (inverse noise variance)
- **`active`**: Boolean mask indicating selected features
- **`sigma`**: Posterior covariance matrix for active features

### Training Information
- **`converged`**: Whether the algorithm converged
- **`scores`**: Log marginal likelihood history (if `compute_score=true`)

### Hyperparameters
- **`n_iter`**: Maximum iterations
- **`tol`**: Convergence tolerance
- **`lambda_reg`**: Regularization parameter
- **`min_beta`**: Minimum noise precision
- **`max_alpha`**: Maximum feature precision

## Algorithm Details

FastARD implements the Sequential Sparse Bayes algorithm:

1. **Initialize** with all features inactive
2. **Select** the most promising feature to add
3. **Update** precision parameters α and β
4. **Recompute** posterior mean and variance
5. **Repeat** until convergence

### Mathematical Framework

The model assumes:
```math
p(\mathbf{c}|\boldsymbol{\alpha}) = \prod_{i=1}^N \mathcal{N}(c_i | 0, \alpha_i^{-1})
```
```math
p(\mathbf{y}|\mathbf{c}, \beta) = \mathcal{N}(\mathbf{y} | \mathbf{\Psi}\mathbf{c}, \beta^{-1}\mathbf{I})
```

where:
- **𝐜** is the coefficient vector
- **𝛂** are the precision parameters  
- **𝛃** is the noise precision
- **𝚿** is the design matrix

## Performance Characteristics

### Computational Complexity
- **Time**: O(M²N) per iteration, where M is number of active features and N is number of samples
- **Space**: O(N²) for precomputed matrices
- **Iterations**: Typically 10-200 depending on problem difficulty

### Scalability Guidelines
- **Samples (N)**: Scales well up to ~10,000 samples
- **Features (P)**: Can handle P >> N (high-dimensional problems)
- **Active Features (M)**: Most efficient when M < 100

### Numerical Considerations
- Uses Cholesky decomposition for efficiency
- Falls back to pseudo-inverse for ill-conditioned problems
- Regularization prevents numerical instabilities
- Supports arbitrary precision arithmetic via MultiFloats.jl

## Complete Function Documentation

```@autodocs
Modules = [FastARD]
```
