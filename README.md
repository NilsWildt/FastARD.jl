# FastARD

[![CI](https://github.com/NilsWildt/FastARD.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/NilsWildt/FastARD.jl/actions/workflows/CI.yml)
[![All Contributors](https://img.shields.io/github/all-contributors/NilsWildt/FastARD.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

A Julia implementation of Fast Automatic Relevance Determination (ARD) for sparse Bayesian regression with uncertainty quantification.

## Key Features

- **Automatic sparsity detection** in high-dimensional regression problems
- **Uncertainty quantification** with well-calibrated confidence intervals
- **Polynomial chaos expansion** support for function approximation
- **Method comparison** against traditional numerical approaches
- **Performance optimization** with timing analysis and convergence monitoring

## Quick Example

```julia
using FastARD
using Random, Statistics, LinearAlgebra

# Test on the Ishigami function
function ishigami(x; a=7.0, b=0.1)
    x1, x2, x3 = x[1], x[2], x[3]
    return sin(x1) + a * sin(x2)^2 + b * x3^4 * sin(x1)
end

# Generate test data
Random.seed!(123)
n_train = 300
X_train = 2π * rand(n_train, 3) .- π
y_train = [ishigami(X_train[i, :]) for i in 1:n_train]

# Create polynomial basis (35 terms for degree 4)
# ... (see examples for full implementation)

# Fit FastARD model
model = FastARDRegressor(verbose=true, compute_score=true)
fit!(model, Psi_train, y_train_noisy)

# Analyze sparsity
active_indices, active_coefs = get_active_coefficients(model)
println("Selected $(length(active_indices)) out of 35 basis functions")

# Get predictions with uncertainty
y_pred, y_std = predict_with_uncertainty(model, Psi_test)
```

## Performance Tips

- **Use a fast BLAS backend.** The fitting loop is dense linear algebra. On Apple
  silicon, `using AppleAccelerate` before `using FastARD` activates the bundled
  package extension and swaps the BLAS backend (~3.5× faster on large problems);
  on Intel CPUs use `using MKL` (AMD users may also like `BLISBLAS.jl`). Add the
  backend package to your environment — FastARD only declares them as optional
  (weak) dependencies. To make this automatic on a given machine, put the
  `using` line in `~/.julia/config/startup.jl`; Julia loads it in every session,
  so the right backend for that CPU is always active. (Swapping BLAS is a
  process-global effect, which is why FastARD leaves the choice to you instead
  of forcing it.)
- **Large active sets:** if fits select many basis functions, the per-iteration
  statistics refresh dominates. `FastARDRegressor(beta_recompute_tol=1e-3)`
  throttles it for a ~5–15× speedup with an essentially unchanged model
  (the default `1e-6` matches the reference algorithm exactly).

## Tested Applications

The package includes comprehensive tests demonstrating performance on the **Ishigami function**, a standard benchmark for:
- Sensitivity analysis and uncertainty quantification
- Polynomial chaos expansion with automatic basis selection
- Comparison against multiple advanced numerical methods
- Uncertainty calibration and coverage probability analysis

See `examples/ishigami_comparison_test.jl` for the complete tested implementation.

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
