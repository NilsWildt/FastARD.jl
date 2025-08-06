# Getting Started

This page will guide you through installing and setting up FastARD.jl for your first sparse Bayesian regression analysis.

## Installation

FastARD.jl is registered in the Julia General registry and can be installed using the built-in package manager:

```julia
using Pkg
Pkg.add("FastARD")
```

For the development version, you can install directly from GitHub:

```julia
Pkg.add(url="https://github.com/NilsWildt/FastARD.jl")
```

## Loading the Package

```julia
using FastARD
```

## Your First Model

Let's start with a simple example to demonstrate the basic workflow:

### 1. Generate Test Data

```julia
using Random
Random.seed!(42)

# Problem dimensions
n_samples = 100
n_features = 50

# Generate random features
X = randn(n_samples, n_features)

# Create a sparse true coefficient vector (only 5 out of 50 features are active)
true_coef = zeros(n_features)
true_coef[1:5] .= [2.0, -1.5, 3.0, -0.8, 1.2]

# Generate observations with noise
noise_level = 0.1
y = X * true_coef + noise_level * randn(n_samples)
```

### 2. Create and Configure the Model

```julia
# Create a model with default settings
model = FastARDRegressor()

# Or customize the parameters
model = FastARDRegressor(
    n_iter=200,        # Maximum iterations
    tol=1e-6,          # Convergence tolerance  
    verbose=true,      # Print progress
    compute_score=true # Track log marginal likelihood
)
```

### 3. Fit the Model

```julia
# Fit the model to the data
fit!(model, X, y)
```

The algorithm will automatically:
- Select relevant features
- Estimate coefficients
- Determine noise level
- Provide uncertainty estimates

### 4. Make Predictions

```julia
# Basic predictions
y_pred = predict(model, X)

# Predictions with uncertainty estimates
y_pred, y_std = predict_with_uncertainty(model, X)

# Calculate prediction accuracy
using Statistics
rmse = sqrt(mean((y_pred .- y).^2))
println("RMSE: $rmse")
```

### 5. Analyze Results

```julia
# Get the selected features
active_indices, active_coefs = get_active_coefficients(model)
println("Selected features: ", active_indices)
println("Coefficients: ", active_coefs)

# Compare with true active features
true_active = findall(abs.(true_coef) .> 1e-10)
println("True active features: ", true_active)

# Check model properties
println("Number of active features: ", sum(model.active))
println("Noise precision (β): ", model.beta)
println("Converged: ", model.converged)
```

## Key Concepts

### Model Parameters

- **`coef`**: Estimated coefficients for all features
- **`alpha`**: Precision parameters for each feature (higher = less relevant)
- **`beta`**: Noise precision (inverse variance)
- **`active`**: Boolean mask indicating which features are selected
- **`sigma`**: Posterior covariance matrix for active features

### Hyperparameters

- **`n_iter`**: Maximum number of iterations (default: 200)
- **`tol`**: Convergence tolerance (default: 1e-6)
- **`lambda_reg`**: Regularization for numerical stability (default: 1e-8)
- **`min_beta`**: Minimum noise precision (default: 1e-6)
- **`max_alpha`**: Maximum feature precision (default: 1e12)

### Convergence

The algorithm converges when:
1. No more features are added or removed
2. The noise precision β stabilizes
3. The maximum number of iterations is reached

## Next Steps

- Check out the [Tutorial](02-tutorial.md) for more detailed examples
- See [Examples](03-examples.md) for practical applications  
- Refer to the [API Reference](95-reference.md) for complete documentation