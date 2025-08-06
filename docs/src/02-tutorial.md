# Tutorial

This tutorial provides a comprehensive guide to using FastARD.jl for various sparse regression scenarios. We'll cover different types of problems and show how to interpret the results.

## Basic Sparse Regression

Let's start with a controlled example where we know the true sparsity pattern.

```julia
using FastARD
using Random, Statistics, LinearAlgebra

Random.seed!(123)

# Create a sparse regression problem
n_samples, n_features = 200, 100
X = randn(n_samples, n_features)

# True sparse coefficient vector
true_coef = zeros(n_features)
active_features = [5, 12, 23, 45, 67, 89]  # 6 out of 100 features are active
true_coef[active_features] = [2.5, -1.8, 3.2, -0.9, 1.6, -2.1]

# Generate observations
noise_std = 0.2
y = X * true_coef + noise_std * randn(n_samples)

println("True number of active features: ", length(active_features))
println("Signal-to-noise ratio: ", std(X * true_coef) / noise_std)
```

### Fitting the Model

```julia
# Create and fit the model
model = FastARDRegressor(verbose=true, compute_score=true)
fit!(model, X, y)

# Analyze the results
active_indices, active_coefs = get_active_coefficients(model)
println("\nSelected features: ", active_indices)
println("True active features: ", active_features)

# Check feature selection accuracy
correctly_selected = length(intersect(active_indices, active_features))
precision = correctly_selected / length(active_indices)
recall = correctly_selected / length(active_features)

println("Precision: $precision")
println("Recall: $recall")
```

### Visualizing Convergence

If you computed scores, you can visualize the convergence:

```julia
using Plots

if !isempty(model.scores)
    plot(model.scores, xlabel="Iteration", ylabel="Log Marginal Likelihood", 
         title="Convergence", linewidth=2)
end
```

## High-Dimensional Regression (n < p)

FastARD excels in high-dimensional settings where the number of features exceeds the number of samples.

```julia
# High-dimensional problem
n_samples, n_features = 50, 500  # More features than samples
X = randn(n_samples, n_features)

# Very sparse solution
true_coef = zeros(n_features)
true_coef[1:3] = [5.0, -3.0, 4.0]  # Only 3 active features
y = X * true_coef + 0.1 * randn(n_samples)

# Fit the model
model = FastARDRegressor(n_iter=300, verbose=false)
fit!(model, X, y)

# Check results
active_indices, active_coefs = get_active_coefficients(model)
println("Selected $(length(active_indices)) features out of $n_features")
println("Active features: ", active_indices)
println("True active features: [1, 2, 3]")

# Prediction accuracy
y_pred = predict(model, X)
println("RMSE: ", sqrt(mean((y_pred .- y).^2)))
```

## Uncertainty Quantification

One of ARD's key advantages is providing uncertainty estimates for predictions.

```julia
# Generate test data
X_test = randn(50, n_features)
y_test_true = X_test * true_coef

# Get predictions with uncertainty
y_pred, y_std = predict_with_uncertainty(model, X_test)

# Analyze uncertainty quality
residuals = abs.(y_pred .- y_test_true)
println("Mean prediction uncertainty: ", mean(y_std))
println("Mean absolute error: ", mean(residuals))

# Check if uncertainties are well-calibrated
# Points within 1σ should contain ~68% of true values
within_1sigma = sum(residuals .<= y_std) / length(residuals)
println("Fraction within 1σ: $within_1sigma (should be ~0.68)")
```

## Handling Multicollinearity

FastARD automatically handles multicollinear features by selecting a representative subset.

```julia
# Create multicollinear features
n_samples, n_base_features = 100, 20
X_base = randn(n_samples, n_base_features)

# Add correlated copies of some features
X_corr1 = X_base[:, 1:5] + 0.1 * randn(n_samples, 5)  # Noisy copies
X_corr2 = 2.0 * X_base[:, 1:5]  # Scaled copies

X = [X_base X_corr1 X_corr2]  # 20 + 5 + 5 = 30 features
n_features = size(X, 2)

# True coefficients only on original features
true_coef = zeros(n_features)
true_coef[1:5] = [1.0, -1.5, 2.0, -0.5, 1.8]
y = X * true_coef + 0.1 * randn(n_samples)

# Fit model
model = FastARDRegressor(verbose=false)
fit!(model, X, y)

active_indices, active_coefs = get_active_coefficients(model)
println("Selected features: ", active_indices)
println("Features 1-5: original")
println("Features 21-25: noisy copies of 1-5") 
println("Features 26-30: scaled copies of 1-5")
```

## Multiple Output Regression

FastARD can handle multiple output variables by fitting separate models for each output.

```julia
# Multiple output problem
n_outputs = 3
Y = randn(n_samples, n_outputs)

# Each output has different sparsity pattern
for i in 1:n_outputs
    true_coef = zeros(n_features)
    active_features = rand(1:n_features, 3+i)  # Different number of active features
    true_coef[active_features] = randn(length(active_features))
    Y[:, i] = X * true_coef + 0.1 * randn(n_samples)
end

# Fit models (returns vector of models)
models = fit!(deepcopy(model), X, Y)

# Analyze each output
for i in 1:n_outputs
    active_indices, _ = get_active_coefficients(models[i])
    println("Output $i: $(length(active_indices)) active features")
end
```

## Working with Different Precision Types

FastARD supports different numerical precisions for enhanced accuracy or performance.

```julia
using MultiFloats

# High precision computation
model_hp = FastARDRegressor(MultiFloat{Float64,4}, verbose=false)
fit!(model_hp, X, y[:, 1])

# Single precision for speed
model_sp = FastARDRegressor(Float32, verbose=false)  
fit!(model_sp, Float32.(X), Float32.(y[:, 1]))

# Compare results
active_hp, coef_hp = get_active_coefficients(model_hp)
active_sp, coef_sp = get_active_coefficients(model_sp)

println("High precision active features: ", active_hp)
println("Single precision active features: ", active_sp)
```

## Parameter Tuning Guidelines

While ARD is largely automatic, some parameters may need adjustment:

### Convergence Parameters

```julia
# For difficult problems, increase iterations and decrease tolerance
model_strict = FastARDRegressor(
    n_iter=500,      # More iterations
    tol=1e-8,        # Stricter convergence
    verbose=true
)
```

### Numerical Stability

```julia
# For ill-conditioned problems
model_stable = FastARDRegressor(
    lambda_reg=1e-6,  # More regularization
    min_beta=1e-8,    # Lower minimum noise precision
    max_alpha=1e10    # Lower maximum feature precision
)
```

### Performance vs. Accuracy Trade-offs

```julia
# Fast but less accurate
model_fast = FastARDRegressor(n_iter=50, tol=1e-4)

# Slow but more accurate  
model_accurate = FastARDRegressor(n_iter=1000, tol=1e-10)
```

## Interpreting Results

### Feature Importance

```julia
# Features with smaller alpha values are more important
active_indices, active_coefs = get_active_coefficients(model)
alpha_active = model.alpha[active_indices]

# Sort by importance (inverse of alpha)
importance_order = sortperm(alpha_active)
println("Features by importance:")
for i in importance_order
    feat_idx = active_indices[i]
    println("Feature $feat_idx: coef=$(active_coefs[i]), α=$(alpha_active[i])")
end
```

### Model Quality Metrics

```julia
# Effective number of parameters (for model comparison)
effective_params = sum(model.active)

# Prediction accuracy
y_pred = predict(model, X)
rmse = sqrt(mean((y_pred .- y).^2))
r2 = 1 - sum((y .- y_pred).^2) / sum((y .- mean(y)).^2)

println("Effective parameters: $effective_params")
println("RMSE: $rmse")
println("R²: $r2")

# Log marginal likelihood (if computed)
if !isempty(model.scores)
    println("Final log marginal likelihood: ", model.scores[end])
end
```

## Best Practices

1. **Data Preprocessing**: Standardize features for better numerical stability
2. **Cross-Validation**: Use cross-validation to assess generalization
3. **Feature Engineering**: ARD works well with polynomial and interaction features
4. **Convergence Monitoring**: Always check convergence, especially for difficult problems
5. **Uncertainty Validation**: Validate uncertainty estimates on held-out data

## Next Steps

- See [Examples](03-examples.md) for real-world applications
- Check the [API Reference](95-reference.md) for detailed function documentation