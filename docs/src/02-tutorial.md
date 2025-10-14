# Tutorial

This tutorial provides a comprehensive guide to using FastARD.jl for sparse regression with uncertainty quantification, based on the tested Ishigami function example.

## Basic Polynomial Chaos Expansion

Let's start with a controlled example using polynomial chaos expansion (PCE) for function approximation with automatic sparsity detection.

```julia
using FastARD
using Random, Statistics, LinearAlgebra

Random.seed!(123)

# Define the Ishigami function - a standard test function for uncertainty quantification
function ishigami(x::AbstractVector; a=7.0, b=0.1)
    x1, x2, x3 = x[1], x[2], x[3]
    return sin(x1) + a * sin(x2)^2 + b * x3^4 * sin(x1)
end

# Generate training data
n_train = 300
X_train = 2π * rand(n_train, 3) .- π  # Sample from [-π, π]³
y_train = [ishigami(X_train[i, :]) for i in 1:n_train]
noise_std = 0.5
y_train_noisy = y_train + noise_std * randn(n_train)

println("Training samples: $n_train")
println("Signal-to-noise ratio: $(std(y_train) / noise_std)")
```

### Generating Polynomial Basis Functions

```julia
# Generate Legendre polynomial basis
function legendre_polynomials(x, max_degree)
    n = length(x)
    polys = zeros(n, max_degree + 1)
    
    # P0(x) = 1, P1(x) = x
    polys[:, 1] .= 1.0
    if max_degree >= 1
        polys[:, 2] = x
    end
    
    # Recurrence relation: (n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}
    for i in 2:max_degree
        polys[:, i+1] = ((2*i-1) * x .* polys[:, i] - (i-1) * polys[:, i-1]) / i
    end
    
    return polys
end

# Generate multivariate polynomial basis
function generate_pce_basis(X, max_degree=3)
    n_samples, n_dims = size(X)
    X_normalized = X / π  # Transform to [-1,1]
    
    # Generate univariate polynomials for each dimension
    polys = [legendre_polynomials(X_normalized[:, i], max_degree) for i in 1:n_dims]
    
    # Create multivariate basis using tensor products
    basis_terms = []
    for i in 0:max_degree
        for j in 0:max_degree
            for k in 0:max_degree
                if i + j + k <= max_degree
                    term = polys[1][:, i+1] .* polys[2][:, j+1] .* polys[3][:, k+1]
                    push!(basis_terms, term)
                end
            end
        end
    end
    
    return hcat(basis_terms...)
end

max_degree = 4
Psi_train = generate_pce_basis(X_train, max_degree)
n_basis = size(Psi_train, 2)
println("Generated $n_basis polynomial basis functions")
```

### Fitting the FastARD Model

```julia
# Create and fit the model
model = FastARDRegressor(verbose=true, compute_score=true, n_iter=200)
fit!(model, Psi_train, y_train_noisy)

# Analyze the results
active_indices, active_coefs = get_active_coefficients(model)
println("Selected $(length(active_indices)) basis functions out of $n_basis")
println("Active indices: $active_indices")
println("Compression ratio: $(round(length(active_indices)/n_basis, digits=3))")
```

### Visualizing Convergence

If you computed scores, you can visualize the convergence:

```julia
using CairoMakie

if !isempty(model.scores)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Iteration", ylabel="Log Marginal Likelihood", 
              title="FastARD Convergence")
    scatterlines!(ax, 1:length(model.scores), model.scores, linewidth=2, markersize=6)
    save("convergence.pdf", fig)
end
```

## Uncertainty Quantification

One of ARD's key advantages is providing uncertainty estimates for predictions.

```julia
# Generate test data
n_test = 100
X_test = 2π * rand(n_test, 3) .- π
y_test_true = [ishigami(X_test[i, :]) for i in 1:n_test]
Psi_test = generate_pce_basis(X_test, max_degree)

# Get predictions with uncertainty
y_pred, y_std = predict_with_uncertainty(model, Psi_test)

# Analyze uncertainty quality
residuals = abs.(y_pred .- y_test_true)
println("Mean prediction uncertainty: ", mean(y_std))
println("Mean absolute error: ", mean(residuals))

# Check if uncertainties are well-calibrated
# Points within 1σ should contain ~68% of true values
within_1sigma = sum(residuals .<= y_std) / length(residuals)
within_2sigma = sum(residuals .<= 2 .* y_std) / length(residuals)

println("Fraction within 1σ: $(round(within_1sigma, digits=3)) (should be ~0.68)")
println("Fraction within 2σ: $(round(within_2sigma, digits=3)) (should be ~0.95)")
```

## Method Comparison

FastARD can be compared against traditional methods like pseudoinverse regression.

```julia
# Compare with pseudoinverse method
coef_pinv = pinv(Psi_train) * y_train_noisy
y_pred_pinv = Psi_test * coef_pinv

# Calculate metrics
rmse_ard = sqrt(mean((y_pred .- y_test_true).^2))
rmse_pinv = sqrt(mean((y_pred_pinv .- y_test_true).^2))

effective_rank_ard = sum(model.active)
effective_rank_pinv = n_basis
compression_ratio = effective_rank_ard / effective_rank_pinv

println("FastARD RMSE: $rmse_ard")
println("Pinv RMSE: $rmse_pinv")
println("FastARD uses $(effective_rank_ard)/$n_basis features")
println("Pinv uses all $effective_rank_pinv features")
println("Compression ratio: $(round(compression_ratio, digits=3))")
```

## Working with Different Precision Types

FastARD supports different numerical precisions for enhanced accuracy or performance.

```julia
using MultiFloats

# High precision computation
model_hp = FastARDRegressor(MultiFloat{Float64,4}, verbose=false)
fit!(model_hp, Psi_train, y_train_noisy)

# Single precision for speed
model_sp = FastARDRegressor(Float32, verbose=false)  
fit!(model_sp, Float32.(Psi_train), Float32.(y_train_noisy))

# Compare results
active_hp, coef_hp = get_active_coefficients(model_hp)
active_sp, coef_sp = get_active_coefficients(model_sp)

println("High precision active features: ", active_hp)
println("Single precision active features: ", active_sp)
```

## Parameter Tuning Guidelines

While ARD is largely automatic, some parameters may need adjustment based on your problem characteristics:

### Convergence Parameters

```julia
# For difficult problems, increase iterations and decrease tolerance
model_strict = FastARDRegressor(
    n_iter=500,      # More iterations for complex problems
    tol=1e-8,        # Stricter convergence criterion
    verbose=true     # Monitor progress
)
```

### Numerical Stability

```julia
# For ill-conditioned polynomial basis matrices
model_stable = FastARDRegressor(
    lambda_reg=1e-6,  # Small regularization for stability
    verbose=true
)
```

## Interpreting Results

### Feature Importance in Polynomial Basis

```julia
# Features with smaller alpha values are more important
active_indices, active_coefs = get_active_coefficients(model)
alpha_active = model.alpha[active_indices]

# Sort by importance (inverse of alpha)
importance_order = sortperm(alpha_active)
println("Most important polynomial terms:")
for i in importance_order[1:min(5, length(importance_order))]
    feat_idx = active_indices[i]
    println("Basis $feat_idx: coef=$(round(active_coefs[i], digits=4)), α=$(round(alpha_active[i], digits=2))")
end
```

### Model Quality Metrics

```julia
# Effective number of parameters (for model comparison)
effective_params = sum(model.active)

# Prediction accuracy on training data
y_pred_train = predict(model, Psi_train)
rmse_train = sqrt(mean((y_pred_train .- y_train_noisy).^2))

println("Effective parameters: $effective_params out of $n_basis")
println("Training RMSE: $rmse_train")
println("Compression achieved: $(round((1 - effective_params/n_basis)*100, digits=1))%")

# Log marginal likelihood (if computed)
if !isempty(model.scores)
    println("Final log marginal likelihood: ", round(model.scores[end], digits=2))
end
```

## Best Practices for Polynomial Chaos Expansion

1. **Basis Selection**: Use orthogonal polynomials (Legendre, Hermite) for better numerical conditioning
2. **Degree Selection**: Start with moderate polynomial degrees (3-5) and increase if needed
3. **Domain Transformation**: Always transform input domains to the standard range [-1,1] or [0,1]
4. **Convergence Monitoring**: Enable `verbose=true` and `compute_score=true` for monitoring
5. **Uncertainty Validation**: Validate uncertainty estimates using coverage probability tests

## Next Steps

- See [Examples](03-examples.md) for the complete Ishigami function test
- Check the [API Reference](95-reference.md) for detailed function documentation