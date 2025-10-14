# Examples

This page showcases practical applications of FastARD.jl across different domains and problem types.

## Ishigami Function Comparison Test

This example demonstrates FastARD's performance on the Ishigami function, a popular test function for sensitivity analysis and uncertainty quantification. This comprehensive test compares FastARD against multiple advanced numerical methods.

```julia
using FastARD
using Random, Statistics, LinearAlgebra
using Distributions
using DispatchDoctor: @stable
using CairoMakie
using MuladdMacro: @muladd
using TimerOutputs
using Printf
Random.seed!(123)

# Import the proper aPCE implementation
using APCE

# Initialize timer for performance analysis
const TO = TimerOutput()

println("="^80)
println("Ishigami Function Test: FastARD vs Advanced Numerical Methods")
println("="^80)

# ============================================================================
# Ishigami Function Implementation
# ============================================================================

"""
    ishigami(x::AbstractVector; a=7.0, b=0.1)

Ishigami function: a popular test function for sensitivity analysis and 
uncertainty quantification.

f(x1, x2, x3) = sin(x1) + a*sin²(x2) + b*x3⁴*sin(x1)

where x1, x2, x3 ∈ [-π, π]

Parameters:
- a: controls the importance of x2 (default 7.0)  
- b: controls the interaction between x1 and x3 (default 0.1)
"""
function ishigami(x::AbstractVector; a=7.0, b=0.1)
    x1, x2, x3 = x[1], x[2], x[3]
    return sin(x1) + a * sin(x2)^2 + b * x3^4 * sin(x1)
end

# ============================================================================
# Test Setup
# ============================================================================

# Generate training data
n_train = 300
n_test = 100

# Sample inputs uniformly from [-π, π]³
X_train = 2π * rand(n_train, 3) .- π
X_test = 2π * rand(n_test, 3) .- π

# Evaluate Ishigami function
y_train = [ishigami(X_train[i, :]) for i in 1:n_train]
y_test_true = [ishigami(X_test[i, :]) for i in 1:n_test]

# Add noise for realistic scenario
noise_std = 0.5
y_train_noisy = y_train + noise_std * randn(n_train)

println("Training samples: $n_train")
println("Test samples: $n_test") 
println("Noise std: $noise_std")
println("Signal-to-noise ratio: $(std(y_train) / noise_std)")
println()

# ============================================================================
# Generate Polynomial Basis
# ============================================================================

"""
Generate Legendre polynomials up to degree n for x ∈ [-1,1]
"""
function legendre_polynomials(x, max_degree)
    n = length(x)
    polys = zeros(n, max_degree + 1)
    
    # P0(x) = 1
    polys[:, 1] .= 1.0
    
    if max_degree >= 1
        # P1(x) = x  
        polys[:, 2] = x
    end
    
    # Recurrence: (n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}
    for i in 2:max_degree
        polys[:, i+1] = ((2*i-1) * x .* polys[:, i] - (i-1) * polys[:, i-1]) / i
    end
    
    return polys
end

"""
Generate multivariate polynomial basis for 3D input
"""
function generate_pce_basis(X, max_degree=3)
    n_samples, n_dims = size(X)
    @assert n_dims == 3 "Expected 3D input for Ishigami function"
    
    # Transform to [-1,1] from [-π,π]
    X_normalized = X / π
    
    # Generate univariate polynomials for each dimension
    polys = [legendre_polynomials(X_normalized[:, i], max_degree) for i in 1:n_dims]
    
    # Multivariate basis using tensor products
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
Psi_test = generate_pce_basis(X_test, max_degree)

n_basis = size(Psi_train, 2)
println("PCE basis functions: $n_basis")
println("Basis matrix condition number: $(cond(Psi_train))")
println()

# ============================================================================
# Method 1: FastARD
# ============================================================================

println("FastARD Regression")
println("-"^30)

# Time FastARD training and prediction
@timeit TO "FastARD" begin
    model_ard = FastARDRegressor(verbose=true, compute_score=true, n_iter=200)
    @timeit TO "FastARD Training" fit!(model_ard, Psi_train, y_train_noisy)

    # Get active coefficients
    active_indices, active_coefs = get_active_coefficients(model_ard)

    # Predictions with uncertainty
    @timeit TO "FastARD Prediction" begin
        y_pred_ard, y_std_ard = predict_with_uncertainty(model_ard, Psi_test)
    end
end

println("Selected $(length(active_indices)) basis functions out of $n_basis")
println("Active indices: $active_indices")

# Metrics
rmse_ard = sqrt(mean((y_pred_ard .- y_test_true).^2))
mae_ard = mean(abs.(y_pred_ard .- y_test_true))

println("FastARD Test RMSE: $rmse_ard")
println("FastARD Test MAE: $mae_ard")
println("Mean uncertainty: $(mean(y_std_ard))")
println()

# ============================================================================
# Method 2: Pseudoinverse (Pinv)
# ============================================================================

println("Pseudoinverse Regression")
println("-"^30)

# Time Pinv method
@timeit TO "Pinv" begin
    @timeit TO "Pinv Training" begin
        coef_pinv = pinv(Psi_train) * y_train_noisy
    end

    @timeit TO "Pinv Prediction" begin
        y_pred_pinv = Psi_test * coef_pinv
    end
end

# Estimate uncertainty using residual variance
residuals_train = y_train_noisy - Psi_train * coef_pinv
residual_var = var(residuals_train)

# Simple uncertainty estimate (not as sophisticated as Bayesian)
y_std_pinv = sqrt(residual_var) * ones(n_test)

# Metrics
rmse_pinv = sqrt(mean((y_pred_pinv .- y_test_true).^2))
mae_pinv = mean(abs.(y_pred_pinv .- y_test_true))

println("All $n_basis basis functions used")
println("Pinv Test RMSE: $rmse_pinv")
println("Pinv Test MAE: $mae_pinv")
println("Mean uncertainty: $(mean(y_std_pinv))")
println()

# ============================================================================
# Uncertainty Calibration Analysis
# ============================================================================

println("Uncertainty Calibration")
println("-"^30)

# Check FastARD uncertainty calibration
residuals_ard = abs.(y_pred_ard .- y_test_true)
within_1sigma_ard = sum(residuals_ard .<= y_std_ard) / n_test
within_2sigma_ard = sum(residuals_ard .<= 2 .* y_std_ard) / n_test

# Check Pinv uncertainty calibration  
residuals_pinv = abs.(y_pred_pinv .- y_test_true)
within_1sigma_pinv = sum(residuals_pinv .<= y_std_pinv) / n_test
within_2sigma_pinv = sum(residuals_pinv .<= 2 .* y_std_pinv) / n_test

println("FastARD Calibration:")
println("  Within 1σ: $(round(within_1sigma_ard, digits=3)) (should be ~0.68)")
println("  Within 2σ: $(round(within_2sigma_ard, digits=3)) (should be ~0.95)")

println("Pinv Calibration:")  
println("  Within 1σ: $(round(within_1sigma_pinv, digits=3)) (should be ~0.68)")
println("  Within 2σ: $(round(within_2sigma_pinv, digits=3)) (should be ~0.95)")
println()

# ============================================================================
# Sparsity Analysis
# ============================================================================

println("Sparsity and Efficiency Analysis")
println("-"^30)

effective_rank_ard = sum(model_ard.active)
effective_rank_pinv = n_basis

compression_ratio = effective_rank_ard / effective_rank_pinv

println("FastARD effective rank: $effective_rank_ard")
println("Pinv effective rank: $effective_rank_pinv")
println("Compression ratio: $(round(compression_ratio, digits=3))")

# Parameter magnitude analysis
println("\nLargest coefficients:")
sorted_indices = sortperm(abs.(coef_pinv), rev=true)[1:min(10, length(coef_pinv))]
for (i, idx) in enumerate(sorted_indices)
    is_selected = idx in active_indices
    marker = is_selected ? "✓" : " "
    # Get FastARD coefficient for this index
    ard_coef = if idx in active_indices
        active_idx_pos = findfirst(==(idx), active_indices)
        round(active_coefs[active_idx_pos], digits=4)
    else
        0.0
    end
    println("  $marker Basis $idx: pinv=$(round(coef_pinv[idx], digits=4)), " *
            "FastARD=$ard_coef")
end
println()

# ============================================================================
# Summary
# ============================================================================

println("COMPREHENSIVE SUMMARY")
println("="^90)
println("Method           | RMSE      | MAE       | Uncertainty | Sparsity")
println("-"^90)
println("FastARD          | $(round(rmse_ard, digits=4))    | $(round(mae_ard, digits=4))    | $(round(mean(y_std_ard), digits=4))      | $(effective_rank_ard)/$n_basis")
println("Pinv             | $(round(rmse_pinv, digits=4))    | $(round(mae_pinv, digits=4))    | $(round(mean(y_std_pinv), digits=4))      | $effective_rank_pinv/$n_basis")

# Find best method
if rmse_ard < rmse_pinv
    println("\nBEST METHOD: FastARD (RMSE: $(round(rmse_ard, digits=4)))")
    println("   FastARD achieved best accuracy with $(round((1-compression_ratio)*100, digits=1))% fewer parameters!")
else
    println("\nBEST METHOD: Pinv (RMSE: $(round(rmse_pinv, digits=4)))")
    println("   Pinv achieved best accuracy but used all parameters")
end

println("\nIshigami function comparison completed successfully!")
```

This comprehensive example demonstrates:

1. **Function approximation** using polynomial chaos expansion (PCE) basis
2. **Sparse regression** with automatic feature selection 
3. **Uncertainty quantification** with proper Bayesian treatment
4. **Method comparison** between FastARD and traditional approaches
5. **Performance analysis** including timing and sparsity metrics
6. **Calibration assessment** of uncertainty estimates

The Ishigami function is particularly challenging because it involves:
- Nonlinear interactions between variables
- Mixed polynomial and trigonometric terms
- High-dimensional polynomial basis (35 terms for degree 4)
- Noisy observations requiring robust regression

FastARD typically identifies the most relevant polynomial terms while providing well-calibrated uncertainty estimates, making it ideal for uncertainty quantification problems where both accuracy and interpretability matter.