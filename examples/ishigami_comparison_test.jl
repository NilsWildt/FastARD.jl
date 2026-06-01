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
using ArbitraryPolynomialChaosExpansion

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
# aPCE Implementation - Now using the imported APCE module
# ============================================================================

# The aPCE implementation is now imported from the APCE module above
# We can use it directly: aPCE, UQ, train!, predict, etc.

# ============================================================================
# Polynomial Basis Generation 
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

# ============================================================================
# Additional Advanced Solvers from notebook.jl
# ============================================================================

"""
Weighted Least Squares - useful when measurement errors vary
"""
@muladd function solve_weighted_least_squares(Psi, y; weights=nothing)
    if weights === nothing
        weights = ones(length(y))
    end
    W = Diagonal(sqrt.(weights))
    Psi_w = W * Psi
    y_w = W * y
    return (Psi_w' * Psi_w) \ (Psi_w' * y_w)
end

"""
Total Least Squares - accounts for errors in both Psi and y
"""
@muladd function solve_total_least_squares(Psi, y)
    augmented = [Psi y]
    U, σ, V = svd(augmented)
    solution = V[:, end]
    return -solution[1:end-1] / solution[end]
end

"""
Orthogonal Matching Pursuit - sparse solution
"""
@muladd function solve_orthogonal_matching_pursuit(Psi, y; sparsity=10)
    n = size(Psi, 2)
    x = zeros(eltype(Psi), n)
    residual = copy(y)
    selected_indices = Int[]
    
    for k in 1:min(sparsity, n)
        correlations = abs.(Psi' * residual)
        _, best_idx = findmax(correlations)
        
        if best_idx ∉ selected_indices
            push!(selected_indices, best_idx)
        end
        
        Psi_selected = Psi[:, selected_indices]
        x_selected = Psi_selected \ y
        
        x[selected_indices] = x_selected
        residual = y - Psi * x
        
        if norm(residual) < 1e-10
            break
        end
    end
    
    return x
end

"""
Elastic Net regularization (Ridge + Lasso)
"""
@muladd function solve_elastic_net(Psi, y; α=0.01, l1_ratio=0.5, max_iter=1000)
    n = size(Psi, 2)
    x = zeros(eltype(Psi), n)
    λ1 = α * l1_ratio
    λ2 = α * (1 - l1_ratio)
    
    for iter in 1:max_iter
        x_old = copy(x)
        
        for j in 1:n
            r_j = y - Psi * x + Psi[:, j] * x[j]
            
            numerator = dot(Psi[:, j], r_j)
            denominator = dot(Psi[:, j], Psi[:, j]) + λ2
            
            if abs(numerator) <= λ1
                x[j] = 0
            else
                x[j] = sign(numerator) * (abs(numerator) - λ1) / denominator
            end
        end
        
        if norm(x - x_old) < 1e-8
            break
        end
    end
    
    return x
end

"""
Levenberg-Marquardt with adaptive regularization
"""
@muladd function solve_levenberg_marquardt(Psi, y; λ_init=1e-3, max_iter=50)
    x = pinv(Psi) * y
    λ = λ_init
    
    for iter in 1:max_iter
        residual = Psi * x - y
        J = Psi
        
        JtJ = J' * J
        Jtr = J' * residual
        
        δx = -(JtJ + λ * I) \ Jtr
        x_new = x + δx
        
        new_residual = Psi * x_new - y
        
        if norm(new_residual) < norm(residual)
            x = x_new
            λ *= 0.3
        else
            λ *= 2.0
        end
        
        if norm(δx) < 1e-15
            break
        end
    end
    
    return x
end

"""
SVD with adaptive truncation strategies
"""
@muladd function solve_truncated_svd_adaptive(Psi, y)
    U, σ, V = svd(Psi)
    
    # Multiple truncation strategies
    strategies = [
        ("machine_precision", eps() * maximum(σ)),
        ("condition_1e12", maximum(σ) / 1e12),
        ("energy_99", find_energy_threshold(σ, 0.99))
    ]
    
    best_residual = Inf
    best_x = nothing
    
    for (name, threshold) in strategies
        valid_idx = σ .> threshold
        σ_inv = zeros(length(σ))
        σ_inv[valid_idx] .= 1 ./ σ[valid_idx]
        
        x_candidate = V * (σ_inv .* (U' * y))
        residual = norm(Psi * x_candidate - y)
        
        if residual < best_residual
            best_residual = residual
            best_x = x_candidate
        end
    end
    
    return best_x
end

"""
Find energy threshold for SVD truncation
"""
function find_energy_threshold(σ, energy_fraction)
    total_energy = sum(σ .* σ)
    cumulative_energy = cumsum(σ .* σ)
    threshold_idx = findfirst(cumulative_energy .>= energy_fraction * total_energy)
    return threshold_idx === nothing ? σ[end] : σ[threshold_idx]
end

"""
Ridge regression with direct implementation
"""
@muladd function solve_ridge_direct(Psi, y; λ=1e-6)
    n = size(Psi, 2)
    AtA = Psi' * Psi
    AtA[diagind(AtA)] .+= λ
    Aty = Psi' * y
    return AtA \ Aty
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
# Method 3: Additional Advanced Methods
# ============================================================================

println("Advanced Regression Methods")
println("-"^30)

# Initialize storage for all methods
methods_results = Dict{String, Any}()

# Test Weighted Least Squares
try
    @timeit TO "Weighted LS" begin
        # Give higher weights to samples with lower noise estimate
        weights = 1 ./ (abs.(y_train_noisy) .+ 0.1)
        @timeit TO "Weighted LS Training" begin
            coef_wls = solve_weighted_least_squares(Psi_train, y_train_noisy; weights=weights)
        end
        @timeit TO "Weighted LS Prediction" begin
            y_pred_wls = Psi_test * coef_wls
        end
        rmse_wls = sqrt(mean((y_pred_wls .- y_test_true).^2))
        methods_results["Weighted LS"] = (coef=coef_wls, pred=y_pred_wls, rmse=rmse_wls, sparsity=length(coef_wls))
        println("Weighted LS RMSE: $rmse_wls")
    end
catch e
    println("Weighted LS failed: $e")
end

# Test Total Least Squares
try
    @timeit TO "Total LS" begin
        @timeit TO "Total LS Training" begin
            coef_tls = solve_total_least_squares(Psi_train, y_train_noisy)
        end
        @timeit TO "Total LS Prediction" begin
            y_pred_tls = Psi_test * coef_tls
        end
        rmse_tls = sqrt(mean((y_pred_tls .- y_test_true).^2))
        methods_results["Total LS"] = (coef=coef_tls, pred=y_pred_tls, rmse=rmse_tls, sparsity=length(coef_tls))
        println("Total LS RMSE: $rmse_tls")
    end
catch e
    println("Total LS failed: $e")
end

# Test Orthogonal Matching Pursuit
try
    @timeit TO "OMP" begin
        target_sparsity = length(active_indices) + 5  # Slightly more than FastARD found
        @timeit TO "OMP Training" begin
            coef_omp = solve_orthogonal_matching_pursuit(Psi_train, y_train_noisy; sparsity=target_sparsity)
        end
        @timeit TO "OMP Prediction" begin
            y_pred_omp = Psi_test * coef_omp
        end
        rmse_omp = sqrt(mean((y_pred_omp .- y_test_true).^2))
        omp_sparsity = sum(abs.(coef_omp) .> 1e-10)
        methods_results["OMP"] = (coef=coef_omp, pred=y_pred_omp, rmse=rmse_omp, sparsity=omp_sparsity)
        println("OMP RMSE: $rmse_omp (sparsity: $omp_sparsity)")
    end
catch e
    println("OMP failed: $e")
end

# Test Elastic Net
try
    @timeit TO "Elastic Net" begin
        @timeit TO "Elastic Net Training" begin
            coef_enet = solve_elastic_net(Psi_train, y_train_noisy; α=0.1, l1_ratio=0.7)
        end
        @timeit TO "Elastic Net Prediction" begin
            y_pred_enet = Psi_test * coef_enet
        end
        rmse_enet = sqrt(mean((y_pred_enet .- y_test_true).^2))
        enet_sparsity = sum(abs.(coef_enet) .> 1e-10)
        methods_results["Elastic Net"] = (coef=coef_enet, pred=y_pred_enet, rmse=rmse_enet, sparsity=enet_sparsity)
        println("Elastic Net RMSE: $rmse_enet (sparsity: $enet_sparsity)")
    end
catch e
    println("Elastic Net failed: $e")
end

# Test Levenberg-Marquardt
try
    @timeit TO "L-M" begin
        @timeit TO "L-M Training" begin
            coef_lm = solve_levenberg_marquardt(Psi_train, y_train_noisy)
        end
        @timeit TO "L-M Prediction" begin
            y_pred_lm = Psi_test * coef_lm
        end
        rmse_lm = sqrt(mean((y_pred_lm .- y_test_true).^2))
        methods_results["L-M"] = (coef=coef_lm, pred=y_pred_lm, rmse=rmse_lm, sparsity=length(coef_lm))
        println("Levenberg-Marquardt RMSE: $rmse_lm")
    end
catch e
    println("Levenberg-Marquardt failed: $e")
end

# Test Adaptive SVD
try
    @timeit TO "Adaptive SVD" begin
        @timeit TO "Adaptive SVD Training" begin
            coef_asvd = solve_truncated_svd_adaptive(Psi_train, y_train_noisy)
        end
        @timeit TO "Adaptive SVD Prediction" begin
            y_pred_asvd = Psi_test * coef_asvd
        end
        rmse_asvd = sqrt(mean((y_pred_asvd .- y_test_true).^2))
        methods_results["Adaptive SVD"] = (coef=coef_asvd, pred=y_pred_asvd, rmse=rmse_asvd, sparsity=length(coef_asvd))
        println("Adaptive SVD RMSE: $rmse_asvd")
    end
catch e
    println("Adaptive SVD failed: $e")
end

# Test Ridge regression
try
    @timeit TO "Ridge" begin
        @timeit TO "Ridge Training" begin
            coef_ridge = solve_ridge_direct(Psi_train, y_train_noisy; λ=0.1)
        end
        @timeit TO "Ridge Prediction" begin
            y_pred_ridge = Psi_test * coef_ridge
        end
        rmse_ridge = sqrt(mean((y_pred_ridge .- y_test_true).^2))
        methods_results["Ridge"] = (coef=coef_ridge, pred=y_pred_ridge, rmse=rmse_ridge, sparsity=length(coef_ridge))
        println("Ridge RMSE: $rmse_ridge")
    end
catch e
    println("Ridge failed: $e")
end

println()

# ============================================================================
# Method 4: PCE Analysis using proper aPCE
# ============================================================================

println("PCE Statistical Analysis")
println("-"^30)

# Create proper aPCE models for comparison
apc_native = aPCE(X_train, max_degree; outdim=1)
train!(apc_native, X_train, y_train_noisy; bayesian_inversion=false)

# Create aPCE structure for FastARD result (sparse)
apc_ard = aPCE(X_train, max_degree; outdim=1)
# Copy the sparse coefficients from FastARD to match the PCE basis
Psi_for_ard = aPCE_PsiPolynomialMatrix(apc_ard, X_train)'
n_available_terms = min(length(active_indices), apc_ard.NumberOfTerms)
for (i, idx) in enumerate(active_indices[1:n_available_terms])
    if idx <= apc_ard.NumberOfTerms
        apc_ard.ExpansionCoefficients[idx, 1] = active_coefs[i]
    end
end

# Create aPCE structure for Pinv result (full)
apc_pinv = aPCE(X_train, max_degree; outdim=1)
n_copy_terms = min(length(coef_pinv), apc_pinv.NumberOfTerms)
apc_pinv.ExpansionCoefficients[1:n_copy_terms, 1] = coef_pinv[1:n_copy_terms]

# Compute statistics
uq_native = UQ(apc_native)
uq_ard = UQ(apc_ard)
uq_pinv = UQ(apc_pinv)

println("PCE Statistics (from coefficients):")
println("Native PCE - Mean: $(uq_native.OutputMean[1]), Variance: $(uq_native.OutputVar[1])")
println("FastARD PCE - Mean: $(uq_ard.OutputMean[1]), Variance: $(uq_ard.OutputVar[1])")
println("Pinv PCE - Mean: $(uq_pinv.OutputMean[1]), Variance: $(uq_pinv.OutputVar[1])")

# Compare with empirical statistics from test data
emp_mean = mean(y_test_true)
emp_var = var(y_test_true)
println("True empirical - Mean: $emp_mean, Variance: $emp_var")

# Show PCE uncertainty bounds vs empirical
println("\nPCE Uncertainty Bounds (±1σ):")
println("Native PCE: $(uq_native.OutputMean[1]) ± $(sqrt(uq_native.OutputVar[1]))")
println("FastARD PCE: $(uq_ard.OutputMean[1]) ± $(sqrt(uq_ard.OutputVar[1]))")
println("Pinv PCE: $(uq_pinv.OutputMean[1]) ± $(sqrt(uq_pinv.OutputVar[1]))")
println("Empirical: $emp_mean ± $(sqrt(emp_var))")
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
# Polynomial Basis Visualization
# ============================================================================

println(" Creating Legendre polynomial visualization...")

# Create figure for polynomial basis visualization
fig_poly = Figure(size=(1200, 800), fontsize=12)

# Generate points for smooth polynomial visualization
x_smooth = range(-1, 1, length=200)
max_degree_vis = 5

# Plot univariate Legendre polynomials
ax_poly = Axis(fig_poly[1, 1], 
    title="Legendre Polynomials P₀(x) to P₅(x)", 
    xlabel="x ∈ [-1, 1]", 
    ylabel="Polynomial Value")

# Generate and plot polynomials
polys_smooth = legendre_polynomials(x_smooth, max_degree_vis)
colors_poly = [:blue, :red, :green, :orange, :purple, :brown]

for i in 1:(max_degree_vis+1)
    lines!(ax_poly, x_smooth, polys_smooth[:, i], 
           color=colors_poly[i], linewidth=2, 
           label="P$(i-1)(x)")
end

axislegend(ax_poly, position=:rt)

# Add grid for better readability
ax_poly.xgridvisible = true
ax_poly.ygridvisible = true

# Plot multivariate basis function magnitudes
ax_basis = Axis(fig_poly[1, 2], 
    title="Multivariate Basis Function Magnitudes", 
    xlabel="Basis Function Index", 
    ylabel="Mean Absolute Value")

# Calculate mean absolute values of basis functions
basis_magnitudes = [mean(abs.(Psi_train[:, i])) for i in 1:size(Psi_train, 2)]
barplot!(ax_basis, 1:length(basis_magnitudes), basis_magnitudes, 
         color=:steelblue, alpha=0.7)

# Highlight active basis functions from FastARD
if !isempty(active_indices)
    scatter!(ax_basis, active_indices, basis_magnitudes[active_indices], 
             color=:red, markersize=8, label="FastARD Active")
    axislegend(ax_basis, position=:rt)
end

# Orthogonality check visualization
ax_ortho = Axis(fig_poly[2, 1:2], 
    title="Basis Function Orthogonality Check (Gram Matrix)", 
    xlabel="Basis Function Index", 
    ylabel="Basis Function Index")

# Compute and visualize Gram matrix (should be near-diagonal for orthogonal functions)
n_vis = min(20, size(Psi_train, 2))  # Visualize first 20 for clarity
Psi_subset = Psi_train[:, 1:n_vis]
gram_matrix = (Psi_subset' * Psi_subset) / size(Psi_train, 1)  # Normalized

heatmap!(ax_ortho, 1:n_vis, 1:n_vis, gram_matrix, 
         colormap=:RdBu, colorrange=(-0.5, 0.5))

save("examples/legendre_polynomials_analysis.pdf", fig_poly)
println("Legendre polynomial analysis saved as 'legendre_polynomials_analysis.pdf'")

# ============================================================================
# Timing Breakdown Visualization
# ============================================================================

println("Creating timing breakdown visualization...")

# Helper function to get timing data from TimerOutput
function get_timer_data(timer_name::String)
    try
        # Access the timer data through inner_timers
        if haskey(TO.inner_timers, timer_name)
            timer_data = TO.inner_timers[timer_name]
            # Check if this timer has accumulated_data with time
            if hasfield(typeof(timer_data), :accumulated_data)
                total_time = timer_data.accumulated_data.time
            else
                return (0.0, 0.0, 0.0)
            end
        else
            return (0.0, 0.0, 0.0)
        end
        
        # Try to get training and prediction times
        training_time = 0.0
        prediction_time = 0.0
        
        # Look for training timer in the main timer's inner_timers
        training_timer_name = timer_name * " Training"
        if haskey(timer_data.inner_timers, training_timer_name)
            training_timer = timer_data.inner_timers[training_timer_name]
            if hasfield(typeof(training_timer), :accumulated_data)
                training_time = training_timer.accumulated_data.time
            end
        end
        
        # Look for prediction timer in the main timer's inner_timers
        prediction_timer_name = timer_name * " Prediction"
        if haskey(timer_data.inner_timers, prediction_timer_name)
            prediction_timer = timer_data.inner_timers[prediction_timer_name]
            if hasfield(typeof(prediction_timer), :accumulated_data)
                prediction_time = prediction_timer.accumulated_data.time
            end
        end
        
        # Convert nanoseconds to milliseconds
        return (training_time/1e6, prediction_time/1e6, total_time/1e6)
    catch e
        return (0.0, 0.0, 0.0)
    end
end

# Create timing breakdown figure
fig_timing = Figure(size=(1000, 600), fontsize=12)

# Panel 1: Training vs Prediction Time Breakdown
ax_breakdown = Axis(fig_timing[1, 1], 
    title="Training vs Prediction Time Breakdown", 
    xlabel="Method", 
    ylabel="Time (ms)",
    xticklabelrotation=π/4)

# Collect training and prediction times
methods_with_timing = []
training_times = []
prediction_times = []

# Helper function to get both training and prediction times
function get_detailed_timing(method_name::String)
    try
        timer_data = TO[method_name]
        training_time = 0.0
        prediction_time = 0.0
        
        # Get sub-timer data
        for (key, sub_timer) in timer_data.inner_timers
            if occursin("Training", key)
                training_time = sub_timer.time / 1e6  # Convert to ms
            elseif occursin("Prediction", key)
                prediction_time = sub_timer.time / 1e6  # Convert to ms
            end
        end
        
        return training_time, prediction_time
    catch
        return 0.0, 0.0
    end
end

# FastARD
fastard_train, fastard_pred = get_detailed_timing("FastARD")
if fastard_train > 0 || fastard_pred > 0
    push!(methods_with_timing, "FastARD")
    push!(training_times, fastard_train)
    push!(prediction_times, fastard_pred)
end

# Pinv
pinv_train, pinv_pred = get_detailed_timing("Pinv")
if pinv_train > 0 || pinv_pred > 0
    push!(methods_with_timing, "Pinv")
    push!(training_times, pinv_train)
    push!(prediction_times, pinv_pred)
end

# Other methods
for (name, _) in methods_results
    train_time, pred_time = get_detailed_timing(name)
    if train_time > 0 || pred_time > 0
        push!(methods_with_timing, name)
        push!(training_times, train_time)
        push!(prediction_times, pred_time)
    end
end

if !isempty(methods_with_timing)
    x_pos = 1:length(methods_with_timing)
    
    # Create stacked bar chart
    barplot!(ax_breakdown, x_pos, training_times, 
             color=:steelblue, alpha=0.8, label="Training")
    barplot!(ax_breakdown, x_pos, prediction_times, 
             color=:coral, alpha=0.8, label="Prediction",
             stack=training_times)
    
    ax_breakdown.xticks = (x_pos, methods_with_timing)
    axislegend(ax_breakdown, position=:rt)
end

# Panel 2: Performance Efficiency (Accuracy per unit time)
ax_efficiency = Axis(fig_timing[1, 2], 
    title="Performance Efficiency (Lower is Better)", 
    xlabel="Method", 
    ylabel="RMSE × Time (lower = more efficient)",
    xticklabelrotation=π/4)

# Calculate efficiency metric (RMSE × Total Time)
efficiency_methods = []
efficiency_scores = []

# Get timing data for efficiency calculation
fastard_training_eff, fastard_prediction_eff, fastard_total_eff = get_timer_data("FastARD")
pinv_training_eff, pinv_prediction_eff, pinv_total_eff = get_timer_data("Pinv")

# FastARD efficiency
if fastard_total_eff > 0
    push!(efficiency_methods, "FastARD")
    push!(efficiency_scores, rmse_ard * fastard_total_eff)
end

# Pinv efficiency
if pinv_total_eff > 0
    push!(efficiency_methods, "Pinv")
    push!(efficiency_scores, rmse_pinv * pinv_total_eff)
end

# Other methods efficiency
for (name, results) in methods_results
    _, _, total_time = get_timer_data(name)
    if total_time > 0
        push!(efficiency_methods, name)
        push!(efficiency_scores, results.rmse * total_time)
    end
end

if !isempty(efficiency_scores)
    barplot!(ax_efficiency, 1:length(efficiency_methods), efficiency_scores, 
             color=:purple, alpha=0.7)
    ax_efficiency.xticks = (1:length(efficiency_methods), efficiency_methods)
end

save("examples/timing_analysis.pdf", fig_timing)
println(" Timing analysis saved as 'timing_analysis.pdf'")

# ============================================================================
# Main Visualization
# ============================================================================

println(" Creating comprehensive comparison visualization...")

# Set up the figure with multiple panels
fig = Figure(size=(1400, 1200), fontsize=12)

# Panel 1: True vs Predicted comparison
ax1 = Axis(fig[1, 1], 
    title="True vs Predicted Values", 
    xlabel="True Values", 
    ylabel="Predicted Values",
    aspect=DataAspect())

# Plot diagonal line for perfect prediction
min_val, max_val = extrema([y_test_true; y_pred_ard; y_pred_pinv])
lines!(ax1, [min_val, max_val], [min_val, max_val], color=:gray, linestyle=:dash, linewidth=2, label="Perfect")

# Scatter plots for predictions
scatter!(ax1, y_test_true, y_pred_ard, color=(:blue, 0.6), markersize=8, label="FastARD")
scatter!(ax1, y_test_true, y_pred_pinv, color=(:red, 0.6), markersize=8, label="Pinv")

# Add predictions from successful additional methods
method_colors = [:green, :orange, :purple, :brown, :pink, :cyan, :yellow]
i_color = 1
for (name, results) in methods_results
    global i_color
    if i_color <= length(method_colors)
        scatter!(ax1, y_test_true, results.pred, 
                color=(method_colors[i_color], 0.6), markersize=6, 
                label=name)
        i_color += 1
    end
end

axislegend(ax1, position=:lt)

# Panel 2: Timing comparison
ax2 = Axis(fig[1, 2], 
    title="Execution Time Comparison", 
    xlabel="Method", 
    ylabel="Time (ms, log scale)",
    yscale=log10,
    xticklabelrotation=π/4)

# Collect timing data for plotting
timing_names = []
timing_values = []

# Add FastARD timing
fastard_training_plot, fastard_prediction_plot, fastard_total_plot = get_timer_data("FastARD")
if fastard_total_plot > 0
    push!(timing_names, "FastARD")
    push!(timing_values, fastard_total_plot)
end

# Add Pinv timing
pinv_training_plot, pinv_prediction_plot, pinv_total_plot = get_timer_data("Pinv")
if pinv_total_plot > 0
    push!(timing_names, "Pinv")
    push!(timing_values, pinv_total_plot)
end

# Add other methods
for (name, results) in methods_results
    training_time, prediction_time, total_time = get_timer_data(name)
    if total_time > 0
        push!(timing_names, name)
        push!(timing_values, total_time)
    end
end

if !isempty(timing_values)
    barplot!(ax2, 1:length(timing_names), timing_values, 
             color=:coral, alpha=0.7)
    ax2.xticks = (1:length(timing_names), timing_names)
end

# Panel 3: Sparsity comparison
ax3 = Axis(fig[2, 1], 
    title="Sparsity Comparison", 
    xlabel="Method", 
    ylabel="Number of Active Features",
    xticklabelrotation=π/4)

# Collect sparsity data
sparsity_names = ["FastARD", "Pinv"]
sparsity_values = [effective_rank_ard, effective_rank_pinv]

for (name, results) in methods_results
    push!(sparsity_names, name)
    push!(sparsity_values, results.sparsity)
end

barplot!(ax3, 1:length(sparsity_names), sparsity_values, 
         color=:orange, alpha=0.7)

ax3.xticks = (1:length(sparsity_names), sparsity_names)

# Add total features line
hlines!(ax3, [n_basis], color=:red, linestyle=:dash, linewidth=2, label="Total Features")
axislegend(ax3, position=:rt)

# Panel 4: Best methods uncertainty visualization
ax4 = Axis(fig[2, 2], 
    title="Predictions with Uncertainty (Best Methods)", 
    xlabel="Test Sample Index", 
    ylabel="Output Value")

# Sort by true values for better visualization
sort_idx = sortperm(y_test_true)
x_plot = 1:length(y_test_true)

# Plot true values
lines!(ax4, x_plot, y_test_true[sort_idx], color=:black, linewidth=2, label="True")

# Plot FastARD predictions with uncertainty
lines!(ax4, x_plot, y_pred_ard[sort_idx], color=:blue, linewidth=2, label="FastARD")
band!(ax4, x_plot, 
      (y_pred_ard[sort_idx] .- y_std_ard[sort_idx]), 
      (y_pred_ard[sort_idx] .+ y_std_ard[sort_idx]), 
      color=(:blue, 0.2), label="FastARD ±1σ")

# Find and plot best additional method
best_method_name = ""
best_plot_rmse = Inf
best_pred = nothing

if !isempty(methods_results)
    # Find best method among additional methods
    for (name, results) in methods_results
        global best_plot_rmse
        global best_pred
        global best_method_name
        if results.rmse < best_plot_rmse
            best_plot_rmse = results.rmse
            best_method_name = name
            best_pred = results.pred
        end
    end
    
    if best_pred !== nothing
        lines!(ax4, x_plot, best_pred[sort_idx], color=:green, linewidth=2, 
               label="$best_method_name (best)")
    end
end

axislegend(ax4, position=:lt)

# Add overall title and metrics
Label(fig[0, :], "FastARD vs Advanced Numerical Methods: Ishigami Function", 
      fontsize=16, font="bold")

# Add summary statistics as text with timing
metrics_lines = []

# Get timing data for metrics
fastard_training_metrics, fastard_prediction_metrics, fastard_total_metrics = get_timer_data("FastARD")
pinv_training_metrics, pinv_prediction_metrics, pinv_total_metrics = get_timer_data("Pinv")

# FastARD with timing
fastard_time_str = fastard_total_metrics > 0 ? " ($(round(fastard_total_metrics, digits=1))ms)" : ""
push!(metrics_lines, "FastARD: RMSE=$(round(rmse_ard, digits=4)), $(effective_rank_ard)/$n_basis features$fastard_time_str")

# Pinv with timing  
pinv_time_str = pinv_total_metrics > 0 ? " ($(round(pinv_total_metrics, digits=1))ms)" : ""
push!(metrics_lines, "Pinv: RMSE=$(round(rmse_pinv, digits=4)), $effective_rank_pinv/$n_basis features$pinv_time_str")

for (name, results) in methods_results
    sparsity_info = results.sparsity == length(results.coef) ? "all" : "$(results.sparsity)"
    training_time, prediction_time, total_time = get_timer_data(name)
    time_str = total_time > 0 ? " ($(round(total_time, digits=1))ms)" : ""
    push!(metrics_lines, "$name: RMSE=$(round(results.rmse, digits=4)), $sparsity_info/$n_basis features$time_str")
end

push!(metrics_lines, "Compression: $(round(compression_ratio*100, digits=1))% of features retained (FastARD)")

metrics_text = join(metrics_lines, "\n")
Label(fig[3, :], metrics_text, fontsize=10, tellwidth=false)

# Save the plot
save("examples/ishigami_comparison_results.pdf", fig)
println(" Visualization saved as 'ishigami_comparison_results.pdf'")

# Display convergence if scores available
if !isempty(model_ard.scores) && length(model_ard.scores) > 1
    fig_conv = Figure(size=(600, 400))
    ax_conv = Axis(fig_conv[1, 1], 
        title="FastARD Convergence", 
        xlabel="Iteration", 
        ylabel="Log Marginal Likelihood")
    
    scatterlines!(ax_conv, 1:length(model_ard.scores), model_ard.scores, 
                  color=:blue, linewidth=2, markersize=6)
    
    save("examples/ishigami_convergence.pdf", fig_conv)
    println(" Convergence plot saved as 'ishigami_convergence.pdf'")
end

println()

# ============================================================================
# Summary
# ============================================================================

println(" COMPREHENSIVE SUMMARY")
println("="^90)
println("Method           | RMSE      | MAE       | Uncertainty | PCE Var    | Sparsity")
println("-"^90)
println("FastARD          | $(round(rmse_ard, digits=4))    | $(round(mae_ard, digits=4))    | $(round(mean(y_std_ard), digits=4))      | $(round(uq_ard.OutputVar[1], digits=4))      | $(effective_rank_ard)/$n_basis")
println("Pinv             | $(round(rmse_pinv, digits=4))    | $(round(mae_pinv, digits=4))    | $(round(mean(y_std_pinv), digits=4))      | $(round(uq_pinv.OutputVar[1], digits=4))      | $effective_rank_pinv/$n_basis")

# Add results from additional methods
for (name, results) in methods_results
    method_name_padded = rpad(name, 16)
    sparsity_str = results.sparsity == length(results.coef) ? "all" : "$(results.sparsity)"
    println("$method_name_padded | $(round(results.rmse, digits=4))    | N/A       | N/A         | N/A        | $sparsity_str/$n_basis")
end
println()

# Find overall best method
all_methods = [("FastARD", rmse_ard), ("Pinv", rmse_pinv)]
for (name, results) in methods_results
    push!(all_methods, (name, results.rmse))
end

best_method, best_overall_rmse = all_methods[argmin([rmse for (_, rmse) in all_methods])]

println(" BEST OVERALL METHOD: $best_method (RMSE: $(round(best_overall_rmse, digits=4)))")

if best_method == "FastARD"
    println("   FastARD achieved best accuracy with $(round((1-compression_ratio)*100, digits=1))% fewer parameters!")
elseif best_method == "Pinv"
    println("   Pinv achieved best accuracy but used all parameters")
else
    # Find sparsity of best method
    if haskey(methods_results, best_method)
        best_sparsity = methods_results[best_method].sparsity
        sparsity_ratio = best_sparsity / n_basis
        println("   $best_method achieved best accuracy using $(round(sparsity_ratio*100, digits=1))% of available features")
    end
end

println()
println(" SPARSITY ANALYSIS:")
sparse_methods = [("FastARD", effective_rank_ard)]
for (name, results) in methods_results
    if haskey(results, :sparsity) && results.sparsity < n_basis
        push!(sparse_methods, (name, results.sparsity))
    end
end

if length(sparse_methods) > 1
    println("   Sparse methods found:")
    for (name, sparsity) in sparse_methods
        println("     - $name: $sparsity/$n_basis features ($(round(sparsity/n_basis*100, digits=1))%)")
    end
else
    println("   Only FastARD achieved meaningful sparsity")
end

# ============================================================================
# Performance Analysis with TimerOutputs
# ============================================================================

println("\n PERFORMANCE ANALYSIS")
println("="^90)

# Display the full timer output
show(TO; allocations=false, compact=false)
println()

# Create a comprehensive timing and performance table
println("\n COMPREHENSIVE PERFORMANCE TABLE")
println("="^110)
println(@sprintf("%-16s | %-8s | %-10s | %-10s | %-10s | %-8s | %-12s", "Method", "RMSE", "Training", "Prediction", "Total", "Sparsity", "Speed Rank"))
println("-"^110)

# Collect all timing and performance data
performance_data = []

# FastARD
fastard_training_perf, fastard_prediction_perf, fastard_total_perf = get_timer_data("FastARD")
push!(performance_data, ("FastARD", rmse_ard, fastard_training_perf, fastard_prediction_perf, fastard_total_perf, effective_rank_ard))

# Pinv
pinv_training_perf, pinv_prediction_perf, pinv_total_perf = get_timer_data("Pinv")
push!(performance_data, ("Pinv", rmse_pinv, pinv_training_perf, pinv_prediction_perf, pinv_total_perf, effective_rank_pinv))

# Additional methods
for (name, results) in methods_results
    training_time, prediction_time, total_time = get_timer_data(name)
    push!(performance_data, (name, results.rmse, training_time, prediction_time, total_time, results.sparsity))
end

# Sort by total time for speed ranking (methods with missing timing data go last)
valid_timing_data = filter(x -> x[5] > 0, performance_data)
invalid_timing_data = filter(x -> x[5] <= 0, performance_data)

sorted_by_speed_valid = sort(valid_timing_data, by=x->x[5])  # Sort by total time
sorted_by_speed = vcat(sorted_by_speed_valid, invalid_timing_data)

speed_ranks = Dict{String, Int}()
for (i, (method_name, _, _, _, _, _)) in enumerate(sorted_by_speed)
    speed_ranks[method_name] = i
end

# Print the table sorted by RMSE (best accuracy first)
sorted_by_rmse = sort(performance_data, by=x->x[2])

for (method_name, rmse, training_ms, prediction_ms, total_ms, sparsity) in sorted_by_rmse
    speed_rank = speed_ranks[method_name]
    
    # Format strings
    rmse_str = @sprintf("%.6f", rmse)
    training_str = training_ms > 0 ? @sprintf("%.2f ms", training_ms) : "N/A"
    prediction_str = prediction_ms > 0 ? @sprintf("%.2f ms", prediction_ms) : "N/A" 
    total_str = total_ms > 0 ? @sprintf("%.2f ms", total_ms) : "N/A"
    sparsity_str = "$sparsity/$n_basis"
    println(@sprintf("%-16s | %-8s | %-10s | %-10s | %-10s | %-8s | %-12s", method_name, rmse_str, training_str, prediction_str, total_str, sparsity_str, "#$speed_rank"))
end

println()

# Performance insights
println(" PERFORMANCE INSIGHTS:")
fastest_method = sorted_by_speed[1][1]
fastest_time = sorted_by_speed[1][5]
println("   Fastest Method: $fastest_method ($(round(fastest_time, digits=2)) ms)")

most_accurate = sorted_by_rmse[1][1]
most_accurate_rmse = sorted_by_rmse[1][2]
println("   Most Accurate: $most_accurate (RMSE: $(round(most_accurate_rmse, digits=6)))")

# FastARD analysis
fastard_speed_rank = speed_ranks["FastARD"]
fastard_accuracy_rank = findfirst(x -> x[1] == "FastARD", sorted_by_rmse)
println("   FastARD Ranking: #$fastard_accuracy_rank in accuracy, #$fastard_speed_rank in speed")

if fastard_speed_rank <= 3 && fastard_accuracy_rank <= 3
    println("    FastARD achieves top-3 performance in both accuracy AND speed!")
elseif fastard_accuracy_rank <= 3
    println("    FastARD achieves top-3 accuracy with sparsity benefits")
elseif fastard_speed_rank <= 3
    println("   ⚡ FastARD is among the fastest methods")
end

# Speed vs accuracy trade-off analysis
println("\n SPEED vs ACCURACY TRADE-OFF:")
for (i, (method_name, rmse, _, _, total_ms, sparsity)) in enumerate(sorted_by_rmse[1:3])
    speed_rank = speed_ranks[method_name]
    efficiency_score = (4 - i) * (length(performance_data) + 1 - speed_rank)  # Higher is better
    println("   $method_name: Accuracy rank #$i, Speed rank #$speed_rank, Efficiency score: $efficiency_score")
end

println("\nPerformance analysis completed successfully!")

# ============================================================================
# Uncertainty Bands Comparison Plot
# ============================================================================

println("\n Creating uncertainty bands comparison plot...")

# Create figure for uncertainty bands comparison
fig_uncertainty = Figure(size=(1400, 800), fontsize=12)

# Panel 1: FastARD uncertainty analysis
ax_unc1 = Axis(fig_uncertainty[1, 1], 
    title="FastARD Predictions with Uncertainty Bands", 
    xlabel="Test Sample Index", 
    ylabel="Output Value")

# Sort by true values for better visualization
sort_idx = sortperm(y_test_true)
x_plot = 1:length(y_test_true)

# Plot true values
lines!(ax_unc1, x_plot, y_test_true[sort_idx], color=:black, linewidth=3, label="True Values")

# Plot FastARD predictions with multiple uncertainty levels
lines!(ax_unc1, x_plot, y_pred_ard[sort_idx], color=:blue, linewidth=2, label="FastARD Prediction")

# Add multiple uncertainty bands
band!(ax_unc1, x_plot, 
      (y_pred_ard[sort_idx] .- y_std_ard[sort_idx]), 
      (y_pred_ard[sort_idx] .+ y_std_ard[sort_idx]), 
      color=(:blue, 0.3), label="±1σ (68% confidence)")

band!(ax_unc1, x_plot, 
      (y_pred_ard[sort_idx] .- 2 .* y_std_ard[sort_idx]), 
      (y_pred_ard[sort_idx] .+ 2 .* y_std_ard[sort_idx]), 
      color=(:blue, 0.15), label="±2σ (95% confidence)")

axislegend(ax_unc1, position=:lt)

# Panel 2: All methods uncertainty comparison (if available)
ax_unc2 = Axis(fig_uncertainty[1, 2], 
    title="Uncertainty Comparison Across Methods", 
    xlabel="Test Sample Index", 
    ylabel="Output Value")

# Plot true values
lines!(ax_unc2, x_plot, y_test_true[sort_idx], color=:black, linewidth=3, label="True Values")

# Plot FastARD with uncertainty
lines!(ax_unc2, x_plot, y_pred_ard[sort_idx], color=:blue, linewidth=2, label="FastARD")
band!(ax_unc2, x_plot, 
      (y_pred_ard[sort_idx] .- y_std_ard[sort_idx]), 
      (y_pred_ard[sort_idx] .+ y_std_ard[sort_idx]), 
      color=(:blue, 0.2), label="FastARD ±1σ")

# Plot Pinv with simple uncertainty estimate
lines!(ax_unc2, x_plot, y_pred_pinv[sort_idx], color=:red, linewidth=2, label="Pinv")
band!(ax_unc2, x_plot, 
      (y_pred_pinv[sort_idx] .- y_std_pinv[sort_idx]), 
      (y_pred_pinv[sort_idx] .+ y_std_pinv[sort_idx]), 
      color=(:red, 0.15), label="Pinv ±1σ")

# Add best additional method if available
if !isempty(methods_results) && best_pred !== nothing
    lines!(ax_unc2, x_plot, best_pred[sort_idx], color=:green, linewidth=2, 
           label="$best_method_name")
    # Note: Most methods don't provide uncertainty estimates, so only prediction line
end

axislegend(ax_unc2, position=:lt)

# Panel 3: Uncertainty calibration analysis
ax_unc3 = Axis(fig_uncertainty[2, 1], 
    title="Uncertainty Calibration Analysis", 
    xlabel="Predicted Uncertainty (σ)", 
    ylabel="Actual Error |y_true - y_pred|")

# Scatter plot of predicted uncertainty vs actual error
scatter!(ax_unc3, y_std_ard, abs.(y_pred_ard .- y_test_true), 
         color=:blue, alpha=0.6, markersize=8, label="FastARD")

# Add perfect calibration line (y=x)
max_val = max(maximum(y_std_ard), maximum(abs.(y_pred_ard .- y_test_true)))
lines!(ax_unc3, [0, max_val], [0, max_val], color=:gray, linestyle=:dash, 
       linewidth=2, label="Perfect Calibration")

# Add linear regression line for actual calibration
if length(y_std_ard) > 1
    # Simple linear regression to show calibration trend
    X_calib = [ones(length(y_std_ard)) y_std_ard]
    β_calib = X_calib \ abs.(y_pred_ard .- y_test_true)
    y_fit = X_calib * β_calib
    lines!(ax_unc3, y_std_ard, y_fit, color=:red, linewidth=2, 
           label="Actual Calibration")
end

axislegend(ax_unc3, position=:rt)

# Panel 4: Coverage probability analysis
ax_unc4 = Axis(fig_uncertainty[2, 2], 
    title="Coverage Probability Analysis", 
    xlabel="Confidence Level", 
    ylabel="Actual Coverage")

# Calculate coverage for different confidence levels
confidence_levels = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
actual_coverage = []

for conf_level in confidence_levels
    # Calculate how many standard deviations for this confidence level
    z_score = if conf_level == 0.5
        0.674  # 50% confidence
    elseif conf_level == 0.68
        1.0    # 68% confidence (1σ)
    elseif conf_level == 0.8
        1.282  # 80% confidence
    elseif conf_level == 0.9
        1.645  # 90% confidence
    elseif conf_level == 0.95
        1.96   # 95% confidence (2σ)
    elseif conf_level == 0.99
        2.576  # 99% confidence
    else
        1.0
    end
    
    # Count how many predictions fall within this confidence interval
    within_interval = sum(abs.(y_pred_ard .- y_test_true) .<= z_score .* y_std_ard)
    coverage = within_interval / length(y_test_true)
    push!(actual_coverage, coverage)
end

# Plot expected vs actual coverage
lines!(ax_unc4, confidence_levels, confidence_levels, color=:gray, linestyle=:dash, 
       linewidth=2, label="Perfect Coverage")
scatterlines!(ax_unc4, confidence_levels, actual_coverage, color=:blue, 
              linewidth=2, markersize=8, label="FastARD Coverage")

axislegend(ax_unc4, position=:rb)

save("examples/uncertainty_analysis.pdf", fig_uncertainty)
println(" Uncertainty analysis saved as 'uncertainty_analysis.pdf'")

# Print uncertainty statistics
println("\n UNCERTAINTY ANALYSIS SUMMARY:")
println("="^60)

# FastARD uncertainty statistics
mean_uncertainty = mean(y_std_ard)
median_uncertainty = median(y_std_ard)
println("FastARD Uncertainty Statistics:")
println("  Mean predicted uncertainty: $(round(mean_uncertainty, digits=4))")
println("  Median predicted uncertainty: $(round(median_uncertainty, digits=4))")

# Coverage analysis
within_1sigma = sum(abs.(y_pred_ard .- y_test_true) .<= y_std_ard) / length(y_test_true)
within_2sigma = sum(abs.(y_pred_ard .- y_test_true) .<= 2 .* y_std_ard) / length(y_test_true)

println("\nCoverage Analysis:")
println("  Within 1σ: $(round(within_1sigma*100, digits=1))% (expected: 68.0%)")
println("  Within 2σ: $(round(within_2sigma*100, digits=1))% (expected: 95.0%)")

# Calibration quality
if within_1sigma > 0.6 && within_1sigma < 0.75
    println("  1σ Calibration: Well calibrated")
elseif within_1sigma > 0.75
    println("  1σ Calibration: ⚠  Conservative (overconfident)")
else
    println("  1σ Calibration: ⚠  Aggressive (underconfident)")
end

if within_2sigma > 0.90 && within_2sigma < 0.98
    println("  2σ Calibration: Well calibrated")
elseif within_2sigma > 0.98
    println("  2σ Calibration: ⚠  Conservative (overconfident)")
else
    println("  2σ Calibration: ⚠  Aggressive (underconfident)")
end

println("\nUncertainty analysis completed successfully!")