module FastARD
using MultiFloats
using LinearAlgebra
using LinearAlgebra: I
using Statistics
using ConcreteStructs
using DispatchDoctor
using TypeUtils
using TestItems
using Test
using GenericLinearAlgebra: eigvals

	
export FastARDRegressor, fit!, predict, get_active_coefficients, predict_with_uncertainty

@inline Base.precision(::Type{MultiFloat{T,N}}) where {T,N} =
    N * precision(T) + (N - 1) # implicit bits of precision between limbs
	
# ============================================================================
# Utility Functions
# ============================================================================

@stable function round_close_to_zero(x::T, eps_val::T = eps(T)) where {T <: Real}
    return abs(x) > eps_val ? x : zero(T)
end

@stable function round_close_to_zero(x::T, eps_val::V = eps(V)) where {T <: Real, V <: Real}
    return abs(x) > eps_val ? x : zero(T)
end

@stable function safe_sqrt(x::T) where T<:Real
    if x >= zero(T)
        return sqrt(x)
    else
        ε = eps(T)
        if -x < ε^(3//4)
            @warn "Taking sqrt of small negative number $x, returning ε"
            return ε
        else
            throw(DomainError(x, "Cannot take sqrt of significantly negative number"))
        end
    end
end


"""
    only_finite(x, y)

Filter paired vectors to keep only elements where y is finite.
"""
function only_finite(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: Real}
    length(x) == length(y) || throw(ArgumentError("Vectors must have equal length"))
    finite_mask = isfinite.(y)
    return x[finite_mask], y[finite_mask]
end

# ============================================================================
# Core Data Structures
# ============================================================================


"""
    PrecomputedCache{T,M,V}

Holds precomputed matrices for efficient computation.
Supports both regular arrays and static arrays.
"""
@concrete struct PrecomputedCache{T<:Real,U<:AbstractArray, M<:AbstractArray{T}, V<:AbstractVector{T}}
    ΨtΨ::M
    Ψty::U
    diag_ΨtΨ::V
end

"""
    FastARDRegressor{T<:Real}

Fast Automatic Relevance Determination for sparse Bayesian regression.
Solves Ψ * c = y where Ψ is a basis matrix, c are coefficients, and y are observations.
"""
@concrete mutable struct FastARDRegressor{T<:Real}
    # Model parameters
    coef::AbstractArray{T}
    alpha::AbstractArray{T}
    beta::T
    beta_prev::T
    active::BitVector
    sigma::AbstractArray{T}
    
    # Training options
    n_iter::Int
    tol::T
    verbose::Bool
    compute_score::Bool
    
    # Training history
    scores::Vector{T}
    converged::Bool
    
    # Regularization parameters
    lambda_reg::T
    min_beta::T
    max_alpha::T
end

"""
    FastARDRegressor(T::Type{<:Real}=Float64; kwargs...)

Create a new FastARD regressor for sparse Bayesian regression.

# Arguments
- `T::Type{<:Real}=Float64`: Numeric type for computations (Float64, Float32, etc.)

# Keyword Arguments
- `n_iter::Int=200`: Maximum number of iterations
- `tol::Real=1e-6`: Convergence tolerance
- `verbose::Bool=true`: Whether to print progress information
- `compute_score::Bool=false`: Whether to compute and store log marginal likelihood scores
- `lambda_reg::Real=1e-8`: Regularization parameter for numerical stability
- `min_beta::Real=1e-6`: Minimum value for noise precision β
- `max_alpha::Real=1e12`: Maximum value for precision parameters α

# Returns
- `FastARDRegressor{T}`: Initialized model ready for fitting

# Examples
```julia
# Create a basic model
model = FastARDRegressor()

# Create with custom parameters
model = FastARDRegressor(Float32; n_iter=100, verbose=false)

# Create with high precision
using MultiFloats
model = FastARDRegressor(MultiFloat{Float64,4})
```
"""
function FastARDRegressor(T::Type{<:Real}=Float64; 
                         n_iter::Int=200, 
                         tol::Real=1e-6, 
                         verbose::Bool=true, 
                         compute_score::Bool=false,
                         lambda_reg::Real=1e-6,
                         min_beta::Real=1e-6,
                         max_alpha::Real=1e8)
    
    return FastARDRegressor{T}(
        Vector{T}(), Vector{T}(), one(T), one(T), BitVector(),
        Matrix{T}(undef, 0, 0), n_iter, T(tol), verbose,
        compute_score, Vector{T}(), false, T(lambda_reg), T(min_beta), T(max_alpha)
    )
end

# ============================================================================
# Matrix Precomputation
# ============================================================================

@stable function precompute_matrices(Ψ::AbstractMatrix{T}, y::AbstractVector{T}) where T<:Real
    ΨtΨ = Ψ' * Ψ
    Ψty = Ψ' * y
    diag_ΨtΨ = diag(ΨtΨ)
    return PrecomputedCache(ΨtΨ, Ψty, diag_ΨtΨ)
end

# ============================================================================
# Initialization Logic
# ============================================================================

@stable function initialize_model_parameters!(model::FastARDRegressor{T}, n_features::Int, var_y::T) where T<:Real
    model.alpha = fill(one(T), n_features)
    model.active = falses(n_features)
    model.coef = zeros(T, n_features)
    empty!(model.scores)
    model.converged = false
    
    # Initialize beta robustly
    model.beta = var_y > eps(T) ? inv(var_y) : T(100)
    model.beta = max(model.beta, model.min_beta)
    model.beta_prev = model.beta
end

function initialize_first_active_feature!(model::FastARDRegressor{T}, 
                                                  cache::PrecomputedCache{T}, 
                                                  var_y::T) where T<:Real
    # Conservative initialization with higher threshold
    proj = cache.Ψty.^2 ./ (cache.diag_ΨtΨ .+ eps(T))
    threshold = var_y * T(3)
    
    valid_features = findall(>(threshold), proj)
    
    if isempty(valid_features)
        start_idx = argmax(proj)
        proj[start_idx] < var_y * T(0.1) && return  # Skip if too weak
    else
        start_idx = valid_features[argmax(view(proj, valid_features))]
    end
    
    model.active[start_idx] = true
    
    # Conservative alpha initialization
    if proj[start_idx] > var_y + eps(T)
        model.alpha[start_idx] = cache.diag_ΨtΨ[start_idx] / 
                                 max(proj[start_idx] - var_y, var_y * T(2))
    else
        model.alpha[start_idx] = T(10)
    end
    
    model.alpha[start_idx] = clamp(model.alpha[start_idx], one(T), T(1000))
end

# ============================================================================
# Posterior Computation
# ============================================================================

struct PosteriorResult{T<:Real}
    μ::Vector{T}
    Σ_diag::Vector{T}
    success::Bool
end

@stable function compute_posterior(ΨtΨ_active::AbstractMatrix{T},
                                   Ψty_active::AbstractVector{T},
                                   alpha_active::AbstractVector{T},
                                   beta::T,
                                   lambda_reg::T) where T<:Real
    
    Σ_inv = beta * ΨtΨ_active + Diagonal(alpha_active .+ lambda_reg)
    
    try
        # Primary path: Cholesky decomposition
        L = cholesky(Σ_inv).L
        z = L \ (beta * Ψty_active)
        μ = L' \ z
        
        L_inv = inv(L)
        Σ_diag = vec(sum(abs2, L_inv, dims=2))
        
        return PosteriorResult(μ, max.(Σ_diag, eps(T)), true)
        
    catch e
        @warn "Cholesky failed, using fallback: $e"
        
        try
            # Fallback: pseudo-inverse
            Σ = pinv(Σ_inv)
            μ = beta * Σ * Ψty_active
            Σ_diag = diag(Σ)
            
            return PosteriorResult(μ, max.(Σ_diag, eps(T)), false)
            
        catch
            # Last resort: heavy regularization
            @warn "Pseudo-inverse failed, using heavy regularization"
            n_active = length(alpha_active)
            I_reg = Diagonal(fill(lambda_reg * T(1000), n_active))
            Σ = inv(Σ_inv + I_reg)
            μ = beta * Σ * Ψty_active
            Σ_diag = diag(Σ)
            
            return PosteriorResult(μ, max.(Σ_diag, eps(T)), false)
        end
    end
end

# ============================================================================
# Sparsity and Quality Parameter Computation
# ============================================================================
@concrete struct SparsityQualityResult
    s
    q
    S
    Q
end

@stable function compute_sparsity_quality_params(cache::PrecomputedCache,
                                                 alpha::AbstractVector{T},
                                                 active::AbstractVector{Bool},
                                                 μ_active::AbstractVector{T},
                                                 Σ_diag::AbstractVector{T},
                                                 beta::T) where T<:Real
    n_features = length(alpha)
    active_idx = findall(active)
    
    # Initialize S and Q - convert to regular arrays for in-place operations
    S = beta * Vector(cache.diag_ΨtΨ)
    Q = beta * Vector(cache.Ψty)
    
    if !isempty(active_idx)
        # Woodbury identity updates
        ΨtΨ_cross = view(cache.ΨtΨ, :, active_idx)
        
        # Update Q
        mul!(Q, ΨtΨ_cross, μ_active, -beta, one(T))
        
        # Update S efficiently
        @inbounds for i in 1:n_features
            row_i = view(ΨtΨ_cross, i, :)
            adjustment = beta^2 * dot(row_i, Σ_diag .* row_i)
            S[i] -= adjustment
        end
    end
    
    # Ensure numerical stability
    min_val = eps(T) * max(maximum(abs, S; init=one(T)), maximum(abs, Q; init=one(T)))
    clamp!(S, min_val, typemax(T))
    
    # Compute s and q with stability
    s = similar(S)
    q = similar(Q)
    
    @inbounds for i in 1:n_features
        if active[i] && isfinite(alpha[i]) && alpha[i] > zero(T)
            denominator = alpha[i] - S[i]
            threshold = min_val * max(abs(alpha[i]), abs(S[i]), one(T))
            
            if abs(denominator) > threshold
                s[i] = alpha[i] * S[i] / denominator
                q[i] = alpha[i] * Q[i] / denominator
            else
                s[i] = S[i]
                q[i] = Q[i]
            end
        else
            s[i] = S[i]
            q[i] = Q[i]
        end
    end
    
    return SparsityQualityResult(s, q, S, Q)
end

# ============================================================================
# Noise Precision Update
# ============================================================================

@stable function update_noise_precision!(model::FastARDRegressor{T},
                                         Ψ_active::AbstractMatrix{T},
                                         y::AbstractVector{T},
                                         μ_active::Vector{T},
                                         alpha_active::AbstractVector{T},
                                         Σ_diag::Vector{T},
                                         n_samples::Int) where T<:Real
    
    residual = y - Ψ_active * μ_active
    rss = dot(residual, residual)
    
    n_active = sum(model.active)
    numerator = T(n_samples - n_active) + dot(alpha_active, Σ_diag)
    denominator = rss + T(2) * eps(T)
    
    model.beta = numerator / denominator
    model.beta = max(model.beta, model.min_beta)
end

# ============================================================================
# Feature Update Logic
# ============================================================================

@stable function compute_feature_theta(q::AbstractVector{T}, s::AbstractVector{T}) where T<:Real
    return q.^2 .- s
end

@stable function classify_features(theta::AbstractVector{T}, active::BitVector) where T<:Real
    add = (theta .> zero(T)) .& (.!active)
    recompute = (theta .> zero(T)) .& active
    delete = .!(add .| recompute)
    return add, recompute, delete
end

@stable function compute_delta_marginal_likelihood!(deltaL::Vector{T},
                                                   theta::Vector{T},
                                                   s::Vector{T}, q::Vector{T},
                                                   S::Vector{T}, Q::Vector{T},
                                                   alpha::Vector{T},
                                                   active::BitVector,
                                                   add_mask::BitVector,
                                                   recompute_mask::BitVector,
                                                   delete_mask::BitVector,
                                                   max_alpha::T) where T<:Real
    fill!(deltaL, zero(T))
    
    # Add features
    @inbounds for i in findall(add_mask)
        if S[i] > eps(T) && Q[i]^2 > eps(T)
            ratio = S[i] / Q[i]^2
            if zero(T) < ratio < one(T)
                deltaL[i] = (Q[i]^2 - S[i]) / S[i] + log(ratio)
            end
        end
    end
    
    # Recompute features
    @inbounds for i in findall(recompute_mask)
        if theta[i] > eps(T) && s[i] > zero(T)
            alpha_new = s[i]^2 / theta[i]
            if eps(T) < alpha_new < max_alpha && alpha[i] > eps(T) && isfinite(alpha[i])
                delta_alpha = inv(alpha_new) - inv(alpha[i])
                log_arg = one(T) + S[i] * delta_alpha
                denominator = S[i] + inv(delta_alpha)
                
                if log_arg > eps(T) && abs(denominator) > eps(T)
                    deltaL[i] = Q[i]^2 / denominator - log(log_arg)
                end
            end
        end
    end
    
    # Delete features
    @inbounds for i in findall(delete_mask)
        if (active[i] && isfinite(alpha[i]) && alpha[i] > eps(T) &&
            isfinite(Q[i]) && isfinite(S[i]))
            
            denominator = S[i] - alpha[i]
            ratio = S[i] / alpha[i]
            log_arg = one(T) - ratio
            
            if (abs(denominator) > eps(T) && ratio < one(T) - eps(T) && log_arg > eps(T))
                deltaL[i] = Q[i]^2 / denominator - log(log_arg)
            end
        end
    end
end

function find_best_feature_update(deltaL::AbstractVector{T}) where T<:Real
    valid_idx = findall(isfinite.(deltaL))
    isempty(valid_idx) && return nothing
    
    # More conservative threshold - require meaningful improvement
    min_improvement = max(T(1e-6), T(100) * eps(T))
    significant_idx = findall(deltaL[valid_idx] .> min_improvement)
    isempty(significant_idx) && return nothing
    
    # Among significant improvements, pick the best
    best_local_idx = argmax(view(deltaL, valid_idx[significant_idx]))
    return valid_idx[significant_idx[best_local_idx]]
end

function apply_feature_update!(alpha::Vector{T}, active::BitVector,
                                      feature_idx::Int, theta::Vector{T}, s::Vector{T},
                                      max_alpha::T, clf_bias::Bool) where T<:Real
    if theta[feature_idx] > eps(T)
        # Add or update feature
        alpha_new = clamp(s[feature_idx]^2 / theta[feature_idx], eps(T), max_alpha)
        alpha[feature_idx] = alpha_new
        active[feature_idx] = true
    elseif active[feature_idx] && sum(active) > 1
        # Delete feature (protect bias for classification)
        if !(feature_idx == 1 && clf_bias)
            active[feature_idx] = false
            alpha[feature_idx] = max_alpha
        end
    end
end

function check_precision_convergence(theta::Vector{T}, s::Vector{T},
                                            alpha::Vector{T}, active::BitVector,
                                            tol::T, max_alpha::T) where T<:Real
    @inbounds for i in findall((theta .> eps(T)) .& active)
        alpha_new = clamp(s[i]^2 / theta[i], eps(T), max_alpha)
        if abs(alpha_new - alpha[i]) > tol * max(alpha_new, alpha[i])
            return false
        end
    end
    return true
end

@stable function update_precision_parameters!(model::FastARDRegressor{T},
                                             sq_result::SparsityQualityResult,
                                             n_samples::Int, clf_bias::Bool) where T<:Real
    
    theta = compute_feature_theta(sq_result.q, sq_result.s)
    add_mask, recompute_mask, delete_mask = classify_features(theta, model.active)
    
    # Adaptive feature limit based on sample size
    max_features = min(n_samples ÷ 3, length(model.alpha) ÷ 2)
    current_features = sum(model.active)
    
    # If we're at the feature limit, only allow deletions and recomputations
    if current_features >= max_features
        add_mask .= false
        if model.verbose && any(add_mask)
            @warn "Feature limit reached ($current_features/$max_features), only deletions allowed"
        end
    end
    
    deltaL = zeros(T, length(model.alpha))
    
    compute_delta_marginal_likelihood!(deltaL, theta, sq_result.s, sq_result.q,
                                      sq_result.S, sq_result.Q, model.alpha,
                                      model.active, add_mask, recompute_mask,
                                      delete_mask, model.max_alpha)
    
    deltaL ./= T(n_samples)
    
    feature_idx = find_best_feature_update(deltaL)
    
    if isnothing(feature_idx)
        return check_precision_convergence(theta, sq_result.s, model.alpha,
                                         model.active, model.tol, model.max_alpha)
    end
    
    apply_feature_update!(model.alpha, model.active, feature_idx, theta,
                         sq_result.s, model.max_alpha, clf_bias)
    
    return false  # Not converged
end

# ============================================================================
# Convergence Checking
# ============================================================================

@stable function check_beta_convergence(model::FastARDRegressor{T}) where T<:Real
    β_change = abs(model.beta - model.beta_prev)
    β_tolerance = model.tol * max(abs(model.beta), abs(model.beta_prev), one(T))
    
    converged = β_change < β_tolerance
    model.beta_prev = model.beta
    return converged
end

# ============================================================================
# Main Iteration Logic
# ============================================================================

@stable function perform_single_iteration!(model::FastARDRegressor{T},
                                          Ψ::AbstractMatrix{T},
                                          y::AbstractVector{T},
                                          cache::PrecomputedCache{T},
                                          iter::Int, n_samples::Int) where T<:Real
    
    active_idx = findall(model.active)
    isempty(active_idx) && return false
    
    # Extract active submatrices (using views for efficiency)
    Ψ_active = view(Ψ, :, active_idx)
    ΨtΨ_active = view(cache.ΨtΨ, active_idx, active_idx)
    Ψty_active = view(cache.Ψty, active_idx)
    alpha_active = view(model.alpha, active_idx)
    
    # Compute posterior
    posterior = compute_posterior(ΨtΨ_active, Ψty_active, alpha_active, 
                                 model.beta, model.lambda_reg)
    
    # Update coefficients
    fill!(model.coef, zero(T))
    model.coef[active_idx] .= posterior.μ
    
    # Compute sparsity and quality parameters
    sq_result = compute_sparsity_quality_params(cache, model.alpha, model.active,
                                               posterior.μ, posterior.Σ_diag, model.beta)
    
    # Update noise precision
    update_noise_precision!(model, Ψ_active, y, posterior.μ, alpha_active,
                           posterior.Σ_diag, n_samples)
    
    # Update precision parameters
    converged = update_precision_parameters!(model, sq_result, n_samples, false)
    
    # Logging
    if model.verbose && (iter % 10 == 0 || iter <= 5)
        println("Iteration $iter: active = $(sum(model.active)), β = $(model.beta)")
    end
    
    # Compute score if requested
    if model.compute_score
        score = compute_log_marginal_likelihood(Ψ_active, y, alpha_active,
                                               model.beta, posterior.μ,
                                               posterior.Σ_diag, model.lambda_reg)
        push!(model.scores, score)
    end
    
    return !converged
end

function finalize_model!(model::FastARDRegressor{T}, cache::PrecomputedCache{T}) where T<:Real
    active_idx = findall(model.active)
    if !isempty(active_idx)
        ΨtΨ_active = cache.ΨtΨ[active_idx, active_idx]
        alpha_active = model.alpha[active_idx]
        
        # More robust regularization for covariance computation
        n_active = length(active_idx)
        base_reg = max(model.lambda_reg, T(1e-8))
        
        # Add stronger regularization if too many features are selected
        if n_active > 20  # Too many features - strengthen regularization
            extra_reg = T(n_active) * base_reg * T(100)
            model.verbose && @warn "Many features selected ($n_active), using stronger regularization"
        else
            extra_reg = base_reg * T(10)
        end
        
        Σ_inv = model.beta * ΨtΨ_active + Diagonal(alpha_active .+ extra_reg)
        
        # Check condition number for numerical stability
        try
            cond_num = cond(Σ_inv)
            if cond_num > T(1e12)
                model.verbose && @warn "Ill-conditioned posterior covariance (cond=$cond_num), adding regularization"
                Σ_inv += T(cond_num * 1e-14) * I
            end
            
            model.sigma = inv(Σ_inv)
            
            # Clamp diagonal entries to prevent extreme values
            sigma_diag = diag(model.sigma)
            max_var = T(1000) / model.beta  # Reasonable maximum variance
            if any(sigma_diag .> max_var)
                model.verbose && @warn "Large posterior variances detected, clamping values"
                # Use diagonal approximation for stability
                model.sigma = Diagonal(min.(sigma_diag, max_var))
            end
            
        catch e
            model.verbose && @warn "Posterior computation failed ($e), using regularized diagonal approximation"
            # Fallback to diagonal approximation
            diag_entries = T(1) ./ (model.beta * diag(ΨtΨ_active) .+ alpha_active .+ extra_reg)
            model.sigma = Diagonal(diag_entries)
        end
    else
        model.sigma = Matrix{T}(undef, 0, 0)
    end
end

# ============================================================================
# Main Fitting Function
# ============================================================================

"""
    fit!(model::FastARDRegressor, Ψ::AbstractMatrix{T}, y::AbstractVecOrMat{T}) where T<:Real

Fit the FastARD model using Sequential Sparse Bayes algorithm.
Handles both vector and matrix observations (multiple outputs).
"""
function fit!(model::FastARDRegressor{T}, 
              Ψ::AbstractMatrix{T}, 
              y::AbstractVecOrMat{T}) where T<:Real
    
    # Handle matrix y by fitting each column separately
    if y isa AbstractMatrix
        n_outputs = size(y, 2)
        models = Vector{typeof(model)}(undef, n_outputs)
        
        for i in 1:n_outputs
            if model.verbose
                println("Fitting output $i/$n_outputs")
            end
            model_copy = deepcopy(model)
            models[i] = fit!(model_copy, Ψ, view(y, :, i))
        end
        return models
    end
    
    # Main fitting routine for vector y
    n_samples, n_features = size(Ψ)
    
    # Initialize model parameters
    var_y = var(y)
    initialize_model_parameters!(model, n_features, var_y)
    
    # Precompute matrices
    cache = precompute_matrices(Ψ, y)
    
    # Initialize first active feature
    initialize_first_active_feature!(model, cache, var_y)
    
    # Main iteration loop
    for iter in 1:model.n_iter
        continue_iteration = perform_single_iteration!(model, Ψ, y, cache, iter, n_samples)
        
        !continue_iteration && break
        
        # Check for early convergence
        if iter > 10 && check_beta_convergence(model)
            model.converged = true
            model.verbose && println("Early convergence at iteration $iter")
            break
        end
    end
    
    # Finalize model
    finalize_model!(model, cache)
    
    return model
end

# ============================================================================
# Prediction Functions
# ============================================================================

"""
    predict(model::FastARDRegressor, Ψ_test::AbstractMatrix{T}) where T<:Real

Make predictions on test data.
"""
function predict(model::FastARDRegressor{T}, Ψ_test::AbstractMatrix{T}) where T<:Real
    active_idx = findall(model.active)
    isempty(active_idx) && return zeros(T, size(Ψ_test, 1))
    
    Ψ_test_active = view(Ψ_test, :, active_idx)
    return Ψ_test_active * view(model.coef, active_idx)
end

"""
    predict_with_uncertainty(model::FastARDRegressor, Ψ_test::AbstractMatrix{T}) where T<:Real

Make predictions with uncertainty estimates.
Returns (y_pred, y_std).
"""
function predict_with_uncertainty(model::FastARDRegressor{T}, 
                                 Ψ_test::AbstractMatrix{T}) where T<:Real
    active_idx = findall(model.active)
    n_test = size(Ψ_test, 1)
    
    if isempty(active_idx)
        y_pred = zeros(T, n_test)
        y_std = fill(safe_sqrt(inv(max(model.beta, eps(T)))), n_test)
        return y_pred, y_std
    end
    
    Ψ_test_active = view(Ψ_test, :, active_idx)
    y_pred = Ψ_test_active * view(model.coef, active_idx)
    
    # Predictive variance with numerical stability
    var_noise = inv(max(model.beta, eps(T)))
    
    if !isempty(model.sigma)
        # Correct predictive variance: diag(Ψ * Σ * Ψ')
        var_param = zeros(T, n_test)
        for i in 1:n_test
            ψ_i = view(Ψ_test_active, i, :)
            var_param[i] = dot(ψ_i, model.sigma * ψ_i)
        end
        clamp!(var_param, zero(T), typemax(T))
        y_std = @. safe_sqrt(var_noise + var_param)
    else
        y_std = fill(safe_sqrt(var_noise), n_test)
    end
    
    return y_pred, y_std
end

"""
    get_active_coefficients(model::FastARDRegressor)

Get the indices and values of non-zero coefficients.
"""
function get_active_coefficients(model::FastARDRegressor)
    active_idx = findall(model.active)
    return active_idx, view(model.coef, active_idx)
end

# ============================================================================
# Log Marginal Likelihood
# ============================================================================


@stable function compute_log_marginal_likelihood(Ψ_active::AbstractArray{T},
                                                y::AbstractArray{T},
                                                alpha_active::AbstractArray{T},
                                                beta::U, μ_active::AbstractArray{T},
                                                Σ_diag::AbstractArray{T},
                                                lambda_reg::V) where {T<:Real,V<:Real,U<:Real}
    n_samples = length(y)
    n_active = length(alpha_active)
    
    n_active == 0 && return -T(Inf)

    # Residual sum of squares
    residual = y - Ψ_active * μ_active
    rss = dot(residual, residual)
    
    # Log determinant computation
    Σ_inv = beta * (Ψ_active' * Ψ_active) + Diagonal(alpha_active .+ lambda_reg)

    log_det = try
        L = cholesky(Σ_inv).L
        T(2) * sum(log ∘ abs, diag(L))
    catch
        # Fallback using eigenvalues
        eigenvals_matrix = eigvals(Symmetric(Matrix(Σ_inv)))
        eigenvals_safe = max.(eigenvals_matrix, lambda_reg)
        sum(log, eigenvals_safe)
    end
    
    # Ensure alpha_active is positive for log
    alpha_safe = max.(alpha_active, lambda_reg)
    
    # Log marginal likelihood
    log_ml = -T(0.5) * (T(n_samples) * log(T(2π)) - 
                        T(n_samples) * log(max(beta, eps(T))) +
                        sum(log, alpha_safe) - log_det +
                        beta * rss + dot(μ_active, alpha_active .* μ_active))
    
    return isfinite(log_ml) ? log_ml : -T(Inf)
end

# ============================================================================
# Module Initialization
# ============================================================================

function __init__()
    # Enable high-precision transcendental functions for MultiFloat types
    # This is called after the module loads, avoiding precompilation issues
    MultiFloats.use_bigfloat_transcendentals()
end
	
end # module