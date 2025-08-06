module FastARD

using LinearAlgebra
using Statistics
using ConcreteStructs
using DispatchDoctor

export FastARDRegressor, fit!, predict, get_active_coefficients, predict_with_uncertainty

# ============================================================================
# Utility Functions
# ============================================================================

@stable function safe_sqrt(x::T) where T<:Real
    x >= zero(T) ? sqrt(x) : sqrt(eps(T))
end

# ============================================================================
# Core Data Structures
# ============================================================================

@concrete mutable struct FastARDRegressor{T<:Real}
    # Model parameters
    coef::Vector{T}
    alpha::Vector{T}
    beta::T
    active::BitVector
    sigma::Matrix{T}
    
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
    max_alpha::T
end

function FastARDRegressor(T::Type{<:Real}=Float64; 
                         n_iter::Int=300, 
                         tol::Real=1e-3, 
                         verbose::Bool=false, 
                         compute_score::Bool=false,
                         lambda_reg::Real=1e-8,
                         max_alpha::Real=1e12)
    
    return FastARDRegressor{T}(
        Vector{T}(), Vector{T}(), one(T), BitVector(),
        Matrix{T}(undef, 0, 0), n_iter, T(tol), verbose,
        compute_score, Vector{T}(), false, T(lambda_reg), T(max_alpha)
    )
end

# ============================================================================
# Type-stable helper functions
# ============================================================================

@stable function compute_posterior_cholesky(Σ_inv::AbstractMatrix{T}, 
                                           XYa::AbstractVector{T},
                                           beta::T) where T<:Real
    L = cholesky(Σ_inv).L
    z = L \ (beta * XYa)
    μ_a = L' \ z
    
    # Compute diagonal of covariance efficiently
    L_inv = inv(L)
    Σ_diag = vec(sum(abs2, L_inv, dims=2))
    
    return μ_a, Σ_diag, true
end

@stable function compute_posterior_fallback(Σ_inv::AbstractMatrix{T}, 
                                           XYa::AbstractVector{T},
                                           beta::T) where T<:Real
    Σ = pinv(Σ_inv)
    μ_a = beta * Σ * XYa
    Σ_diag = diag(Σ)
    return μ_a, Σ_diag, false
end

@stable function update_sparsity_quality!(S::Vector{T}, Q::Vector{T},
                                         XX::AbstractMatrix{T},
                                         XXd::AbstractVector{T},
                                         XY::AbstractVector{T},
                                         active_idx::Vector{Int},
                                         μ_a::Vector{T},
                                         Σ_diag::Vector{T},
                                         beta::T) where T<:Real
    n_features = length(S)
    
    # Initialize with basic values
    S .= beta .* XXd
    Q .= beta .* XY
    
    if !isempty(active_idx)
        # Woodbury identity updates using views
        XXcross = view(XX, :, active_idx)
        
        # Update Q efficiently
        mul!(Q, XXcross, μ_a, -beta, one(T))
        
        # Update S efficiently
        @inbounds for i in 1:n_features
            row_i = view(XXcross, i, :)
            adjustment = beta^2 * dot(row_i, Σ_diag .* row_i)
            S[i] -= adjustment
        end
    end
    
    # Ensure numerical stability
    clamp!(S, eps(T), typemax(T))
end

@stable function compute_delta_likelihood(Q::T, S::T, theta::T, 
                                         alpha::T, is_active::Bool,
                                         max_alpha::T) where T<:Real
    if !is_active && theta > zero(T)
        # Add feature
        if S > eps(T) && Q^2 > eps(T)
            return (Q^2 - S) / S + log(S / Q^2)
        end
    elseif is_active && theta > zero(T)
        # Recompute feature
        if theta > eps(T)
            alpha_new = S^2 / theta
            if eps(T) < alpha_new < max_alpha
                delta_alpha_inv = inv(alpha_new) - inv(alpha)
                if abs(delta_alpha_inv) > eps(T)
                    return Q^2 / (S + inv(delta_alpha_inv)) - log(one(T) + S * delta_alpha_inv)
                end
            end
        end
    elseif is_active && theta <= zero(T)
        # Delete feature
        if alpha < max_alpha
            return Q^2 / (S - alpha) - log(one(T) - S/alpha)
        end
    end
    
    return zero(T)
end

# ============================================================================
# Main Fitting Function (not type-stable due to dynamic nature)
# ============================================================================

function fit!(model::FastARDRegressor{T}, X::AbstractMatrix{T}, y::AbstractVector{T}) where T<:Real
    n_samples, n_features = size(X)
    
    # Center data
    X_mean = mean(X, dims=1)
    y_mean = mean(y)
    X_centered = X .- X_mean
    y_centered = y .- y_mean
    
    # Precompute matrices
    XX = X_centered' * X_centered
    XY = X_centered' * y_centered
    XXd = diag(XX)
    
    # Initialize model
    var_y = var(y_centered)
    model.beta = var_y > eps(T) ? inv(var_y) : T(10)
    model.alpha = fill(model.max_alpha, n_features)
    model.active = falses(n_features)
    model.coef = zeros(T, n_features)
    empty!(model.scores)
    model.converged = false
    
    # Initialize first feature
    proj = @. XY^2 / (XXd + eps(T))
    start_idx = argmax(proj)
    model.active[start_idx] = true
    model.alpha[start_idx] = XXd[start_idx] / max(proj[start_idx] - var_y, eps(T))
    
    # Pre-allocate work arrays
    S = zeros(T, n_features)
    Q = zeros(T, n_features)
    s = zeros(T, n_features)
    q = zeros(T, n_features)
    deltaL = zeros(T, n_features)
    
    # Main iteration loop
    for iter in 1:model.n_iter
        active_idx = findall(model.active)
        isempty(active_idx) && break
        
        # Extract active submatrices using views
        XXa = view(XX, active_idx, active_idx)
        XYa = view(XY, active_idx)
        X_active = view(X_centered, :, active_idx)
        alpha_a = view(model.alpha, active_idx)
        
        # Compute posterior
        Σ_inv = model.beta * XXa + Diagonal(alpha_a) + model.lambda_reg * I
        
        local μ_a::Vector{T}, Σ_diag::Vector{T}
        try
            μ_a, Σ_diag, _ = compute_posterior_cholesky(Σ_inv, XYa, model.beta)
        catch
            μ_a, Σ_diag, _ = compute_posterior_fallback(Σ_inv, XYa, model.beta)
        end
        
        # Update coefficients
        fill!(model.coef, zero(T))
        @inbounds for (i, idx) in enumerate(active_idx)
            model.coef[idx] = μ_a[i]
        end
        
        # Update S and Q
        update_sparsity_quality!(S, Q, XX, XXd, XY, active_idx, μ_a, Σ_diag, model.beta)
        
        # Compute s and q
        s .= S
        q .= Q
        
        @inbounds for i in active_idx
            if model.alpha[i] < model.max_alpha
                denom = model.alpha[i] - S[i]
                if abs(denom) > eps(T) * max(model.alpha[i], S[i])
                    s[i] = model.alpha[i] * S[i] / denom
                    q[i] = model.alpha[i] * Q[i] / denom
                end
            end
        end
        
        # Update noise precision
        residual = y_centered - X_active * μ_a
        rss = dot(residual, residual)
        n_active = sum(model.active)
        model.beta = (n_samples - n_active + dot(alpha_a, Σ_diag)) / (rss + eps(T))
        model.beta = clamp(model.beta, T(1e-6), T(1e6))
        
        # Compute feature updates
        theta = @. q^2 - s
        
        # Compute change in marginal likelihood
        fill!(deltaL, zero(T))
        @inbounds for i in 1:n_features
            deltaL[i] = compute_delta_likelihood(Q[i], S[i], theta[i], 
                                                model.alpha[i], model.active[i],
                                                model.max_alpha)
        end
        
        deltaL ./= T(n_samples)
        
        # Find best feature update
        valid_idx = findall(isfinite.(deltaL) .& (deltaL .> eps(T)))
        
        if isempty(valid_idx)
            model.converged = true
            model.verbose && println("Converged at iteration $iter")
            break
        end
        
        feature_idx = valid_idx[argmax(view(deltaL, valid_idx))]
        
        # Apply update
        if theta[feature_idx] > zero(T)
            model.alpha[feature_idx] = clamp(s[feature_idx]^2 / theta[feature_idx], 
                                            eps(T), model.max_alpha)
            model.active[feature_idx] = true
        elseif model.active[feature_idx] && sum(model.active) > 1
            model.active[feature_idx] = false
            model.alpha[feature_idx] = model.max_alpha
        end
        
        # Logging
        if model.verbose && (iter % 10 == 0 || iter <= 5)
            println("Iteration $iter: active = $(sum(model.active)), β = $(model.beta)")
        end
    end
    
    # Finalize model
    active_idx = findall(model.active)
    if !isempty(active_idx)
        XXa = XX[active_idx, active_idx]
        alpha_a = model.alpha[active_idx]
        Σ_inv = model.beta * XXa + Diagonal(alpha_a) + model.lambda_reg * I
        
        try
            model.sigma = inv(Σ_inv)
        catch
            model.sigma = Diagonal(@. inv(model.beta * diag(XXa) + alpha_a))
        end
    else
        model.sigma = Matrix{T}(undef, 0, 0)
    end
    
    return model
end

# ============================================================================
# Prediction Functions (not type-stable due to dynamic arrays)
# ============================================================================

function predict(model::FastARDRegressor{T}, X::AbstractMatrix{T}) where T<:Real
    active_idx = findall(model.active)
    isempty(active_idx) && return zeros(T, size(X, 1))
    
    X_active = view(X, :, active_idx)
    coef_active = view(model.coef, active_idx)
    return X_active * coef_active
end

function predict_with_uncertainty(model::FastARDRegressor{T}, 
                                 X::AbstractMatrix{T}) where T<:Real
    active_idx = findall(model.active)
    n_test = size(X, 1)
    
    if isempty(active_idx)
        y_pred = zeros(T, n_test)
        y_std = fill(safe_sqrt(inv(model.beta)), n_test)
        return y_pred, y_std
    end
    
    X_active = view(X, :, active_idx)
    coef_active = view(model.coef, active_idx)
    y_pred = X_active * coef_active
    
    # Predictive variance
    var_noise = inv(model.beta)
    var_param = zeros(T, n_test)
    
    if !isempty(model.sigma)
        @inbounds for i in 1:n_test
            x_i = view(X_active, i, :)
            var_param[i] = dot(x_i, model.sigma * x_i)
        end
    end
    
    y_std = @. safe_sqrt(var_noise + var_param)
    return y_pred, y_std
end

function get_active_coefficients(model::FastARDRegressor)
    active_idx = findall(model.active)
    return active_idx, view(model.coef, active_idx)
end

end # module