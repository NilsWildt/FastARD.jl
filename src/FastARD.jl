module FastARD

using LinearAlgebra
using LinearAlgebra: I
using Statistics
using ConcreteStructs
using DispatchDoctor
using MuladdMacro
using MultiFloats
using GenericLinearAlgebra: eigvals
using TypeUtils
if Sys.isapple() && Sys.ARCH in (:aarch64, :arm64)
    @info "Using `AppleAccelerate.jl` for Apple Silicon."
    using AppleAccelerate
elseif Sys.ARCH == :x86_64 && occursin(r"intel"i, CPU_MODEL)
    @info "Detected Intel x86_64 CPU. Loading `MKL.jl`."
    using MKL
end

export FastARDRegressor, fit!, predict, get_active_coefficients, predict_with_uncertainty

# ============================================================================
# Utility Functions
# ============================================================================

@stable @muladd function safe_sqrt(x::T) where {T <: Real}
    x >= zero(T) ? sqrt(x) : sqrt(eps(T))
end

@inline Base.precision(::Type{MultiFloat{T, N}}) where {T, N} =
    N * precision(T) + (N - 1) # implicit bits of precision between limbs


# ============================================================================
# Core Data Structures
# ============================================================================

"""
    FastARDRegressor{T<:Real}

Fast Automatic Relevance Determination for sparse Bayesian regression.
Solves Ψ * c = y where Ψ is a basis matrix, c are coefficients, and y are observations.
"""
@concrete mutable struct FastARDRegressor{T <: Real}
    # Model parameters
    coef::Vector{T}
    alpha::Vector{T}
    beta::T
    active::BitVector
    sigma::Matrix{T}

    # Data centering and scaling (for proper prediction)
    X_mean::Matrix{T}
    X_std::Matrix{T}
    y_mean::T

    # Training options
    n_iter::Int
    tol::T
    verbose::Bool
    compute_score::Bool
    standardize::Bool

    # Training history
    scores::Vector{T}
    converged::Bool

    # Regularization parameters
    lambda_reg::T
    max_alpha::T
    min_beta::T
    max_beta::T
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
- `standardize::Bool=true`: Whether to standardize features (center and scale). Recommended for numerical stability unless data is already preprocessed
- `lambda_reg::Real=1e-8`: Regularization parameter for numerical stability
- `max_alpha::Real=1e12`: Maximum value for precision parameters α
- `min_beta::Real=1e-6`: Minimum value for noise precision β
- `max_beta::Real=1e6`: Maximum value for noise precision β

# Returns
- `FastARDRegressor{T}`: Initialized model ready for fitting

# Examples
```julia
# Create a basic model (standardization enabled by default)
model = FastARDRegressor()

# Create with custom parameters
model = FastARDRegressor(Float32; n_iter=100, verbose=false)

# Use raw data without centering/scaling (only if data is already preprocessed)
model = FastARDRegressor(standardize=false)

# Create with high precision
using MultiFloats
model = FastARDRegressor(MultiFloat{Float64,4})

# Create with custom beta bounds
model = FastARDRegressor(min_beta=1e-8, max_beta=1e8)
```
"""
function FastARDRegressor(
        T::Type{<:Real} = Float64;
        n_iter::Int = 300,
        tol::Real = 1.0e-6,
        verbose::Bool = false,
        compute_score::Bool = false,
        standardize::Bool = true,
        lambda_reg::Real = 1.0e-8,
        max_alpha::Real = 1.0e12,
        min_beta::Real = 1.0e-6,
        max_beta::Real = 1.0e6
    )

    return FastARDRegressor{T}(
        Vector{T}(), Vector{T}(), one(T), BitVector(),
        Matrix{T}(undef, 0, 0), Matrix{T}(undef, 1, 0), Matrix{T}(undef, 1, 0), zero(T),
        n_iter, T(tol), verbose, compute_score, standardize, Vector{T}(), false, 
        T(lambda_reg), T(max_alpha), T(min_beta), T(max_beta)
    )
end

# ============================================================================
# Type-stable helper functions
# ============================================================================

@stable @muladd function compute_posterior_cholesky(
        Σ_inv::AbstractMatrix{T},
        XYa::AbstractVector{T},
        beta::T
    ) where {T <: Real}
    # Primary path: Cholesky decomposition
    L = cholesky(Σ_inv).L
    z = L \ (beta * XYa)
    μ = L' \ z

    L_inv = inv(L)
    Σ_diag = vec(sum(abs2, L_inv, dims = 2))

    return μ, max.(Σ_diag, eps(T)), true
end


@stable @muladd function compute_posterior_fallback(
        Σ_inv::AbstractMatrix{T},
        XYa::AbstractVector{T},
        beta::T
    ) where {T <: Real}
    # For MultiFloat types, avoid SVD-based pinv and use regularized inverse
    if T <: MultiFloat
        # Add regularization and use direct inverse
        reg_amount = T(1.0e-8) * maximum(diag(Σ_inv))
        Σ_inv_reg = Σ_inv + reg_amount * I
        try
            Σ = inv(Σ_inv_reg)
            μ_a = beta * Σ * XYa
            Σ_diag = diag(Σ)
            return μ_a, max.(Σ_diag, eps(T)), false
        catch
            # Last resort: diagonal approximation
            μ_a = beta * XYa ./ diag(Σ_inv_reg)
            Σ_diag = inv.(diag(Σ_inv_reg))
            return μ_a, max.(Σ_diag, eps(T)), false
        end
    else
        # For standard Float types, use pinv
        try
            Σ = pinv(Σ_inv)
            μ_a = beta * Σ * XYa
            Σ_diag = diag(Σ)
            return μ_a, max.(Σ_diag, eps(T)), false
        catch
            # Fallback to diagonal approximation
            μ_a = beta * XYa ./ diag(Σ_inv)
            Σ_diag = inv.(diag(Σ_inv))
            return μ_a, max.(Σ_diag, eps(T)), false
        end
    end
end

@stable @muladd function update_sparsity_quality!(
        S::Vector{T}, Q::Vector{T},
        XX::AbstractMatrix{T},
        XXd::AbstractVector{T},
        XY::AbstractVector{T},
        active_idx::Vector{Int},
        μ_a::Vector{T},
        Σ_diag::Vector{T},
        beta::T
    ) where {T <: Real}
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
            adjustment = beta * beta * dot(row_i, Σ_diag .* row_i)
            S[i] -= adjustment
        end
    end

    # Ensure numerical stability
    clamp!(S, eps(T), typemax(T))
end

@stable @muladd function compute_delta_likelihood(
        Q::T, S::T, theta::T,
        alpha::T, is_active::Bool,
        max_alpha::T
    ) where {T <: Real}
    if !is_active && theta > zero(T)
        # Add feature
        if S > eps(T) && Q * Q > eps(T)
            return (Q * Q - S) / S + log(S / (Q * Q))
        end
    elseif is_active && theta > zero(T)
        # Recompute feature
        if theta > eps(T)
            alpha_new = S * S / theta
            if eps(T) < alpha_new < max_alpha
                delta_alpha_inv = inv(alpha_new) - inv(alpha)
                if abs(delta_alpha_inv) > eps(T)
                    return Q * Q / (S + inv(delta_alpha_inv)) - log(one(T) + S * delta_alpha_inv)
                end
            end
        end
    elseif is_active && theta <= zero(T)
        # Delete feature - only if S < alpha to ensure log argument is positive
        if alpha < max_alpha && S < alpha
            ratio = S / alpha
            if ratio < one(T) - eps(T)  # Ensure log argument is positive
                return Q * Q / (S - alpha) - log(one(T) - ratio)
            end
        end
    end

    return zero(T)
end


# ============================================================================
# Main Fitting Function (not type-stable due to dynamic nature)
# ============================================================================

@muladd function fit!(
        model::FastARDRegressor{T},
        X::AbstractMatrix{T},
        y::AbstractVecOrMat{T}
    ) where {T <: Real}  # Note: AbstractVecOrMat

    # Handle matrix y by fitting each column separately
    if y isa AbstractMatrix
        n_outputs = size(y, 2)
        models = Vector{typeof(model)}(undef, n_outputs)

        for i in 1:n_outputs
            if model.verbose
                println("Fitting output $i/$n_outputs")
            end
            model_copy = deepcopy(model)
            models[i] = fit!(model_copy, X, view(y, :, i))
        end
        return models
    end


    n_samples, n_features = size(X)

    # Optionally standardize features (center and scale)
    if model.standardize
        model.X_mean = mean(X, dims = 1)
        model.y_mean = mean(y)
        X_processed = X .- model.X_mean
        y_processed = y .- model.y_mean
        
        model.X_std = std(X_processed, dims = 1)
        # Handle zero variance features by setting std to 1 (they remain constant)
        model.X_std = map(s -> s < eps(T) ? one(T) : s, model.X_std)
        X_processed = X_processed ./ model.X_std
    else
        # Use raw data as-is
        model.X_mean = zeros(T, 1, n_features)
        model.X_std = ones(T, 1, n_features)
        model.y_mean = zero(T)
        X_processed = X
        y_processed = y
    end

    # Precompute matrices
    XX = X_processed' * X_processed
    XY = X_processed' * y_processed
    XXd = diag(XX)

    # Initialize model
    var_y = var(y_processed)
    model.beta = var_y > eps(T) ? inv(var_y) : T(10)
    model.alpha = fill(model.max_alpha, n_features)
    model.active = falses(n_features)
    model.coef = zeros(T, n_features)
    empty!(model.scores)
    model.converged = false

    # Initialize first feature
    proj = @. XY * XY / (XXd + eps(T))
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
        X_active = view(X_processed, :, active_idx)
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
        residual = y_processed - X_active * μ_a
        rss = dot(residual, residual)
        n_active = sum(model.active)
        model.beta = (n_samples - n_active + dot(alpha_a, Σ_diag)) / (rss + eps(T))
        model.beta = clamp(model.beta, model.min_beta, model.max_beta)

        if model.compute_score
            score = compute_log_marginal_likelihood(
                X_active, y_processed,
                collect(alpha_a), model.beta,
                μ_a, Σ_diag
            )
            push!(model.scores, score)
        end

        # Compute feature updates
        theta = @. q * q - s

        # Compute change in marginal likelihood
        fill!(deltaL, zero(T))
        @inbounds for i in 1:n_features
            deltaL[i] = compute_delta_likelihood(
                Q[i], S[i], theta[i],
                model.alpha[i], model.active[i],
                model.max_alpha
            )
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
            model.alpha[feature_idx] = clamp(
                s[feature_idx] * s[feature_idx] / theta[feature_idx],
                eps(T), model.max_alpha
            )
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

@muladd function predict(model::FastARDRegressor{T}, X::AbstractMatrix{T}) where {T <: Real}
    active_idx = findall(model.active)
    isempty(active_idx) && return fill(model.y_mean, size(X, 1))

    # Apply same preprocessing as during training
    X_processed = if model.standardize
        (X .- model.X_mean) ./ model.X_std
    else
        X  # Use raw data as-is
    end
    X_active = view(X_processed, :, active_idx)
    coef_active = view(model.coef, active_idx)
    
    # Add back the y_mean to get predictions on original scale
    return X_active * coef_active .+ model.y_mean
end

@muladd function predict_with_uncertainty(
        model::FastARDRegressor{T},
        X::AbstractMatrix{T}
    ) where {T <: Real}
    active_idx = findall(model.active)
    n_test = size(X, 1)

    if isempty(active_idx)
        y_pred = fill(model.y_mean, n_test)
        y_std = fill(safe_sqrt(inv(model.beta)), n_test)
        return y_pred, y_std
    end

    # Apply same preprocessing as during training
    X_processed = if model.standardize
        (X .- model.X_mean) ./ model.X_std
    else
        X  # Use raw data as-is
    end
    X_active = view(X_processed, :, active_idx)
    coef_active = view(model.coef, active_idx)
    
    # Predictions on processed scale, then add back y_mean
    y_pred_processed = X_active * coef_active
    y_pred = y_pred_processed .+ model.y_mean

    # Predictive variance (uncertainty doesn't change with centering/scaling)
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

# ============================================================================
# Log Marginal Likelihood
# ============================================================================

# ============================================================================
# Log Marginal Likelihood
# ============================================================================

@muladd function compute_log_marginal_likelihood(
        X_active::AbstractMatrix{T},
        y::AbstractVector{T},
        alpha_active::AbstractVector{T},
        beta::T, μ_active::AbstractVector{T},
        Σ_diag::AbstractVector{T}
    ) where {T <: Real}
    n_samples = length(y)
    n_active = length(alpha_active)

    n_active == 0 && return -T(Inf)

    # Residual sum of squares
    residual = y - X_active * μ_active
    rss = dot(residual, residual)

    # Log determinant of posterior precision
    Σ_inv = beta * (X_active' * X_active) + Diagonal(alpha_active)

    log_det = try
        L = cholesky(Σ_inv).L
        2 * sum(log, diag(L))
    catch
        # Fallback using eigenvalues
        eigenvals = eigvals(Symmetric(Matrix(Σ_inv)))
        sum(log, max.(eigenvals, eps(T)))
    end

    # Log marginal likelihood
    log_ml = -T(0.5) * (
        n_samples * log(2π) -
            n_samples * log(max(beta, eps(T))) +
            sum(log, max.(alpha_active, eps(T))) - log_det +
            beta * rss + dot(μ_active, alpha_active .* μ_active)
    )

    return isfinite(log_ml) ? log_ml : -T(Inf)
end


function __init__()
    # Enable high-precision transcendental functions for MultiFloat types -- needed for log(MultiFloat)
    return MultiFloats.use_bigfloat_transcendentals()
end

end # module
