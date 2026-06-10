module FastARD

using LinearAlgebra
using LinearAlgebra: I, mul!, dot, cholesky!, bunchkaufman!, Hermitian, issuccess,
    ldiv!, logabsdet
using Statistics
using ConcreteStructs
using DispatchDoctor
using MuladdMacro
using MultiFloats
using TypeUtils

# Optional BLAS backends ship as package extensions (see ext/): if the user's
# environment has AppleAccelerate (Apple Silicon) or MKL (Intel) installed, the
# corresponding extension loads it and swaps the BLAS backend as a side effect.
# FastARD itself does not force a backend, so it stays platform-portable.

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

Fast Automatic Relevance Determination for sparse Bayesian regression
(Tipping & Faul, 2003). Solves Ψ * c = y where Ψ is a basis matrix, c are
coefficients, and y are observations, using the fast sequential algorithm with
rank-1 covariance updates, deletion-priority basis selection and an alignment
(collinearity) guard.
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
- `n_iter::Int=1000`: Maximum number of sequential update iterations
- `tol::Real=1e-6`: Reserved for interface compatibility (the algorithm uses
  evidence-based stopping thresholds internally)
- `verbose::Bool=false`: Whether to print progress information
- `compute_score::Bool=false`: Whether to store the running log marginal likelihood
- `standardize::Bool=true`: Whether to center and scale features. Recommended for
  numerical stability unless data is already preprocessed
- `lambda_reg::Real=1e-8`: Ridge added to the precision matrix only if the
  Cholesky factorization is not positive definite (numerical safeguard)
- `max_alpha::Real=1e12`: Reported precision for inactive coefficients
- `min_beta::Real=1e-6`: Minimum value for noise precision β
- `max_beta::Real=1e6`: Maximum value for noise precision β

# Examples
```julia
model = FastARDRegressor()
model = FastARDRegressor(Float32; n_iter=100, verbose=false)
model = FastARDRegressor(standardize=false)
using MultiFloats
model = FastARDRegressor(MultiFloat{Float64,4})
```
"""
function FastARDRegressor(
        T::Type{<:Real} = Float64;
        n_iter::Int = 1000,
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
# Internal workspace (preallocated, concrete buffers — type-stable)
# ============================================================================

@concrete struct ARDWorkspace
    Ψ            # n_samples × n_features (internally column-scaled design)
    output_proj  # n_features            (Ψ' y)
    cross        # n_features × K_max    (Ψ' Ψ_active, columns ↔ active terms)
    sparseΨ      # n_samples × K_max     (Ψ_active)
    betaproj     # n_features × K_max    (β · cross)
    Cov          # K_max × K_max         (posterior covariance over active set)
    cholW        # K_max × K_max         (scratch for Cholesky)
    M            # n_features × K_max    (scratch: cross · Cov)
    s_in         # n_features
    q_in         # n_features
    s_out        # n_features
    q_out        # n_features
    theta        # n_features
    proj_nf      # n_features            (scratch)
    proj_nf2     # n_features            (scratch)
    cov_col      # K_max                 (scratch)
    cross_cov    # K_max                 (scratch)
    mean_change  # K_max                 (scratch)
    Gmean        # K_max                 (scratch)
end

function ARDWorkspace(Ψ::AbstractMatrix{T}, output_proj::Vector{T}, Kmax::Int) where {T}
    ns, nf = size(Ψ)
    return ARDWorkspace(
        Ψ, output_proj,
        Matrix{T}(undef, nf, Kmax),   # cross
        Matrix{T}(undef, ns, Kmax),   # sparseΨ
        Matrix{T}(undef, nf, Kmax),   # betaproj
        Matrix{T}(undef, Kmax, Kmax), # Cov
        Matrix{T}(undef, Kmax, Kmax), # cholW
        Matrix{T}(undef, nf, Kmax),   # M
        Vector{T}(undef, nf),         # s_in
        Vector{T}(undef, nf),         # q_in
        Vector{T}(undef, nf),         # s_out
        Vector{T}(undef, nf),         # q_out
        Vector{T}(undef, nf),         # theta
        Vector{T}(undef, nf),         # proj_nf
        Vector{T}(undef, nf),         # proj_nf2
        Vector{T}(undef, Kmax),       # cov_col
        Vector{T}(undef, Kmax),       # cross_cov
        Vector{T}(undef, Kmax),       # mean_change
        Vector{T}(undef, Kmax),       # Gmean
    )
end

# ============================================================================
# Linear algebra helpers (no try/catch: use Cholesky `check=false` + issuccess)
# ============================================================================

"""
    _spd_inverse!(Cov, ws, K, alpha, beta, active_idx, lambda_reg) -> logdetP

Compute `Cov[1:K,1:K] = inv(P)` for the active posterior precision
`P = β·G + diag(α)` (G the active Gram block of `ws.cross`), and return
`logdet(P)`. Uses an in-place Cholesky; on a non-PD matrix it retries with a
ridge then Bunch–Kaufman. Never throws.
"""
@muladd function _spd_inverse!(
        Cov, ws::ARDWorkspace, K::Int,
        alpha::AbstractVector{T}, beta::T,
        active_idx::AbstractVector{Int}, lambda_reg::T
    ) where {T <: Real}
    W = view(ws.cholW, 1:K, 1:K)
    G = view(ws.cross, view(active_idx, 1:K), 1:K)   # K×K active Gram block
    @. W = beta * G
    @inbounds for c in 1:K
        W[c, c] += alpha[c]
    end

    Σ = view(Cov, 1:K, 1:K)
    C = cholesky!(Hermitian(W, :U), check = false)
    if issuccess(C)
        logdetP = zero(T)
        @inbounds for c in 1:K
            logdetP += log(W[c, c])
        end
        logdetP *= 2
        # Σ = P⁻¹ via in-place triangular solves against the identity (no allocation).
        _set_identity!(Σ, K)
        ldiv!(C, Σ)
        return logdetP
    end

    # Fallback: ridge + Bunch–Kaufman (indefinite-robust), still no exceptions.
    @. W = beta * G
    @inbounds for c in 1:K
        W[c, c] += alpha[c] + lambda_reg
    end
    F = bunchkaufman!(Hermitian(W, :U), check = false)
    logdetP = first(logabsdet(F))
    _set_identity!(Σ, K)
    ldiv!(F, Σ)
    return logdetP
end

@inline function _set_identity!(Σ::AbstractMatrix{T}, K::Int) where {T <: Real}
    fill!(Σ, zero(T))
    @inbounds for c in 1:K
        Σ[c, c] = one(T)
    end
    return nothing
end

# ============================================================================
# Sparse-Bayes statistics: full recompute (init + after a large β change)
# Port of bayesSparsify.m `computeSparseBayesStatisticsOpt`.
# ============================================================================

@muladd function _recompute_statistics!(
        ws::ARDWorkspace, active_idx::AbstractVector{Int},
        alpha::AbstractVector{T}, mean::AbstractVector{T},
        beta::T, output_energy::T, lambda_reg::T
    ) where {T <: Real}
    K = length(active_idx)
    nf = length(ws.output_proj)
    aidx = view(active_idx, 1:K)
    cross = view(ws.cross, :, 1:K)

    logdetP = _spd_inverse!(ws.Cov, ws, K, alpha, beta, active_idx, lambda_reg)
    Cov = view(ws.Cov, 1:K, 1:K)

    # mean = β · Σ · output_proj[active]
    aproj = view(ws.output_proj, aidx)
    meanv = view(mean, 1:K)
    mul!(meanv, Cov, aproj)
    @. meanv *= beta

    # betaproj = β · cross
    bproj = view(ws.betaproj, :, 1:K)
    @. bproj = beta * cross

    # s_in_i = β − β² · (crossᵢ · Σ · crossᵢ)   via M = cross·Σ
    M = view(ws.M, :, 1:K)
    mul!(M, cross, Cov)
    @inbounds for i in 1:nf
        acc = zero(T)
        for c in 1:K
            acc += M[i, c] * cross[i, c]
        end
        ws.s_in[i] = beta - beta * beta * acc
    end

    # q_in = β · (output_proj − cross·mean)
    mul!(ws.proj_nf, cross, meanv)
    @. ws.q_in = beta * (ws.output_proj - ws.proj_nf)

    _refresh_out_factors!(ws, active_idx, alpha, K)

    # Residual energy and log marginal likelihood (matches the MATLAB reference)
    mul!(view(ws.Gmean, 1:K), view(ws.cross, aidx, 1:K), meanv)  # G·mean
    resid_energy = output_energy - 2 * dot(meanv, aproj) + dot(meanv, view(ws.Gmean, 1:K))
    data_like = (size(ws.Ψ, 1) * log(beta) - beta * resid_energy) / 2
    quad = zero(T)
    sum_log_alpha = zero(T)
    @inbounds for c in 1:K
        quad += meanv[c] * meanv[c] * alpha[c]
        sum_log_alpha += log(alpha[c])
    end
    log_ml = data_like - quad / 2 + sum_log_alpha / 2 - logdetP / 2
    return log_ml
end

"""
    _refresh_out_factors!(ws, active_idx, alpha, K)

Recompute `s_out`, `q_out`, `theta` (the per-candidate excluded-basis factors)
from the maintained `s_in`/`q_in` via the active-feature rescaling.
"""
@muladd function _refresh_out_factors!(
        ws::ARDWorkspace, active_idx::AbstractVector{Int},
        alpha::AbstractVector{T}, K::Int
    ) where {T <: Real}
    @. ws.s_out = ws.s_in
    @. ws.q_out = ws.q_in
    @inbounds for p in 1:K
        t = active_idx[p]
        denom = alpha[p] - ws.s_in[t]
        scale = alpha[p] / denom
        ws.s_out[t] = scale * ws.s_in[t]
        ws.q_out[t] = scale * ws.q_in[t]
    end
    @. ws.theta = ws.q_out^2 - ws.s_out
    return nothing
end

# ============================================================================
# In-place buffer compaction helpers (deletion of an active term)
# ============================================================================

@inline function _drop_col!(A::AbstractMatrix, j::Int, K::Int)
    @inbounds for c in j:(K - 1)
        @views A[:, c] .= A[:, c + 1]
    end
    return nothing
end

@inline function _drop_rowcol!(Cov::AbstractMatrix, j::Int, K::Int)
    @inbounds for c in j:(K - 1)            # shift columns left
        for r in 1:K
            Cov[r, c] = Cov[r, c + 1]
        end
    end
    @inbounds for c in 1:(K - 1)            # shift rows up
        for r in j:(K - 1)
            Cov[r, c] = Cov[r + 1, c]
        end
    end
    return nothing
end

# ============================================================================
# Main Fitting Function (vector target)
# ============================================================================

@muladd function fit!(
        model::FastARDRegressor{T},
        X::AbstractMatrix{T},
        y::AbstractVector{T}
    ) where {T <: Real}

    n_samples, n_features = size(X)

    # --- Preprocessing: optional standardization (API-level) -----------------
    if model.standardize
        model.X_mean = mean(X, dims = 1)
        model.y_mean = mean(y)
        Xp = X .- model.X_mean
        yp = y .- model.y_mean
        model.X_std = std(Xp, dims = 1)
        model.X_std = map(s -> s < eps(T) ? one(T) : s, model.X_std)
        Xp = Xp ./ model.X_std
    else
        model.X_mean = zeros(T, 1, n_features)
        model.X_std = ones(T, 1, n_features)
        model.y_mean = zero(T)
        Xp = Matrix{T}(X)
        yp = Vector{T}(y)
    end

    empty!(model.scores)
    model.converged = false

    # --- Internal column-L2 scaling (conditioning, like the references) ------
    scales = vec(sqrt.(sum(abs2, Xp, dims = 1)))
    @inbounds for j in eachindex(scales)
        scales[j] < eps(T) && (scales[j] = one(T))
    end
    Ψ = Xp ./ reshape(scales, 1, :)

    output_proj = Ψ' * yp
    output_energy = dot(yp, yp)
    var_y = var(yp)

    # Degenerate (near-constant) target: nothing to fit.
    if var_y < eps(T) || all(<(eps(T)), abs.(output_proj))
        model.alpha = fill(model.max_alpha, n_features)
        model.active = falses(n_features)
        model.coef = zeros(T, n_features)
        model.sigma = Matrix{T}(undef, 0, 0)
        model.beta = clamp(inv(max(var_y, eps(T))), model.min_beta, model.max_beta)
        return model
    end

    # --- Algorithm constants (Tipping & Faul / bayesSparsify.m) --------------
    zero_factor = T(1.0e-12)
    min_dlog_alpha = T(1.0e-3)
    min_dlog_beta = T(1.0e-6)
    alignment_max = one(T) - T(1.0e-3)
    initial_alpha_max = T(1.0e3)
    snr = T(0.1)
    max_beta = T(1.0e6) / var_y

    Kmax = min(n_features, n_samples)
    ws = ARDWorkspace(Ψ, output_proj, Kmax)

    # --- Initialization: single most-correlated term -------------------------
    std_y = max(T(1.0e-6), std(yp))
    beta = inv((std_y * snr)^2)

    t0 = argmax(abs.(output_proj))
    active_idx = Int[]
    sizehint!(active_idx, Kmax)
    push!(active_idx, t0)
    alpha = T[]
    sizehint!(alpha, Kmax)
    mean_vec = T[]
    sizehint!(mean_vec, Kmax)

    scaled_power = beta                              # unit-norm columns ⇒ β·‖ψ‖²=β
    scaled_proj = beta * output_proj[t0]
    a0 = scaled_power^2 / (scaled_proj^2 - scaled_power)
    a0 = a0 < zero(T) ? initial_alpha_max : a0
    push!(alpha, a0)
    push!(mean_vec, zero(T))

    mul!(view(ws.cross, :, 1), Ψ', view(Ψ, :, t0))
    @views ws.sparseΨ[:, 1] .= Ψ[:, t0]

    log_ml = _recompute_statistics!(
        ws, active_idx, alpha, mean_vec, beta, output_energy, model.lambda_reg
    )

    # Alignment bookkeeping (deferred near-duplicate candidates)
    aligned_out = Int[]
    aligned_in = Int[]

    # Action codes
    ACT_REEST = 0
    ACT_ADD = 1
    ACT_DELETE = -1
    ACT_TERM = 10
    ACT_NOISE = 11
    ACT_ALIGN = 12

    iters_run = 0
    for iter in 1:model.n_iter
        iters_run = iter
        K = length(active_idx)
        action = ACT_TERM
        sel_term = 0
        sel_pos = 0
        delta_lml = zero(T)

        # --- STEP 1: deletion priority --------------------------------------
        any_can_delete = false
        if K > 1
            best_del = zero(T)
            best_pos = 0
            @inbounds for p in 1:K
                t = active_idx[p]
                if ws.theta[t] <= zero_factor
                    any_can_delete = true
                    qo = ws.q_out[t]
                    so = ws.s_out[t]
                    a = alpha[p]
                    d = -(qo * qo / (so + a) - log(one(T) + so / a)) / 2
                    if d > best_del
                        best_del = d
                        best_pos = p
                    end
                end
            end
            if any_can_delete && best_del > zero(T)
                delta_lml = best_del
                sel_pos = best_pos
                sel_term = active_idx[best_pos]
                action = ACT_DELETE
            end
        end

        # --- STEP 2: add / re-estimate (only if no useful deletion) ----------
        if !any_can_delete
            best_dl = typemin(T)
            best_t = 0
            best_active = false
            best_p = 0
            # 2a. re-estimate active terms with positive relevance
            @inbounds for p in 1:K
                t = active_idx[p]
                if ws.theta[t] > zero_factor
                    cand_alpha = ws.s_out[t]^2 / ws.theta[t]
                    dinv = inv(cand_alpha) - inv(alpha[p])
                    si = ws.s_in[t]
                    qi = ws.q_in[t]
                    dl = (dinv * qi * qi / (dinv * si + one(T)) - log(one(T) + si * dinv)) / 2
                    if isfinite(dl) && dl > best_dl
                        best_dl = dl
                        best_t = t
                        best_active = true
                        best_p = p
                    end
                end
            end
            # 2b. add candidates (positive relevance, inactive, not deferred)
            if K < Kmax
                @inbounds for t in 1:n_features
                    (ws.theta[t] > zero_factor) || continue
                    (t in active_idx) && continue
                    (t in aligned_out) && continue
                    ratio = ws.q_in[t]^2 / ws.s_in[t]
                    dl = (ratio - one(T) - log(ratio)) / 2
                    if isfinite(dl) && dl > best_dl
                        best_dl = dl
                        best_t = t
                        best_active = false
                        best_p = 0
                    end
                end
            end

            if best_dl > zero(T) && best_t != 0
                delta_lml = best_dl
                sel_term = best_t
                if best_active
                    action = ACT_REEST
                    sel_pos = best_p
                else
                    action = ACT_ADD
                end
            end
        end

        sel_alpha = zero(T)
        if action == ACT_REEST || action == ACT_ADD
            sel_alpha = ws.s_out[sel_term]^2 / ws.theta[sel_term]
        end

        # --- STEP 2c: re-estimation convergence guard ------------------------
        if action == ACT_REEST
            if abs(log(sel_alpha) - log(alpha[sel_pos])) < min_dlog_alpha
                action = ACT_TERM
            end
        end

        # --- STEP 3: alignment (collinearity) test ---------------------------
        if action == ACT_ADD
            n_aligned = 0
            @inbounds for p in 1:K
                al = dot(view(Ψ, :, sel_term), view(ws.sparseΨ, :, p))
                if al > alignment_max
                    push!(aligned_out, sel_term)
                    push!(aligned_in, active_idx[p])
                    n_aligned += 1
                end
            end
            n_aligned > 0 && (action = ACT_ALIGN)
        elseif action == ACT_DELETE
            # un-defer candidates that were aligned to the term being deleted
            keep = aligned_in .!= sel_term
            if !all(keep)
                aligned_in = aligned_in[keep]
                aligned_out = aligned_out[keep]
            end
        end

        # --- STEP 4: rank-1 posterior update ---------------------------------
        if action == ACT_REEST
            p = sel_pos
            K1 = K
            Cov = view(ws.Cov, 1:K1, 1:K1)
            cc = view(ws.cov_col, 1:K1)
            cc .= view(Cov, :, p)
            old_alpha = alpha[p]
            alpha[p] = sel_alpha
            dinv = inv(sel_alpha - old_alpha)
            kappa = inv(Cov[p, p] + dinv)
            # Cov -= kappa · cc · ccᵀ
            @inbounds for cj in 1:K1, ri in 1:K1
                Cov[ri, cj] -= kappa * cc[ri] * cc[cj]
            end
            mc = view(ws.mean_change, 1:K1)
            mp = mean_vec[p]
            @. mc = -mp * kappa * cc
            @views mean_vec[1:K1] .+= mc
            # s_in += kappa · (betaproj·cc)² ; q_in -= betaproj·mc
            bproj = view(ws.betaproj, :, 1:K1)
            mul!(ws.proj_nf, bproj, cc)
            mul!(ws.proj_nf2, bproj, mc)
            @. ws.s_in += kappa * ws.proj_nf^2
            @. ws.q_in -= ws.proj_nf2
            _post_update!(ws, active_idx, alpha, beta, K1)
            log_ml += delta_lml

        elseif action == ACT_ADD
            K1 = K + 1
            t = sel_term
            mul!(view(ws.cross, :, K1), Ψ', view(Ψ, :, t))   # new projection column
            @views ws.sparseΨ[:, K1] .= Ψ[:, t]
            Cov = view(ws.Cov, 1:K, 1:K)
            cc = view(ws.cross_cov, 1:K)
            # cc = β · Cov · cross[t, 1:K]   (= ((βψ_t'Ψ_active)·Cov)ᵀ)
            mul!(cc, Cov, view(ws.cross, t, 1:K))
            @. cc *= beta
            new_var = inv(sel_alpha + ws.s_in[t])
            # top-left: Cov += new_var · cc · ccᵀ   (correct Schur sign)
            @inbounds for cj in 1:K, ri in 1:K
                ws.Cov[ri, cj] += new_var * cc[ri] * cc[cj]
            end
            @inbounds for r in 1:K
                ncc = -new_var * cc[r]
                ws.Cov[r, K1] = ncc
                ws.Cov[K1, r] = ncc
            end
            ws.Cov[K1, K1] = new_var
            new_mean = new_var * ws.q_in[t]
            @inbounds for r in 1:K
                mean_vec[r] += -new_mean * cc[r]
            end
            push!(mean_vec, new_mean)
            push!(alpha, sel_alpha)
            # proj_res = β·cross[:,K1] − (β·cross[:,1:K])·cc
            bproj = view(ws.betaproj, :, 1:K)
            mul!(ws.proj_nf2, bproj, cc)
            @inbounds for i in 1:n_features
                pr = beta * ws.cross[i, K1] - ws.proj_nf2[i]
                ws.s_in[i] -= new_var * pr * pr
                ws.q_in[i] -= new_mean * pr
            end
            push!(active_idx, t)
            _post_update!(ws, active_idx, alpha, beta, K1)
            log_ml += delta_lml

        elseif action == ACT_DELETE
            p = sel_pos
            Cov = view(ws.Cov, 1:K, 1:K)
            rv = Cov[p, p]
            rc = view(ws.cov_col, 1:K)
            rc .= view(Cov, :, p)
            bproj = view(ws.betaproj, :, 1:K)
            mul!(ws.proj_nf, bproj, rc)                  # projcol = betaproj·rc
            rem_mean = mean_vec[p]
            # Cov -= rc·rcᵀ / rv
            @inbounds for cj in 1:K, ri in 1:K
                Cov[ri, cj] -= rc[ri] * rc[cj] / rv
            end
            @inbounds for r in 1:K
                mean_vec[r] += -rem_mean * rc[r] / rv
            end
            @inbounds for i in 1:n_features
                pc = ws.proj_nf[i]
                ws.s_in[i] += pc * pc / rv
                ws.q_in[i] += pc * rem_mean / rv
            end
            # structural removal of active position p
            _drop_rowcol!(ws.Cov, p, K)
            _drop_col!(ws.cross, p, K)
            _drop_col!(ws.sparseΨ, p, K)
            deleteat!(active_idx, p)
            deleteat!(alpha, p)
            deleteat!(mean_vec, p)
            _post_update!(ws, active_idx, alpha, beta, K - 1)
            log_ml += delta_lml
        end

        # --- STEP 5: noise precision (β) update ------------------------------
        K = length(active_idx)
        old_beta = beta
        aidx = view(active_idx, 1:K)
        meanv = view(mean_vec, 1:K)
        mul!(view(ws.Gmean, 1:K), view(ws.cross, aidx, 1:K), meanv)
        resid_energy = output_energy - 2 * dot(meanv, view(ws.output_proj, aidx)) +
            dot(meanv, view(ws.Gmean, 1:K))
        sum_gamma = zero(T)
        @inbounds for p in 1:K
            sum_gamma += alpha[p] * ws.Cov[p, p]
        end
        sum_gamma = K - sum_gamma
        beta = (n_samples - sum_gamma) / max(resid_energy, T(1.0e-12))
        beta = min(beta, max_beta)
        if abs(log(beta) - log(old_beta)) > min_dlog_beta
            log_ml = _recompute_statistics!(
                ws, active_idx, alpha, mean_vec, beta, output_energy, model.lambda_reg
            )
            action == ACT_TERM && (action = ACT_NOISE)
        end

        model.compute_score && push!(model.scores, log_ml)

        if action == ACT_TERM
            model.converged = true
            model.verbose && println("Converged at iteration $iter")
            break
        end

        if model.verbose && (iter % 50 == 0)
            println("Iteration $iter: active = $(length(active_idx)), β = $beta")
        end
    end

    # --- Finalize: back-transform to the standardized (Xp) scale -------------
    K = length(active_idx)
    model.alpha = fill(model.max_alpha, n_features)
    model.active = falses(n_features)
    model.coef = zeros(T, n_features)
    @inbounds for p in 1:K
        t = active_idx[p]
        model.active[t] = true
        model.coef[t] = mean_vec[p] / scales[t]
        model.alpha[t] = alpha[p] / (scales[t] * scales[t])
    end
    model.beta = beta

    if K > 0
        sigma = Matrix{T}(undef, K, K)
        @inbounds for cj in 1:K, ri in 1:K
            sigma[ri, cj] = ws.Cov[ri, cj] / (scales[active_idx[ri]] * scales[active_idx[cj]])
        end
        # store sigma ordered to match findall(active) (ascending term index)
        perm = sortperm(active_idx)
        model.sigma = sigma[perm, perm]
    else
        model.sigma = Matrix{T}(undef, 0, 0)
    end

    model.verbose && println("Finished after $iters_run iterations; active = $K")
    return model
end

"""
    _post_update!(ws, active_idx, alpha, beta, K)

After a rank-1 structural/precision change, refresh the excluded-basis factors
and the cached `betaproj = β·cross`.
"""
@muladd function _post_update!(
        ws::ARDWorkspace, active_idx::AbstractVector{Int},
        alpha::AbstractVector{T}, beta::T, K::Int
    ) where {T <: Real}
    _refresh_out_factors!(ws, active_idx, alpha, K)
    @views @. ws.betaproj[:, 1:K] = beta * ws.cross[:, 1:K]
    return nothing
end

# ============================================================================
# Multi-output convenience: fit one independent model per column of Y
# ============================================================================

@muladd function fit!(
        model::FastARDRegressor{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    ) where {T <: Real}
    n_outputs = size(Y, 2)
    models = Vector{typeof(model)}(undef, n_outputs)
    for i in 1:n_outputs
        model.verbose && println("Fitting output $i/$n_outputs")
        models[i] = fit!(deepcopy(model), X, view(Y, :, i))
    end
    return models
end

# ============================================================================
# Prediction Functions
# ============================================================================

@muladd function predict(model::FastARDRegressor{T}, X::AbstractMatrix{T}) where {T <: Real}
    active_idx = findall(model.active)
    isempty(active_idx) && return fill(model.y_mean, size(X, 1))

    X_processed = if model.standardize
        (X .- model.X_mean) ./ model.X_std
    else
        X
    end
    X_active = view(X_processed, :, active_idx)
    coef_active = view(model.coef, active_idx)
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

    X_processed = if model.standardize
        (X .- model.X_mean) ./ model.X_std
    else
        X
    end
    X_active = view(X_processed, :, active_idx)
    coef_active = view(model.coef, active_idx)

    y_pred = X_active * coef_active .+ model.y_mean

    var_noise = inv(model.beta)
    # var_param = rowwise xᵀ Σ x   (batched: sum((Xa·Σ) .* Xa, dims=2))
    if !isempty(model.sigma)
        XΣ = X_active * model.sigma
        var_param = vec(sum(XΣ .* X_active, dims = 2))
    else
        var_param = zeros(T, n_test)
    end

    y_std = @. safe_sqrt(var_noise + var_param)
    return y_pred, y_std
end

function get_active_coefficients(model::FastARDRegressor)
    active_idx = findall(model.active)
    return active_idx, view(model.coef, active_idx)
end


function __init__()
    # NOTE: use_bigfloat_transcendentals() uses eval() which breaks Julia 1.12 incremental compilation.
    # Wrapped in try-catch to allow precompilation on Julia >= 1.12.
    try
        MultiFloats.use_bigfloat_transcendentals()
    catch e
        @warn "MultiFloats.use_bigfloat_transcendentals() failed (Julia 1.12+ compat): $e"
    end
    return nothing
end

end # module
