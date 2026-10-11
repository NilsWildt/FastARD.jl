# Core behavioural tests for FastARD, migrated from the previous flat
# `runtests.jl` into discrete @testitem blocks. Each block runs in its own
# module under TestItemRunner and declares its own imports.

@testitem "fit/predict on multicollinear data" begin
    using FastARD, Test, Statistics, LinearAlgebra

    X = [
        0.1 -0.1 -0.2 0.02;
        0.3 -0.3 -0.6 0.06;
        0.4 -0.4 -0.8 0.08;
        0.5 -0.5 -1.0 0.1
    ]
    y = [2.0, 6.0, 8.0, 10.0]

    model = FastARD.FastARDRegressor(verbose = false)
    fitted_model = FastARD.fit!(model, X, y)
    @test length(model.coef) == size(X, 2)
    @test model.beta > 0  # noise precision should be positive

    y_pred = FastARD.predict(model, X)
    @test length(y_pred) == length(y)
    @test all(isfinite.(y_pred))
    @test isapprox(y_pred, y, atol = 0.5)
end

@testitem "predict_with_uncertainty" begin
    using FastARD, Test

    X = [
        0.1 -0.1 -0.2 0.02;
        0.3 -0.3 -0.6 0.06;
        0.4 -0.4 -0.8 0.08;
        0.5 -0.5 -1.0 0.1
    ]
    y = [2.0, 6.0, 8.0, 10.0]

    model = FastARD.FastARDRegressor(verbose = false)
    FastARD.fit!(model, X, y)

    y_pred, y_std = FastARD.predict_with_uncertainty(model, X)
    @test length(y_pred) == length(y_std)
    @test length(y_pred) == size(X, 1)
    @test all(y_std .>= 0)
    @test all(isfinite.(y_pred))
    @test all(isfinite.(y_std))
    @test isapprox(y_pred, y, atol = 0.5)

    y_pred_simple = FastARD.predict(model, X)
    @test isapprox(y_pred, y_pred_simple, rtol = 1.0e-10)
end

@testitem "active coefficient recovery on sparse data" begin
    using FastARD, Test, Random

    Random.seed!(0)
    X = randn(50, 20)
    true_coef = zeros(20)
    true_coef[[1, 5, 10]] .= [2.0, -1.5, 3.0]
    y = X * true_coef + 0.1 * randn(50)

    model = FastARD.FastARDRegressor(verbose = false, n_iter = 100)
    FastARD.fit!(model, X, y)

    active_idx, active_coef = FastARD.get_active_coefficients(model)
    @test isa(active_idx, Vector{Int})
    @test length(active_idx) == length(active_coef)
    @test length(active_idx) <= size(X, 2)
    @test all(active_idx .>= 1)
    @test all(active_idx .<= size(X, 2))
    @test all(abs.(active_coef) .> 1.0e-10)
end

@testitem "well-conditioned recovery quality" begin
    using FastARD, Test, Statistics, Random

    Random.seed!(1)
    n, p = 30, 10
    X = randn(n, p)
    true_coef = randn(p)
    y = X * true_coef + 0.01 * randn(n)

    model = FastARD.FastARDRegressor(verbose = false, n_iter = 200, tol = 1.0e-8)
    FastARD.fit!(model, X, y)

    @test model.beta > 0
    @test all(model.alpha .> 0)
    @test sum(model.active) > 0
    @test sum(model.active) <= p

    y_pred = FastARD.predict(model, X)
    rmse = sqrt(mean((y_pred .- y) .^ 2))
    @test rmse < 1.0
end

@testitem "single and constant features" begin
    using FastARD, Test

    X_single = reshape([1.0, 2.0, 3.0, 4.0], 4, 1)
    y_single = [2.0, 4.0, 6.0, 8.0]

    model = FastARD.FastARDRegressor(verbose = false)
    FastARD.fit!(model, X_single, y_single)
    @test length(model.coef) == 1
    @test sum(model.active) >= 1

    y_pred = FastARD.predict(model, X_single)
    @test length(y_pred) == length(y_single)

    X_const = ones(5, 3)
    y_const = [1.0, 1.0, 1.0, 1.0, 1.0]
    model_const = FastARD.FastARDRegressor(verbose = false)
    FastARD.fit!(model_const, X_const, y_const)  # must not crash
    y_pred_const = FastARD.predict(model_const, X_const)
    @test length(y_pred_const) == length(y_const)
end

@testitem "constructor parameters" begin
    using FastARD, Test

    @test FastARD.FastARDRegressor(n_iter = 50).n_iter == 50
    @test FastARD.FastARDRegressor(tol = 1.0e-10).tol == 1.0e-10
    @test FastARD.FastARDRegressor(verbose = true).verbose == true
    @test FastARD.FastARDRegressor(compute_score = true).compute_score == true

    model5 = FastARD.FastARDRegressor(lambda_reg = 1.0e-5, max_alpha = 1.0e10)
    @test model5.lambda_reg == 1.0e-5
    @test model5.max_alpha == 1.0e10
end

@testitem "ill-conditioned problem is stable" begin
    using FastARD, Test, Random

    Random.seed!(2)
    n = 20
    X = randn(n, n)
    X = X + 1.0e-10 * randn(n, n)
    X[:, end] = X[:, 1] + 1.0e-12 * randn(n)
    y = randn(n)

    model = FastARD.FastARDRegressor(verbose = false, lambda_reg = 1.0e-6)
    @test_nowarn FastARD.fit!(model, X, y)

    y_pred = FastARD.predict(model, X)
    @test all(isfinite.(y_pred))
    @test length(y_pred) == n

    y_pred_unc, y_std_unc = FastARD.predict_with_uncertainty(model, X)
    @test all(isfinite.(y_pred_unc))
    @test all(isfinite.(y_std_unc))
    @test all(y_std_unc .>= 0)
end

@testitem "score progression" begin
    using FastARD, Test, Random

    Random.seed!(4)
    X = randn(25, 8)
    y = randn(25)

    model = FastARD.FastARDRegressor(verbose = false, compute_score = true)
    FastARD.fit!(model, X, y)

    @test length(model.scores) > 0
    @test all(isfinite.(model.scores))
    if length(model.scores) > 1
        @test model.scores[end] >= model.scores[1] - 1.0
    end
end

@testitem "rank-1 posterior self-consistency" begin
    using FastARD, Test, LinearAlgebra, Random

    # Recompute μ = β (β ΨₐᵀΨₐ + diag(α_Ψ))⁻¹ Ψₐᵀ yp from scratch on the
    # internally column-scaled design Ψ and compare to the model coefficients.
    function bruteforce_mean_diff(model, X, y)
        Xp = model.standardize ? (X .- model.X_mean) ./ model.X_std : X
        scales = vec(sqrt.(sum(abs2, Xp, dims = 1)))
        scales[scales .< eps()] .= 1.0
        Ψ = Xp ./ scales'
        yp = model.standardize ? y .- model.y_mean : y
        ai = findall(model.active)
        isempty(ai) && return 0.0
        Ψa = Ψ[:, ai]
        αΨ = [model.alpha[t] * scales[t]^2 for t in ai]
        P = model.beta * (Ψa'Ψa) + Diagonal(αΨ)
        μΨ = model.beta * (P \ (Ψa'yp))
        coefΨ = [model.coef[t] * scales[t] for t in ai]
        return maximum(abs.(μΨ .- coefΨ))
    end

    Random.seed!(2024)
    X1 = [0.1 -0.1 -0.2 0.02; 0.3 -0.3 -0.6 0.06; 0.4 -0.4 -0.8 0.08; 0.5 -0.5 -1.0 0.1]
    y1 = [2.0, 6.0, 8.0, 10.0]
    m1 = FastARD.FastARDRegressor(verbose = false)
    FastARD.fit!(m1, X1, y1)
    @test bruteforce_mean_diff(m1, X1, y1) < 1.0e-6

    for (n, p) in ((50, 20), (40, 12), (80, 30))
        X = randn(n, p)
        tc = zeros(p); tc[rand(1:p, 3)] .= randn(3) .* 2
        y = X * tc + 0.05 * randn(n)
        m = FastARD.FastARDRegressor(verbose = false)
        FastARD.fit!(m, X, y)
        @test bruteforce_mean_diff(m, X, y) < 1.0e-5
    end

    X = randn(45, 10); y = X * randn(10) + 0.05 * randn(45)
    m = FastARD.FastARDRegressor(verbose = false, standardize = false)
    FastARD.fit!(m, X, y)
    @test bruteforce_mean_diff(m, X, y) < 1.0e-5
end

@testitem "sparse feature recovery" begin
    using FastARD, Test, Random

    Random.seed!(11)
    X = randn(120, 25)
    tc = zeros(25); tc[[3, 11, 20]] .= [2.5, -1.8, 3.3]
    y = X * tc + 0.05 * randn(120)
    m = FastARD.FastARDRegressor(verbose = false)
    FastARD.fit!(m, X, y)
    active = findall(m.active)
    @test all(in(active), (3, 11, 20))
    @test length(active) <= 12
    @test m.converged
end

@testitem "type stability (@inferred)" begin
    using FastARD, Test, Random

    Random.seed!(5)
    X = randn(50, 15); y = X * vcat(randn(4), zeros(11)) + 0.05 * randn(50)
    m = FastARD.FastARDRegressor(verbose = false)
    @test_nowarn @inferred FastARD.fit!(m, X, y)
    @test_nowarn @inferred FastARD.predict(m, X)
    @test_nowarn @inferred FastARD.predict_with_uncertainty(m, X)
end

@testitem "bounded allocations" begin
    using FastARD, Test, Random

    Random.seed!(9)
    X = randn(60, 25); y = X * vcat(randn(4), zeros(21)) + 0.05 * randn(60)
    FastARD.fit!(FastARD.FastARDRegressor(verbose = false), X, y)  # warm
    bytes = @allocated FastARD.fit!(FastARD.FastARDRegressor(verbose = false), X, y)
    @test bytes < 5_000_000
end

@testitem "alignment test defers collinear duplicate" begin
    using FastARD, Test, Random

    Random.seed!(3)
    X = randn(40, 6)
    X[:, 4] = X[:, 2]                      # exact duplicate (positive collinearity)
    y = X[:, 2] * 3.0 + 0.01 * randn(40)
    m = FastARD.FastARDRegressor(verbose = false, standardize = false)
    FastARD.fit!(m, X, y)
    active = findall(m.active)
    @test !(2 in active && 4 in active)
end

@testitem "MultiFloat high precision" begin
    using FastARD, Test, Random, MultiFloats

    Random.seed!(13)
    T = MultiFloat{Float64, 2}
    X = randn(40, 10); y = X * vcat(randn(3), zeros(7)) + 0.02 * randn(40)
    m = FastARD.FastARDRegressor(T; verbose = false)
    FastARD.fit!(m, T.(X), T.(y))
    @test sum(m.active) > 0
    yp = FastARD.predict(m, T.(X))
    @test all(isfinite, yp)
    @test eltype(yp) == T
end

@testitem "multi-output returns a vector of models" begin
    using FastARD, Test, Random

    Random.seed!(17)
    X = randn(50, 8)
    Y = X * randn(8, 3) + 0.05 * randn(50, 3)
    m = FastARD.FastARDRegressor(verbose = false)
    models = FastARD.fit!(m, X, Y)
    @test models isa Vector
    @test length(models) == 3
    @test all(mi -> mi isa FastARD.FastARDRegressor, models)
end

@testitem "beta_recompute_tol: default exact, throttled stays accurate" begin
    using FastARD, Test, Random, Statistics, LinearAlgebra

    Random.seed!(21)
    n, p = 300, 60
    X = randn(n, p)
    coef = vcat(randn(6) .* 3, zeros(p - 6))
    y = X * coef + 0.3 * std(X * coef) * randn(n)

    # default tol must be exactly the historical behavior (same trajectory)
    m_ref = FastARD.FastARDRegressor(compute_score = true)
    FastARD.fit!(m_ref, X, y)
    m_def = FastARD.FastARDRegressor(compute_score = true, beta_recompute_tol = 1.0e-6)
    FastARD.fit!(m_def, X, y)
    @test length(m_def.scores) == length(m_ref.scores)
    @test findall(m_def.active) == findall(m_ref.active)
    @test m_def.coef ≈ m_ref.coef

    # throttled: must run, converge, and predict close to the reference
    m_throttled = FastARD.FastARDRegressor(beta_recompute_tol = 1.0e-3)
    FastARD.fit!(m_throttled, X, y)
    @test sum(m_throttled.active) > 0
    y_ref = FastARD.predict(m_ref, X)
    y_throttled = FastARD.predict(m_throttled, X)
    @test maximum(abs.(y_ref .- y_throttled)) / maximum(abs.(y_ref)) < 0.05
end
