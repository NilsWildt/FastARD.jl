using FastARD
using Test
using Statistics
using LinearAlgebra
using Random
using MultiFloats

Test.@testset let
    # Test data with perfect multicollinearity (same as Python test)
    X = [0.1  -0.1  -0.2   0.02;
         0.3  -0.3  -0.6   0.06;
         0.4  -0.4  -0.8   0.08;
         0.5  -0.5  -1.0   0.1]
    y = [2.0, 6.0, 8.0, 10.0]
    
    # Test model creation
    model = FastARD.FastARDRegressor(verbose=false)    
    # Test fitting
    fitted_model = FastARD.fit!(model, X, y)
        Test.@test length(model.coef) == size(X, 2)
        Test.@test model.beta > 0  # Noise precision should be positive
    
    # # Test prediction
    y_pred = FastARD.predict(model, X)
        Test.@test length(y_pred) == length(y)
        Test.@test all(isfinite.(y_pred))
    
    # # Test prediction accuracy (with tolerance for regularization)
        Test.@test isapprox(y_pred, y, atol=0.5)
end

Test.@testset let
    
    
    # Test data
    X = [0.1  -0.1  -0.2   0.02;
         0.3  -0.3  -0.6   0.06;
         0.4  -0.4  -0.8   0.08;
         0.5  -0.5  -1.0   0.1]
    y = [2.0, 6.0, 8.0, 10.0]
    
    model = FastARD.FastARDRegressor(verbose=false)
    FastARD.fit!(model, X, y)
    
    # Test predict_with_uncertainty
    y_pred, y_std = FastARD.predict_with_uncertainty(model, X)
    
    # Test shapes match
    Test.@test length(y_pred) == length(y_std)
    Test.@test length(y_pred) == size(X, 1)
    
    # Test all standard deviations are non-negative
    Test.@test all(y_std .>= 0)
    
    # Test that all predictions are finite
    Test.@test all(isfinite.(y_pred))
    Test.@test all(isfinite.(y_std))
    
    # Test that point predictions are close to true values
    Test.@test isapprox(y_pred, y, atol=0.5)
    
    # Test that predict() and predict_with_uncertainty() return same point estimates
    y_pred_simple = FastARD.predict(model, X)
    Test.@test isapprox(y_pred, y_pred_simple, rtol=1e-10)
end


Test.@testset let
    # Sparse test case
    X = randn(50, 20)
    true_coef = zeros(20)
    true_coef[[1, 5, 10]] .= [2.0, -1.5, 3.0]  # Only 3 active features
    y = X * true_coef + 0.1 * randn(50)
    
    model = FastARD.FastARDRegressor(verbose=false, n_iter=100)
    FastARD.fit!(model, X, y)
    
    # Test active coefficients
    active_idx, active_coef = FastARD.get_active_coefficients(model)
    
    Test.@test isa(active_idx, Vector{Int})
    Test.@test length(active_idx) == length(active_coef)
    Test.@test length(active_idx) <= size(X, 2)  # Can't have more active than features
    Test.@test all(active_idx .>= 1)  # Valid indices
    Test.@test all(active_idx .<= size(X, 2))  # Valid indices
    
    # Test that active coefficients are non-zero (within tolerance)
    Test.@test all(abs.(active_coef) .> 1e-10)
end

Test.@testset let
    
    # Well-conditioned problem
    n, p = 30, 10
    X = randn(n, p)
    true_coef = randn(p)
    y = X * true_coef + 0.01 * randn(n)  # Low noise
    
    model = FastARD.FastARDRegressor(verbose=false, n_iter=200, tol=1e-8)
    FastARD.fit!(model, X, y)
    
    # Test basic properties
    Test.@test model.beta > 0
    Test.@test all(model.alpha .> 0)
    Test.@test sum(model.active) > 0  # Should find some active features
    Test.@test sum(model.active) <= p  # Can't exceed number of features
    
    # Test prediction quality
    y_pred = FastARD.predict(model, X)
    rmse = sqrt(mean((y_pred .- y).^2))
    Test.@test rmse < 1.0  # Should achieve reasonable fit
end

Test.@testset let
    
    # Test with single feature
    X_single = reshape([1.0, 2.0, 3.0, 4.0], 4, 1)
    y_single = [2.0, 4.0, 6.0, 8.0]
    
    model = FastARD.FastARDRegressor(verbose=false)
    FastARD.fit!(model, X_single, y_single)
    
    Test.@test length(model.coef) == 1
    Test.@test sum(model.active) >= 1  # Should activate the single feature
    
    y_pred = FastARD.predict(model, X_single)
    Test.@test length(y_pred) == length(y_single)
    
    # Test with constant features (should handle gracefully)
    X_const = ones(5, 3)  # All constant features
    y_const = [1.0, 1.0, 1.0, 1.0, 1.0]
    
    model_const = FastARD.FastARDRegressor(verbose=false)
    # Should not crash
    FastARD.fit!(model_const, X_const, y_const)
    y_pred_const = FastARD.predict(model_const, X_const)
    Test.@test length(y_pred_const) == length(y_const)
end


Test.@testset let
    # Test constructor parameters
    model1 = FastARD.FastARDRegressor(n_iter=50)
    Test.@test model1.n_iter == 50
    
    model2 = FastARD.FastARDRegressor(tol=1e-10)
    Test.@test model2.tol == 1e-10
    
    model3 = FastARD.FastARDRegressor(verbose=true)
    Test.@test model3.verbose == true
    
    model4 = FastARD.FastARDRegressor(compute_score=true)
    Test.@test model4.compute_score == true
    
    # Test with custom regularization
    model5 = FastARD.FastARDRegressor(lambda_reg=1e-5, max_alpha=1e10)
    Test.@test model5.lambda_reg == 1e-5
    Test.@test model5.max_alpha == 1e10
end

Test.@testset let
    # Test with ill-conditioned problem
    n = 20
    X = randn(n, n)
    X = X + 1e-10 * randn(n, n)  # Add tiny perturbation
    X[:, end] = X[:, 1] + 1e-12 * randn(n)  # Nearly collinear column
    
    y = randn(n)
    
    model = FastARD.FastARDRegressor(verbose=false, lambda_reg=1e-6)
    
    # Should not crash despite ill-conditioning
    Test.@test_nowarn FastARD.fit!(model, X, y)
    
    y_pred = FastARD.predict(model, X)
    Test.@test all(isfinite.(y_pred))
    Test.@test length(y_pred) == n
    
    # Test uncertainty estimation doesn't crash
    y_pred_unc, y_std_unc = FastARD.predict_with_uncertainty(model, X)
    Test.@test all(isfinite.(y_pred_unc))
    Test.@test all(isfinite.(y_std_unc))
    Test.@test all(y_std_unc .>= 0)
end

Test.@testset let
    X = randn(25, 8)
    y = randn(25)
    
    model = FastARD.FastARDRegressor(verbose=false, compute_score=true)
    FastARD.fit!(model, X, y)
    
    Test.@test length(model.scores) > 0  # Should have computed scores
    Test.@test all(isfinite.(model.scores))  # All scores should be finite
    
    # Scores should generally increase (better likelihood) or stabilize
    if length(model.scores) > 1
        # Allow for some numerical noise in score progression
        Test.@test model.scores[end] >= model.scores[1] - 1.0
    end
end

# ===========================================================================
# Rank-1 correctness: the maintained posterior mean must equal an independent
# ground-truth posterior solve on the (internally column-scaled) design for the
# selected active set. This catches sign/algebra errors in the rank-1 updates
# regardless of which active set is chosen.
# ===========================================================================
"""
    bruteforce_mean_diff(model, X, y) -> Float64

Recompute μ = β (β ΨₐᵀΨₐ + diag(α_Ψ))⁻¹ Ψₐᵀ yp from scratch on the internally
column-scaled design Ψ, and compare to the model's coefficients (mapped back to
the Ψ scale). Returns the max absolute difference.
"""
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

Test.@testset "rank-1 posterior self-consistency" begin
    Random.seed!(2024)
    # multicollinear hardcoded case
    X1 = [0.1 -0.1 -0.2 0.02; 0.3 -0.3 -0.6 0.06; 0.4 -0.4 -0.8 0.08; 0.5 -0.5 -1.0 0.1]
    y1 = [2.0, 6.0, 8.0, 10.0]
    m1 = FastARD.FastARDRegressor(verbose = false); FastARD.fit!(m1, X1, y1)
    Test.@test bruteforce_mean_diff(m1, X1, y1) < 1e-6

    # random sparse + dense problems
    for (n, p) in ((50, 20), (40, 12), (80, 30))
        X = randn(n, p)
        tc = zeros(p); tc[rand(1:p, 3)] .= randn(3) .* 2
        y = X * tc + 0.05 * randn(n)
        m = FastARD.FastARDRegressor(verbose = false); FastARD.fit!(m, X, y)
        Test.@test bruteforce_mean_diff(m, X, y) < 1e-5
    end

    # no-standardize path
    X = randn(45, 10); y = X * randn(10) + 0.05 * randn(45)
    m = FastARD.FastARDRegressor(verbose = false, standardize = false)
    FastARD.fit!(m, X, y)
    Test.@test bruteforce_mean_diff(m, X, y) < 1e-5
end

Test.@testset "sparse feature recovery" begin
    Random.seed!(11)
    X = randn(120, 25)
    tc = zeros(25); tc[[3, 11, 20]] .= [2.5, -1.8, 3.3]
    y = X * tc + 0.05 * randn(120)
    m = FastARD.FastARDRegressor(verbose = false); FastARD.fit!(m, X, y)
    active = findall(m.active)
    # all three true features recovered
    Test.@test all(in(active), (3, 11, 20))
    # reasonably sparse (does not select everything)
    Test.@test length(active) <= 12
    Test.@test m.converged
end

Test.@testset "type stability (@inferred)" begin
    Random.seed!(5)
    X = randn(50, 15); y = X * vcat(randn(4), zeros(11)) + 0.05 * randn(50)
    m = FastARD.FastARDRegressor(verbose = false)
    Test.@test_nowarn @inferred FastARD.fit!(m, X, y)
    Test.@test_nowarn @inferred FastARD.predict(m, X)
    Test.@test_nowarn @inferred FastARD.predict_with_uncertainty(m, X)
end

Test.@testset "bounded allocations" begin
    Random.seed!(9)
    X = randn(60, 25); y = X * vcat(randn(4), zeros(21)) + 0.05 * randn(60)
    FastARD.fit!(FastARD.FastARDRegressor(verbose = false), X, y)  # warm
    bytes = @allocated FastARD.fit!(FastARD.FastARDRegressor(verbose = false), X, y)
    # Workspace is preallocated; a full fit must stay far below the
    # per-iteration-reallocation behaviour of the reference implementations.
    Test.@test bytes < 5_000_000
end

Test.@testset "alignment test defers collinear duplicate" begin
    Random.seed!(3)
    X = randn(40, 6)
    X[:, 4] = X[:, 2]                      # exact duplicate (positive collinearity)
    y = X[:, 2] * 3.0 + 0.01 * randn(40)
    m = FastARD.FastARDRegressor(verbose = false, standardize = false)
    FastARD.fit!(m, X, y)
    active = findall(m.active)
    # the two identical columns must not both be selected
    Test.@test !(2 in active && 4 in active)
end

Test.@testset "MultiFloat high precision" begin
    Random.seed!(13)
    T = MultiFloat{Float64, 2}
    X = randn(40, 10); y = X * vcat(randn(3), zeros(7)) + 0.02 * randn(40)
    m = FastARD.FastARDRegressor(T; verbose = false)
    FastARD.fit!(m, T.(X), T.(y))
    Test.@test sum(m.active) > 0
    yp = FastARD.predict(m, T.(X))
    Test.@test all(isfinite, yp)
    Test.@test eltype(yp) == T
end

Test.@testset "multi-output returns a vector of models" begin
    Random.seed!(17)
    X = randn(50, 8)
    Y = X * randn(8, 3) + 0.05 * randn(50, 3)
    m = FastARD.FastARDRegressor(verbose = false)
    models = FastARD.fit!(m, X, Y)
    Test.@test models isa Vector
    Test.@test length(models) == 3
    Test.@test all(mi -> mi isa FastARD.FastARDRegressor, models)
end