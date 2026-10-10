# Benchmarks run by AirspeedVelocity.jl (see .github/workflows/Benchmark.yml).
# `fit!` mutates its model, so every sample builds a fresh one in `setup` and
# runs exactly once (`evals = 1`).
using BenchmarkTools
using FastARD
using Random

const SUITE = BenchmarkGroup()

function sparse_problem(n, p, k; seed = 20261010)
    rng = Random.Xoshiro(seed)
    X = randn(rng, n, p)
    coef = zeros(p)
    coef[randperm(rng, p)[1:k]] .= 2 .* randn(rng, k)
    y = X * coef .+ 0.05 .* randn(rng, n)
    return X, y
end

SUITE["fit"] = BenchmarkGroup()
for (n, p, k) in ((60, 25, 4), (400, 200, 10))
    X, y = sparse_problem(n, p, k)
    SUITE["fit"]["$(n)x$(p)"] = @benchmarkable fit!(model, $X, $y) setup = (
        model = FastARDRegressor(verbose = false, n_iter = 300)
    ) evals = 1
end

let (X, y) = sparse_problem(400, 200, 10)
    model = FastARDRegressor(verbose = false, n_iter = 300)
    fit!(model, X, y)
    SUITE["predict"] = @benchmarkable predict($model, $X)
end
