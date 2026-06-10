# JET static analysis: assert the public numerical API is free of optimization
# failures (dynamic dispatch / runtime-dispatched calls) on concrete Float64
# inputs. The targets below are verified clean (0 reports); if a future change
# introduces a dynamic dispatch, this testitem fails loudly.

@testitem "JET optimization clean on public API" begin
    using JET, FastARD, Random

    Random.seed!(0)
    X = randn(40, 8)
    y = X * vcat(randn(3), zeros(5)) + 0.05 * randn(40)
    model = FastARD.FastARDRegressor(verbose = false)
    FastARD.fit!(model, X, y)

    targets = [
        (FastARD.safe_sqrt, (Float64,)),
        (FastARD.predict, (typeof(model), Matrix{Float64})),
        (FastARD.predict_with_uncertainty, (typeof(model), Matrix{Float64})),
    ]
    for (f, types) in targets
        report = JET.report_opt(f, types)
        @test isempty(JET.get_reports(report))
    end
end
