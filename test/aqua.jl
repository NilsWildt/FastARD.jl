# Aqua code-quality checks: undefined exports, stale/missing deps, compat bounds,
# method ambiguities, piracy, project formatting.
#
# FastARD intentionally extends `Base.precision(::Type{MultiFloat})` (src/FastARD.jl)
# to report the implicit precision across MultiFloat limbs. Aqua flags this as
# type piracy because `MultiFloat` is owned by MultiFloats; `treat_as_own`
# declares the extension as deliberate so the rest of the piracy check still runs.

@testitem "Aqua quality" begin
    using Aqua
    using MultiFloats: MultiFloat
    Aqua.test_all(
        FastARD;
        piracies = (treat_as_own = [MultiFloat],),
    )
end
