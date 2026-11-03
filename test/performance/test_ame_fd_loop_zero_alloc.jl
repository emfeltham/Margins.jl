# test_ame_fd_loop_zero_alloc.jl
#
# Per-row allocation of the population AME hot loop with the finite-difference backend:
# `_compute_continuous_effects!` (→ `accumulate_continuous!` with an `FDEvaluator`), on the
# engine's own buffers and workspace, for both scales, unweighted and weighted, for
# `:effect` and an elasticity measure. Allocation must not grow with the number of rows.
#
# (Rewritten in 3.0.0: the previous version called a removed
# `marginal_effects_eta!(g, de, β, row; backend=:fd)` signature and a removed 5-argument
# `get_or_build_engine`, and was not run by runtests.jl.)

using Test
using DataFrames, Tables, CategoricalArrays, Random
using GLM, StatsModels
using FormulaCompiler, Margins

function _fd_ame_loop_allocs(engine, vars, data_nt, rows, scale, measure, weights)
    total_weight = isnothing(weights) ? Float64(length(rows)) : Float64(sum(weights[rows]))
    Margins._compute_continuous_effects!(engine, vars, data_nt, rows, scale, measure, weights, total_weight)  # warm-up
    return @allocated Margins._compute_continuous_effects!(engine, vars, data_nt, rows, scale, measure, weights, total_weight)
end

@testset "AME FD hot loop: allocation independent of rows" begin
    Random.seed!(42)
    n = 20_000
    df = DataFrame(x = randn(n), z = abs.(randn(n)) .+ 0.1,
                   group = categorical(rand(["A", "B", "C"], n)))
    df.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.3 .* df.x .- 0.2 .* df.z .+ 0.1 .* (df.group .== "B"))))
    model = glm(@formula(y ~ x * group + log(z)), df, Binomial(), LogitLink())
    data_nt = Margins._prepare_margins_data(Tables.columntable(df), model)
    vars = [:x, :z]
    engine = Margins.build_engine(Margins.PopulationUsage, Margins.HasDerivatives, model, data_nt,
                                  vars, GLM.vcov, :fd)
    @test engine.de isa FormulaCompiler.FDEvaluator
    weights = rand(n) .+ 0.1

    for scale in (:link, :response), w in (nothing, weights), measure in (:effect, :elasticity)
        small = _fd_ame_loop_allocs(engine, vars, data_nt, 1:1_000, scale, measure, w)
        large = _fd_ame_loop_allocs(engine, vars, data_nt, 1:n, scale, measure, w)
        @test large == small            # O(1) in rows
        @test large <= (measure === :effect ? 128 : 512)  # effectively zero: one dynamic dispatch on the
                                                        # evaluator type (+ the column tuple of a measure)
    end
end
