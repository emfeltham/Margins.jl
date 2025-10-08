# test_ame_fd_loop_zero_alloc.jl
# Verifies near-zero allocation hot loop for AME (effects) with FD backend

using Test
using BenchmarkTools
using DataFrames, Tables, CategoricalArrays
using GLM, StatsModels
using FormulaCompiler, Margins
using Margins: marginal_effects_eta!

function _ame_fd_loop_typed!(de::DerivativeEvaluator,
                             gview::AbstractVector{Float64}, β::Vector{Float64},
                             n_calls::Int, n_rows::Int)
    @inbounds for i in 1:n_calls
        row = ((i - 1) % n_rows) + 1
        marginal_effects_eta!(gview, de, β, row; backend=:fd)
    end
    return nothing
end

@testset "AME FD Hot Loop Zero Allocation" begin
    # Small, representative dataset
    n = 5000
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        z = abs.(randn(n)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], n))
    )
    data_nt = Tables.columntable(df)

    # Fit model and build engine with derivatives (PopulationUsage)
    model = lm(@formula(y ~ x + z + group + x & group), df)
    engine = Margins.get_or_build_engine(Margins.PopulationUsage, model, data_nt, [:x, :z], GLM.vcov)

    # Warmup: compute FD marginal effects a few times
    gview = @view engine.g_buf[1:length(engine.de.vars)]
    for row in 1:10
        marginal_effects_eta!(gview, engine.de, engine.β, row; backend=:fd)
    end

    # Typed loop helper to force specialization on de    

    n_rows = length(first(data_nt))
    b = @benchmark _ame_fd_loop_typed!($engine.de, $gview, $engine.β, 50_000, $n_rows) samples=5 evals=1
    @test minimum(b.memory) <= 64
end
