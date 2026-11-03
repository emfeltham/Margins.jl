# test_ame_fd_loop_zero_alloc.jl
#
# Allocation of the population AME hot loop: `_compute_continuous_effects!`
# (→ `accumulate_continuous!`), on the engine's own buffers and workspace, with the
# finite-difference and automatic-differentiation backends, for both scales, unweighted and
# weighted, for `:effect` and every measure. A call allocates nothing at all: the engine's
# type records its exact evaluator types (no dynamic dispatch through `engine.de`), and the
# measures' data columns go into the engine's `xcols_buf`.
#
# (Rewritten in 3.0.0: the previous version called a removed
# `marginal_effects_eta!(g, de, β, row; backend=:fd)` signature and a removed 5-argument
# `get_or_build_engine`, and was not run by runtests.jl.)

using Test
using DataFrames, Tables, CategoricalArrays, Random
using GLM, StatsModels
using FormulaCompiler, Margins

function _ame_loop_allocs(engine, vars, data_nt, rows, scale, measure, weights)
    total_weight = isnothing(weights) ? Float64(length(rows)) : Float64(sum(weights[rows]))
    Margins._compute_continuous_effects!(engine, vars, data_nt, rows, scale, measure, weights, total_weight)  # warm-up
    return @allocated Margins._compute_continuous_effects!(engine, vars, data_nt, rows, scale, measure, weights, total_weight)
end

@testset "AME hot loop: zero allocation (FD and AD, every measure)" begin
    Random.seed!(42)
    n = 20_000
    df = DataFrame(x = randn(n), z = abs.(randn(n)) .+ 0.1,
                   group = categorical(rand(["A", "B", "C"], n)))
    df.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.3 .* df.x .- 0.2 .* df.z .+ 0.1 .* (df.group .== "B"))))
    model = glm(@formula(y ~ x * group + log(z)), df, Binomial(), LogitLink())
    data_nt = Margins._prepare_margins_data(Tables.columntable(df), model)
    vars = [:x, :z]
    weights = rand(n) .+ 0.1
    measures = (:effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx)

    for (backend, E) in ((:fd, FormulaCompiler.FDEvaluator), (:ad, FormulaCompiler.ADEvaluator))
        engine = Margins.build_engine(Margins.PopulationUsage, model, data_nt, vars, GLM.vcov, backend)
        @test engine.de isa E
        @test isconcretetype(fieldtype(typeof(engine), :de))
        for scale in (:link, :response), w in (nothing, weights), measure in measures
            small = _ame_loop_allocs(engine, vars, data_nt, 1:1_000, scale, measure, w)
            large = _ame_loop_allocs(engine, vars, data_nt, 1:n, scale, measure, w)
            @test small == 0
            @test large == 0
        end
    end
end

@testset "Engine type: concrete evaluators, none where unused" begin
    Random.seed!(7)
    n = 500
    df = DataFrame(x = randn(n), group = categorical(rand(["A", "B", "C"], n)))
    df.y = rand(n) .< 1 ./ (1 .+ exp.(-0.3 .* df.x))
    model = glm(@formula(y ~ x + group), df, Binomial(), LogitLink())
    data_nt = Margins._prepare_margins_data(Tables.columntable(df), model)
    vars = [:x, :group]
    T(e) = typeof(e)

    # Population engines: exact evaluator types in the engine's type
    pop = Margins.build_engine(Margins.PopulationUsage, model, data_nt, vars, GLM.vcov, :ad)
    @test pop.de isa FormulaCompiler.ADEvaluator && pop.contrast isa FormulaCompiler.ContrastEvaluator
    @test isconcretetype(fieldtype(T(pop), :de)) && isconcretetype(fieldtype(T(pop), :contrast))
    # No continuous (or categorical) variable requested: no evaluator of that kind
    cat_only = Margins.build_engine(Margins.PopulationUsage, model, data_nt, [:group], GLM.vcov, :ad)
    @test cat_only.de === nothing && fieldtype(T(cat_only), :de) === Nothing
    # Profile engines evaluate on the reference grid, the scenarios/groups path per context:
    # neither carries evaluators over the data rows
    prof = Margins.build_engine(Margins.ProfileUsage, model, data_nt, vars, GLM.vcov, :ad)
    ctx = Margins.build_engine(Margins.PopulationUsage, model, data_nt, vars, GLM.vcov, :ad; evaluators=false)
    for e in (prof, ctx)
        @test e.de === nothing && e.contrast === nothing
        @test fieldtype(T(e), :de) === Nothing && fieldtype(T(e), :contrast) === Nothing
    end
    # ... and their results are unchanged
    @test DataFrame(profile_margins(model, df, means_grid(df); type=:effects, vars=vars)).estimate isa Vector{Float64}
    @test !isempty(DataFrame(population_margins(model, df; vars=[:x], scenarios=(group=["A", "B"],))).estimate)
end
