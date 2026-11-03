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

# 3.0 step 5: an η offset (GLM `offset=`, GLMM `offset=`, FixedEffectModels' absorbed fixed
# effects) enters every kernel through the engine's `offset::O` (`O = Vector{Float64}`), so the
# loops stay allocation-free and concretely typed with an offset too.
function _cat_loop_allocs(engine, var, pairs, rows, scale, weights)
    n = length(pairs); p = length(engine.β)
    ame = zeros(n); se = zeros(n); G = zeros(n, p)
    s = Margins.predscale(scale, engine.link)
    f() = Margins.categorical_contrast_ame_batch!(ame, se, G, engine.contrast_buf, engine.contrast, var,
                                                  pairs, engine.β, engine.Σ, s, rows, weights, engine.offset)
    f()
    return @allocated f()
end

function _aap_loop_allocs(engine, data_nt, rows, scale, weights)
    g = zeros(length(engine.β))
    s = Margins.predscale(scale, engine.link)
    tw = isnothing(weights) ? Float64(length(rows)) : sum(weights[rows])
    # function barrier: `s` is a Union of the two scales here, and a closure over it would
    # measure Julia 1.10's boxing of the dynamic call's Float64 result (16 B), not the loop
    return _aap_loop_allocs(g, engine, data_nt, rows, s, weights, tw)
end
function _aap_loop_allocs(g, engine, data_nt, rows, s::Margins.PredScale, weights, tw)
    f() = Margins.average_prediction!(g, engine.row_buf, engine.compiled, data_nt, engine.β, s, rows,
                                      weights, tw, engine.offset)
    f()
    return @allocated f()
end

@testset "Hot loops with an offset: zero allocation, concrete engine type" begin
    Random.seed!(11)
    n = 20_000
    df = DataFrame(x = randn(n), z = abs.(randn(n)) .+ 0.1, e = 0.5 .+ rand(n),
                   group = categorical(rand(["A", "B", "C"], n)))
    df.y = Float64.(rand(n) .< 0.3)
    model = glm(@formula(y ~ x * group + log(z)), df, Poisson(), LogLink(); offset=log.(df.e))
    data_nt = Margins._prepare_margins_data(Tables.columntable(df), model)
    off = Margins._offset(model, data_nt)
    @test off isa Vector{Float64} && length(off) == n
    weights = rand(n) .+ 0.1
    measures = (:effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx)
    for backend in (:fd, :ad)
        engine = Margins.build_engine(Margins.PopulationUsage, model, data_nt, [:x, :z, :group], GLM.vcov, backend;
                                      offset=off)
        @test engine.offset === off
        @test fieldtype(typeof(engine), :offset) === Vector{Float64}
        for scale in (:link, :response), w in (nothing, weights)
            for measure in measures
                @test _ame_loop_allocs(engine, [:x, :z], data_nt, 1:n, scale, measure, w) == 0
            end
            pairs = Margins.generate_contrast_pairs(data_nt.group, 1:n, :baseline, model, :group, data_nt)
            @test _cat_loop_allocs(engine, :group, pairs, 1:n, scale, w) == 0
            @test _aap_loop_allocs(engine, data_nt, 1:n, scale, w) == 0
        end
    end
    # and without one: the engine records `Nothing`
    e0 = Margins.build_engine(Margins.PopulationUsage, model, data_nt, [:x], GLM.vcov, :ad)
    @test e0.offset === nothing && fieldtype(typeof(e0), :offset) === Nothing
    for scale in (:link, :response)
        @test _cat_loop_allocs(Margins.build_engine(Margins.PopulationUsage, model, data_nt, [:group], GLM.vcov, :ad),
                               :group, Margins.generate_contrast_pairs(data_nt.group, 1:n, :baseline, model, :group, data_nt),
                               1:n, scale, nothing) == 0
        @test _aap_loop_allocs(e0, data_nt, 1:n, scale, nothing) == 0
    end
end
