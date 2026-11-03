# test_result_metadata_contract.jl
# Every result carries the documented metadata keys with the documented types
# (Margins.result_metadata_keys), its inference (`result.inference`) and its context columns
# (`result.context`), whichever path produced it.

using Test
using Margins
using DataFrames
using GLM
using CategoricalArrays
using Random

@testset "result metadata contract" begin
    Random.seed!(31)
    n = 400
    df = DataFrame(x = randn(n), z = randn(n), t = rand([0.0, 1.0], n), g = categorical(rand(["a", "b", "c"], n)),
                   sw = 0.5 .+ rand(n))
    df.y = 0.3 .+ 0.5df.x .+ 0.2df.z .+ 0.4df.t .+ 0.3df.x .* df.t .+ randn(n)
    df.yb = df.y .> 0.3
    lin = lm(@formula(y ~ x * t + z + g), df)
    lg = glm(@formula(yb ~ x * t + z + g), df, Binomial(), LogitLink())
    lgw = glm(@formula(yb ~ x * t + z + g), df, Binomial(), LogitLink(); wts = df.sw)
    grid = DataFrame(z = [0.0, 1.0])

    produced = Dict(
        "AME" => population_margins(lg, df; vars = :x),
        "AME weighted" => population_margins(lg, df; vars = :x, weights = :sw),
        "AAP" => population_margins(lg, df; type = :predictions),
        "scenarios" => population_margins(lg, df; vars = :x, scenarios = (t = [0.0, 1.0],)),
        "groups" => population_margins(lg, df; vars = :x, groups = :g),
        "scenarios+groups+weights" => population_margins(lg, df; vars = :x, scenarios = (t = [0.0, 1.0],), groups = :g, weights = :sw),
        "categorical focal" => population_margins(lg, df; vars = :g),
        "link scale" => population_margins(lg, df; vars = :x, scale = :link),
        "elasticity" => population_margins(lg, df; vars = :x, measure = :elasticity),
        "linear model" => population_margins(lin, df; vars = :x),
        "profile effects" => profile_margins(lg, df, grid; vars = :x),
        "profile predictions" => profile_margins(lg, df, grid; type = :predictions),
        "profile linear" => profile_margins(lin, df, grid; vars = :x),
        "weighted model" => population_margins(lgw, df; vars = :x),
        "weighted model, contexts" => population_margins(lgw, df; vars = :x, scenarios = (t = [0.0, 1.0],)),
        "weighted model, :none" => population_margins(lgw, df; type = :predictions, weights = :none),
        "weighted model, profile" => profile_margins(lgw, df, grid; vars = :x),
    )

    for (label, r) in produced
        @testset "$label" begin
            for (key, T) in Margins.result_metadata_keys()
                @test haskey(r.metadata, key)
                haskey(r.metadata, key) && @test r.metadata[key] isa T
            end
            p = size(r.gradients, 2)
            @test r.inference isa Inference
            # the inference is not metadata (3.0 step 6b)
            @test !any(k -> haskey(r.metadata, k), (:vcov, :df, :alpha))
            V = r.inference.vcov
            @test size(V) == (p, p)
            @test V ≈ V'
            @test r.metadata[:analysis_type] in (:population, :profile)
            @test r.inference.alpha == 0.05
            @test r.metadata[:has_contexts] == (!isempty(r.metadata[:groups_vars]) || !isempty(r.metadata[:scenarios_vars]))
            # the context: every column one value per row; kinds match the analysis
            ctx = r.context
            if r.metadata[:analysis_type] == :profile
                @test ctx isa ResultContext && all(==(:grid), ctx.kinds)
            elseif r.metadata[:has_contexts]
                @test ctx isa ResultContext
                @test collect(keys(ctx)) == [r.metadata[:groups_vars]; r.metadata[:scenarios_vars]]
                @test ctx.kinds == [fill(:group, length(r.metadata[:groups_vars])); fill(:scenario, length(r.metadata[:scenarios_vars]))]
            else
                @test isnothing(ctx)
            end
            isnothing(ctx) || @test all(c -> length(c) == length(r.estimates), values(ctx.columns))
        end
    end

    @testset "specific values" begin
        @test produced["AME"].metadata[:weighted] == false
        @test produced["AME weighted"].metadata[:weighted] == true
        @test produced["scenarios+groups+weights"].metadata[:weighted] == true   # was missing on the contexts path
        @test produced["AME"].metadata[:weights_source] === :none
        @test produced["AME weighted"].metadata[:weights_source] === :user
        @test produced["scenarios+groups+weights"].metadata[:weights_source] === :user
        @test produced["weighted model"].metadata[:weights_source] === :model && produced["weighted model"].metadata[:weighted]
        @test produced["weighted model, contexts"].metadata[:weights_source] === :model
        @test produced["weighted model, :none"].metadata[:weights_source] === :none
        @test produced["weighted model, profile"].metadata[:weights_source] === :none   # profile margins are unweighted
        @test produced["profile effects"].metadata[:weights_source] === :none
        @test produced["scenarios+groups+weights"].metadata[:groups_vars] == [:g]
        @test produced["scenarios+groups+weights"].metadata[:scenarios_vars] == [:t]
        @test produced["AME"].metadata[:groups_vars] == Symbol[]
        @test produced["elasticity"].metadata[:measure] == :elasticity
        @test produced["link scale"].metadata[:scale] == :link
        @test produced["profile effects"].metadata[:analysis_type] == :profile
        @test occursin("GeneralizedLinearModel", produced["AME"].metadata[:model_type])  # was the string "DataType"
        # 3.0 step 5: df is a property of the model class (D1), so no result records the model's
        # own covariance and residual df for second differences to compare against any more
        @test !haskey(produced["linear model"].metadata, :model_vcov) && !haskey(produced["linear model"].metadata, :model_df)
        @test produced["linear model"].inference.df > 0 && isfinite(produced["linear model"].inference.df)
        @test produced["AME"].inference.df == Inf
    end

    @testset "accessors apply one default everywhere" begin
        r = produced["AME"]
        bare = Margins.EffectsResult(r.estimates, r.standard_errors, r.variables, r.terms, nothing, r.gradients,
                                     r.inference, Dict{Symbol,Any}())
        @test Margins._result_has_contexts(bare) == false
        @test Margins._result_is_weighted(bare) == false
        @test Margins._result_weights_source(bare) === :none
        @test Margins._result_weights_source(produced["weighted model"]) === :model
        @test Margins._result_analysis_type(bare) == :unknown
        # string-valued entries (a result restored from text) read as Symbols
        s = Margins.EffectsResult(r.estimates, r.standard_errors, r.variables, r.terms, nothing, r.gradients,
                                  r.inference, Dict{Symbol,Any}(:analysis_type => "population"))
        @test Margins._result_analysis_type(s) == :population
        # a result's alpha and df come from its inference, whatever its metadata holds
        @test DataFrame(bare).df == fill(Inf, length(r.estimates))
        @test DataFrame(bare).ci_lower ≈ DataFrame(r).ci_lower
    end

    @testset "Inference and ResultContext validate their contents" begin
        V = [1.0 0.0; 0.0 1.0]
        @test Inference(V, 10, 0.1) isa Inference && Inference(V, 10, 0.1).df === 10.0
        @test Inference(V, Inf, 0.05).df == Inf
        @test_throws ArgumentError Inference(V, 0, 0.05)
        @test_throws ArgumentError Inference(V, NaN, 0.05)
        @test_throws ArgumentError Inference(V, 5, 1.0)
        @test_throws ArgumentError Inference(ones(2, 3), 5, 0.05)
        ctx = ResultContext((b = [1, 2], a = ["u", "v"]), [:group, :scenario])
        @test collect(keys(ctx)) == [:b, :a] && ctx[:a] == ["u", "v"]
        @test Margins.context_kind(ctx, :b) === :group && Margins.context_kind(ctx, :a) === :scenario
        @test_throws ArgumentError Margins.context_kind(ctx, :zz)
        @test_throws ArgumentError ResultContext((a = [1, 2],), [:other])
        @test_throws ArgumentError ResultContext((a = [1, 2],), Symbol[])
        @test_throws ArgumentError ResultContext((a = [1, 2], b = [1]), [:grid, :grid])
    end
end
