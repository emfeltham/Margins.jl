using Test
using Random
using DataFrames, CategoricalArrays
using Statistics
using GLM
using StatsModels
using Margins

include("bootstrap_se_validation.jl")

@testset "Bootstrap SE: population_margins with scenarios (continuous)" begin
    Random.seed!(123)
    n = 400
    x = randn(n)
    z = randn(n)
    β0, β1, β2 = 1.0, 0.8, -0.5
    y = β0 .+ β1 .* x .+ β2 .* z .+ 0.5 .* randn(n)
    df = DataFrame(y=y, x=x, z=z)
    model_func = lm
    formula = @formula(y ~ x + z)

    # Compute SEs under scenario at(z=z0) for effect of x on link scale
    original_model = model_func(formula, df)
    res = population_margins(original_model, df; type=:effects, vars=[:x], scenarios=Dict(:z => 0.5), scale=:link)
    dfr = DataFrame(res)
    computed_ses = dfr.se

    # Bootstrap under same scenario
    _, boot_ses, n_successful = bootstrap_margins_computation(
        model_func, formula, df, population_margins;
        n_bootstrap=150, vars=[:x], type=:effects, scenarios=Dict(:z => 0.5), scale=:link
    )

    validation = validate_bootstrap_se_agreement(computed_ses, boot_ses; tolerance=0.20, var_names=["x @ z=0.5"]) 
    @test n_successful >= 100  # at least 2/3 success
    @test validation.agreement_rate >= 0.8
end

@testset "Bootstrap SE: population_margins with scenarios (categorical)" begin
    Random.seed!(234)
    n = 500
    g = categorical(rand(["A","B","C"], n))
    x = randn(n)
    # Construct linear outcome with category effects
    y = 2 .+ (g .== "B") .* 0.6 .+ (g .== "C") .* (-0.4) .+ 0.3 .* x .+ 0.5 .* randn(n)
    df = DataFrame(y=y, x=x, g=g)
    model_func = lm
    formula = @formula(y ~ x + g)

    original_model = model_func(formula, df)
    res = population_margins(original_model, df; type=:effects, vars=[:g], scenarios=Dict(:x => 0.0), scale=:link)
    dfr = DataFrame(res)
    computed_ses = dfr.se

    _, boot_ses, n_successful = bootstrap_margins_computation(
        model_func, formula, df, population_margins;
        n_bootstrap=200, vars=[:g], type=:effects, scenarios=Dict(:x => 0.0), scale=:link
    )

    # Categorical often noisier; tolerance looser
    validation = validate_bootstrap_se_agreement(computed_ses, boot_ses; tolerance=0.25, var_names=dfr.contrast)
    @test n_successful >= 140
    @test validation.agreement_rate >= 0.7
end

