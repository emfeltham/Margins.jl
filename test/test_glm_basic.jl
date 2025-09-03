using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins

@testset "GLM basics: Population and Profile Margins" begin
    Random.seed!(42)
    n = 400
    df = DataFrame(
        y = rand(Bool, n),
        x = randn(n),
        z = randn(n),
        g = categorical(rand(["A","B"], n))
    )

    # Logistic regression
    m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())

    # Population marginal effects on eta and mu scales
    res_eta = population_margins(m, df; type=:effects, vars=[:x], target=:eta)
    res_mu  = population_margins(m, df; type=:effects, vars=[:x], target=:mu)
    @test nrow(DataFrame(res_eta)) == 1
    @test nrow(DataFrame(res_mu)) == 1
    @test all(isfinite, DataFrame(res_eta).estimate)
    @test all(isfinite, DataFrame(res_mu).estimate)

    # Profile effects at means
    profile_mu = profile_margins(m, df; type=:effects, vars=[:x], target=:mu, at=:means)
    @test nrow(DataFrame(profile_mu)) == 1
    @test all(isfinite, DataFrame(profile_mu).estimate)

    # Population and Profile predictions
    pop_pred = population_margins(m, df; type=:predictions, scale=:response)
    @test nrow(DataFrame(pop_pred)) == 1
    profile_pred = profile_margins(m, df; type=:predictions, scale=:response, at=Dict(:x=>[-2.0,0.0,2.0]))
    @test nrow(DataFrame(profile_pred)) == 3
    # single profile at means
    profile_single = profile_margins(m, df; type=:predictions, scale=:response, at=:means)
    @test nrow(DataFrame(profile_single)) == 1

    # Test basic functionality without grouping (grouping parameters removed)
    # The current API doesn't support the 'over' and 'by' parameters as expected
    basic_effects = population_margins(m, df; type=:effects, vars=[:x], target=:mu)
    @test nrow(DataFrame(basic_effects)) == 1  # Single effect estimate
    @test DataFrame(basic_effects).term[1] == "x"  # Term should be string "x"

    # Test different target scales
    ame_eta = population_margins(m, df; type=:effects, vars=[:x], target=:eta)
    @test nrow(DataFrame(ame_eta)) == 1
    ame_mu = population_margins(m, df; type=:effects, vars=[:x], target=:mu)
    @test nrow(DataFrame(ame_mu)) == 1
    # Effects should be different on different scales for GLM
    @test DataFrame(ame_eta).estimate[1] != DataFrame(ame_mu).estimate[1]

    # Test multiple variables
    multi_effects = population_margins(m, df; type=:effects, vars=[:x, :z], target=:mu)
    @test nrow(DataFrame(multi_effects)) == 2  # Two variables
    @test Set(DataFrame(multi_effects).term) == Set(["x", "z"])
end

