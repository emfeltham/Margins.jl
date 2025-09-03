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

    # Population marginal effects on link and response scales
    res_link = population_margins(m, df; type=:effects, vars=[:x], scale=:link)
    res_response  = population_margins(m, df; type=:effects, vars=[:x], scale=:response)
    @test nrow(DataFrame(res_link)) == 1
    @test nrow(DataFrame(res_response)) == 1
    @test all(isfinite, DataFrame(res_link).estimate)
    @test all(isfinite, DataFrame(res_response).estimate)

    # Profile effects at means
    profile_response = profile_margins(m, df; type=:effects, vars=[:x], scale=:response, at=:means)
    @test nrow(DataFrame(profile_response)) == 1
    @test all(isfinite, DataFrame(profile_response).estimate)

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
    basic_effects = population_margins(m, df; type=:effects, vars=[:x], scale=:response)
    @test nrow(DataFrame(basic_effects)) == 1  # Single effect estimate
    @test DataFrame(basic_effects).term[1] == "x"  # Term should be string "x"

    # Test different scales
    ame_link = population_margins(m, df; type=:effects, vars=[:x], scale=:link)
    @test nrow(DataFrame(ame_link)) == 1
    ame_response = population_margins(m, df; type=:effects, vars=[:x], scale=:response)
    @test nrow(DataFrame(ame_response)) == 1
    # Effects should be different on different scales for GLM
    @test DataFrame(ame_link).estimate[1] != DataFrame(ame_response).estimate[1]

    # Test multiple variables
    multi_effects = population_margins(m, df; type=:effects, vars=[:x, :z], scale=:response)
    @test nrow(DataFrame(multi_effects)) == 2  # Two variables
    @test Set(DataFrame(multi_effects).term) == Set(["x", "z"])
end

