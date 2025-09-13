using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins

@testset "Categorical contrasts" begin
    Random.seed!(789)
    n = 250
    df = DataFrame(
        y = Float64.(randn(n)),  # Ensure Float64
        x = Float64.(randn(n)),  # Ensure Float64
        cat_var = categorical(rand(["Low","Medium","High"], n)),
        binary_var = categorical(rand(["Yes","No"], n))
    )

    m = lm(@formula(y ~ x + cat_var + binary_var), df)

    # Test categorical effects
    @testset "Categorical marginal effects" begin
        cat_effects = population_margins(m, df; type=:effects, vars=[:cat_var])
        df_results = DataFrame(cat_effects)
        @debug "Categorical effects computation" n_contrasts=nrow(df_results) estimates=df_results.estimate terms=df_results.variable all_finite=all(isfinite, df_results.estimate)
        @test nrow(DataFrame(cat_effects)) >= 1  # Should have contrasts
        @test all(isfinite, DataFrame(cat_effects).estimate)
        @test any(contains.(DataFrame(cat_effects).variable, "cat_var"))
    end

    # Test binary categorical effects  
    @testset "Binary categorical effects" begin
        binary_effects = population_margins(m, df; type=:effects, vars=[:binary_var])
        df_results = DataFrame(binary_effects)
        @debug "Binary categorical effects computation" n_contrasts=nrow(df_results) estimates=df_results.estimate terms=df_results.variable all_finite=all(isfinite, df_results.estimate)
        @test nrow(DataFrame(binary_effects)) >= 1
        @test all(isfinite, DataFrame(binary_effects).estimate)
        @test any(contains.(DataFrame(binary_effects).variable, "binary_var"))
    end

    # Test mixed continuous and categorical
    @testset "Mixed variable types" begin
        mixed_effects = population_margins(m, df; type=:effects, vars=[:x, :cat_var])
        df_results = DataFrame(mixed_effects)
        @debug "Mixed variable types computation" n_effects=nrow(df_results) estimates=df_results.estimate terms=df_results.variable continuous_present=("x" in df_results.variable) categorical_present=any(contains.("cat_var", df_results.variable))
        @test nrow(DataFrame(mixed_effects)) >= 2  # At least one continuous + categorical contrasts
        @test all(isfinite, DataFrame(mixed_effects).estimate)
        @test "x" in DataFrame(mixed_effects).variable
        @test any(contains.(DataFrame(mixed_effects).variable, "cat_var"))
    end

    # Test contrasts parameter
    @testset "Contrasts parameter" begin
        pairwise_effects = population_margins(m, df; type=:effects, vars=[:cat_var], contrasts=:pairwise)
        @test nrow(DataFrame(pairwise_effects)) >= 1
        @test all(isfinite, DataFrame(pairwise_effects).estimate)
    end
end