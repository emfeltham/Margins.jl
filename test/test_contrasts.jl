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
        @test nrow(DataFrame(cat_effects)) >= 1  # Should have contrasts
        @test all(isfinite, DataFrame(cat_effects).estimate)
        @test "cat_var" in DataFrame(cat_effects).term
    end

    # Test binary categorical effects  
    @testset "Binary categorical effects" begin
        binary_effects = population_margins(m, df; type=:effects, vars=[:binary_var])
        @test nrow(DataFrame(binary_effects)) >= 1
        @test all(isfinite, DataFrame(binary_effects).estimate)
        @test "binary_var" in DataFrame(binary_effects).term
    end

    # Test mixed continuous and categorical
    @testset "Mixed variable types" begin
        mixed_effects = population_margins(m, df; type=:effects, vars=[:x, :cat_var])
        @test nrow(DataFrame(mixed_effects)) >= 2  # At least one continuous + categorical contrasts
        @test all(isfinite, DataFrame(mixed_effects).estimate)
        @test "x" in DataFrame(mixed_effects).term
        @test any(contains.("cat_var", DataFrame(mixed_effects).term))
    end

    # Test contrasts parameter
    @testset "Contrasts parameter" begin
        pairwise_effects = population_margins(m, df; type=:effects, vars=[:cat_var], contrasts=:pairwise)
        @test nrow(DataFrame(pairwise_effects)) >= 1
        @test all(isfinite, DataFrame(pairwise_effects).estimate)
    end
end