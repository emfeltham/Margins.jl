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
        @test nrow(cat_effects.table) >= 1  # Should have contrasts
        @test all(isfinite, cat_effects.table.dydx)
        @test "cat_var" in cat_effects.table.term
    end

    # Test binary categorical effects  
    @testset "Binary categorical effects" begin
        binary_effects = population_margins(m, df; type=:effects, vars=[:binary_var])
        @test nrow(binary_effects.table) >= 1
        @test all(isfinite, binary_effects.table.dydx)
        @test "binary_var" in binary_effects.table.term
    end

    # Test mixed continuous and categorical
    @testset "Mixed variable types" begin
        mixed_effects = population_margins(m, df; type=:effects, vars=[:x, :cat_var])
        @test nrow(mixed_effects.table) >= 2  # At least one continuous + categorical contrasts
        @test all(isfinite, mixed_effects.table.dydx)
        @test "x" in mixed_effects.table.term
        @test any(contains.("cat_var", mixed_effects.table.term))
    end

    # Test contrasts parameter
    @testset "Contrasts parameter" begin
        pairwise_effects = population_margins(m, df; type=:effects, vars=[:cat_var], contrasts=:pairwise)
        @test nrow(pairwise_effects.table) >= 1
        @test all(isfinite, pairwise_effects.table.dydx)
    end
end