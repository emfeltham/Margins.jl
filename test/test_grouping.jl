using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins

@testset "Grouping functionality" begin
    Random.seed!(456)
    n = 300
    df = DataFrame(
        y = rand(Bool, n),
        x = Float64.(randn(n)),  # Ensure Float64
        z = Float64.(randn(n)),  # Ensure Float64  
        group1 = categorical(rand(["A","B"], n)),
        group2 = categorical(rand(["X","Y","Z"], n))
    )

    m = glm(@formula(y ~ x + z + group1), df, Binomial(), LogitLink())

    # Test over parameter
    @testset "Over parameter" begin
        res_over = population_margins(m, df; type=:effects, vars=[:x], over=:group1)
        @test nrow(res_over.table) == length(levels(df.group1))
        @test haskey(res_over.table, :group1)
        @test all(isfinite, res_over.table.dydx)
    end

    # Test by parameter  
    @testset "By parameter" begin
        res_by = population_margins(m, df; type=:effects, vars=[:x], by=:group1)
        @test nrow(res_by.table) == length(levels(df.group1))
        @test haskey(res_by.table, :group1)
        @test all(isfinite, res_by.table.dydx)
    end

    # Test within parameter
    @testset "Within parameter" begin
        res_within = population_margins(m, df; type=:effects, vars=[:x], within=:group1)
        @test nrow(res_within.table) == length(levels(df.group1))
        @test haskey(res_within.table, :group1) 
        @test all(isfinite, res_within.table.dydx)
    end

    # Test multiple grouping variables
    @testset "Multiple grouping" begin
        res_multi = population_margins(m, df; type=:effects, vars=[:x], over=[:group1, :group2])
        expected_rows = length(levels(df.group1)) * length(levels(df.group2))
        @test nrow(res_multi.table) <= expected_rows  # May be fewer due to empty combinations
        @test haskey(res_multi.table, :group1)
        @test haskey(res_multi.table, :group2)
    end
end