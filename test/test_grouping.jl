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

    # Test unified groups parameter - simple categorical (basic test first)
    @testset "Simple categorical groups" begin
        # First test without groups to make sure basic functionality works
        res_basic = population_margins(m, df; type=:effects, vars=[:x])
        @test nrow(DataFrame(res_basic)) == 1
        @test all(isfinite, DataFrame(res_basic).estimate)
        
        # Now test with groups
        res_groups = population_margins(m, df; type=:effects, vars=[:x], groups=:group1)
        @test nrow(DataFrame(res_groups)) >= 1  # May have multiple groups
        @test all(isfinite, DataFrame(res_groups).estimate)
    end

    # Test unified groups parameter - multiple categorical 
    @testset "Multiple categorical groups" begin
        res_multi = population_margins(m, df; type=:effects, vars=[:x], groups=[:group1, :group2])
        expected_rows = length(levels(df.group1)) * length(levels(df.group2))
        @test nrow(DataFrame(res_multi)) <= expected_rows  # May be fewer due to empty combinations
        @test haskey(DataFrame(res_multi), Symbol("at_group1"))
        @test haskey(DataFrame(res_multi), Symbol("at_group2"))
    end

    # Test unified groups parameter - continuous groups (quartiles)
    @testset "Continuous groups (quartiles)" begin
        res_continuous = population_margins(m, df; type=:effects, vars=[:z], groups=(:x, 4))
        @test nrow(DataFrame(res_continuous)) == 4  # Four quartiles
        @test haskey(DataFrame(res_continuous), Symbol("at_x"))
        @test all(isfinite, DataFrame(res_continuous).estimate)
        # Check that we have distinct x values (quartile midpoints)
        @test length(unique(DataFrame(res_continuous)[!, Symbol("at_x")])) == 4
    end

    # Test unified groups parameter - nested groups
    @testset "Nested groups" begin
        res_nested = population_margins(m, df; type=:effects, vars=[:x], groups=(main=:group1, within=:group2))
        expected_rows = length(levels(df.group1)) * length(levels(df.group2))
        @test nrow(DataFrame(res_nested)) <= expected_rows  # May be fewer due to empty combinations  
        @test haskey(DataFrame(res_nested), Symbol("at_group1"))
        @test haskey(DataFrame(res_nested), Symbol("at_group2"))
    end

    # Test unified groups parameter - predictions
    @testset "Groups with predictions" begin
        res_pred = population_margins(m, df; type=:predictions, groups=:group1)
        @test nrow(DataFrame(res_pred)) == length(levels(df.group1))
        @test haskey(DataFrame(res_pred), Symbol("at_group1"))
        @test all(isfinite, DataFrame(res_pred).estimate)
        @test all(DataFrame(res_pred).term .== "AAP")
    end
end