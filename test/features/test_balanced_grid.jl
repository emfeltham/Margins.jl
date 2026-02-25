using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Statistics
using Margins

@testset "balanced_grid() reference grid builder" begin
    Random.seed!(06515)
    n = 300
    df = DataFrame(
        y = randn(n),
        x = Float64.(randn(n)),
        z = Float64.(randn(n)),
        g = categorical(rand(["A", "B", "C"], n)),
        region = categorical(rand(["North", "South"], n))
    )

    @testset "Basic balanced_grid with :all" begin
        grid = balanced_grid(df; g=:all)
        @test nrow(grid) == 1
        @test :g in propertynames(grid)
        @test :x in propertynames(grid)
        # Continuous variables should be at their means
        @test grid.x[1] ≈ mean(df.x) atol=1e-12
        @test grid.z[1] ≈ mean(df.z) atol=1e-12
    end

    @testset "balanced_grid with specific levels" begin
        grid = balanced_grid(df; g=["A", "B"])
        @test nrow(grid) == 1
        @test :g in propertynames(grid)
    end

    @testset "balanced_grid with multiple variables" begin
        grid = balanced_grid(df; g=:all, region=:all)
        @test nrow(grid) == 1
        @test :g in propertynames(grid)
        @test :region in propertynames(grid)
    end

    @testset "balanced_grid errors on missing variable" begin
        @test_throws ArgumentError balanced_grid(df; nonexistent=:all)
    end

    @testset "balanced_grid errors on invalid levels" begin
        @test_throws ArgumentError balanced_grid(df; g=["A", "MISSING"])
    end

    @testset "balanced_grid integrates with profile_margins" begin
        model = lm(@formula(y ~ x + z + g), df)

        grid = balanced_grid(df; g=:all)
        result = profile_margins(model, df, grid; type=:predictions, scale=:response)
        result_df = DataFrame(result)
        @test nrow(result_df) == 1
        @test all(isfinite, result_df.estimate)
        @test all(isfinite, result_df.se)
        @test all(result_df.se .> 0)

        # Effects should also work
        eff = profile_margins(model, df, grid; type=:effects, vars=[:x], scale=:link)
        eff_df = DataFrame(eff)
        @test nrow(eff_df) == 1
        @test all(isfinite, eff_df.estimate)
    end
end
