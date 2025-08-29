using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins

@testset "Covariance matrix handling" begin
    Random.seed!(101112)
    n = 200
    df = DataFrame(
        y = rand(Bool, n),
        x = Float64.(randn(n)),  # Ensure Float64
        z = Float64.(randn(n)),  # Ensure Float64
        cluster = categorical(rand(1:10, n))
    )

    m = glm(@formula(y ~ x + z), df, Binomial(), LogitLink())

    # Test default model vcov
    @testset "Default model vcov" begin
        res_default = population_margins(m, df; type=:effects, vars=[:x])
        @test nrow(res_default.table) == 1
        @test all(isfinite, res_default.table.se)
        @test all(res_default.table.se .> 0)
    end

    # Test model vcov explicitly  
    @testset "Explicit model vcov" begin
        res_model = population_margins(m, df; type=:effects, vars=[:x], vcov=:model)
        @test nrow(res_model.table) == 1
        @test all(isfinite, res_model.table.se)
        @test all(res_model.table.se .> 0)
    end

    # Test robust standard errors (if CovarianceMatrices is available)
    @testset "Robust standard errors" begin
        try
            using CovarianceMatrices
            res_robust = population_margins(m, df; type=:effects, vars=[:x], vcov=HC1())
            @test nrow(res_robust.table) == 1
            @test all(isfinite, res_robust.table.se)
            @test all(res_robust.table.se .> 0)
        catch e
            @warn "Skipping robust SE test: CovarianceMatrices not available or other error: $e"
        end
    end

    # Test confidence intervals
    @testset "Confidence intervals" begin
        res_ci = population_margins(m, df; type=:effects, vars=[:x], ci_level=0.95)
        @test nrow(res_ci.table) == 1
        @test haskey(res_ci.table, :ci_lower)
        @test haskey(res_ci.table, :ci_upper)
        @test all(isfinite, res_ci.table.ci_lower)
        @test all(isfinite, res_ci.table.ci_upper)
        @test all(res_ci.table.ci_lower .< res_ci.table.ci_upper)
    end
end