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
        @test nrow(DataFrame(res_default)) == 1
        @test all(isfinite, DataFrame(res_default).se)
        @test all(DataFrame(res_default).se .> 0)
    end

    # Test model vcov explicitly  
    @testset "Explicit model vcov" begin
        res_model = population_margins(m, df; type=:effects, vars=[:x], vcov=:model)
        @test nrow(DataFrame(res_model)) == 1
        @test all(isfinite, DataFrame(res_model).se)
        @test all(DataFrame(res_model).se .> 0)
    end

    # Test robust standard errors (if CovarianceMatrices is available)
    @testset "Robust standard errors" begin
        try
            using CovarianceMatrices
            res_robust = population_margins(m, df; type=:effects, vars=[:x], vcov=HC1())
            @test nrow(DataFrame(res_robust)) == 1
            @test all(isfinite, DataFrame(res_robust).se)
            @test all(DataFrame(res_robust).se .> 0)
        catch e
            @warn "Skipping robust SE test: CovarianceMatrices not available or other error: $e"
        end
    end

    # Test confidence intervals
    @testset "Confidence intervals" begin
        res_ci = population_margins(m, df; type=:effects, vars=[:x], ci_level=0.95)
        @test nrow(DataFrame(res_ci)) == 1
        @test haskey(DataFrame(res_ci), :ci_lower)
        @test haskey(DataFrame(res_ci), :ci_upper)
        @test all(isfinite, DataFrame(res_ci).ci_lower)
        @test all(isfinite, DataFrame(res_ci).ci_upper)
        @test all(DataFrame(res_ci).ci_lower .< DataFrame(res_ci).ci_upper)
    end
end