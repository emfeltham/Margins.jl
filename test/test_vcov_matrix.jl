# test_vcov_matrix.jl
# A covariance matrix as `vcov` in population_margins / profile_margins

using Test
using Margins
using DataFrames
using GLM
using LinearAlgebra: Symmetric
using Random

@testset "vcov as a matrix" begin
    Random.seed!(5)
    n = 500
    df = DataFrame(x = randn(n), z = rand([0.0, 1.0], n))
    df.y = 1 .+ 0.5df.x .+ 0.4df.z .+ 0.3df.x .* df.z .+ randn(n)
    m = lm(@formula(y ~ x * z), df)
    V = Matrix(GLM.vcov(m))
    se(r) = r.standard_errors

    @testset "same answer as the equivalent function" begin
        a = population_margins(m, df; vars = :x, vcov = V)
        b = population_margins(m, df; vars = :x, vcov = _ -> V)
        @test se(a) ≈ se(b)
        @test se(a) ≈ se(population_margins(m, df; vars = :x))
        @test se(population_margins(m, df; vars = :x, vcov = Symmetric(V))) ≈ se(a)
        p = profile_margins(m, df, DataFrame(z = [0.0]); vars = :x, vcov = 4 .* V)
        @test se(p)[1] ≈ 2 * se(profile_margins(m, df, DataFrame(z = [0.0]); vars = :x))[1]
    end

    @testset "changing the matrix in place is always picked up" begin
        M = copy(V)
        s1 = se(population_margins(m, df; vars = :x, vcov = M))
        M .*= 4
        @test se(population_margins(m, df; vars = :x, vcov = M)) ≈ 2 .* s1
        sc(v) = se(population_margins(m, df; vars = :x, vcov = v, scenarios = (z = [0.0, 1.0],), type = :effects))
        base = sc(V)
        M .= V
        @test sc(M) ≈ base
        M .*= 16
        @test sc(M) ≈ 4 .* base
    end

    @testset "t for the model's own matrix, z otherwise" begin
        @test population_margins(m, df; vars = :x, vcov = V).metadata[:df] == dof_residual(m)
        @test population_margins(m, df; vars = :x, vcov = 1.5 .* V).metadata[:df] == Inf
    end

    @testset "invalid matrices are rejected" begin
        @test_throws ArgumentError population_margins(m, df; vars = :x, vcov = V[1:3, 1:3])
        @test_throws ArgumentError population_margins(m, df; vars = :x, vcov = V[:, 1:3])
        W = copy(V); W[1, 2] += 1.0
        @test_throws ArgumentError population_margins(m, df; vars = :x, vcov = W)
        N = copy(V); N[1, 1] = NaN
        @test_throws ArgumentError population_margins(m, df; vars = :x, vcov = N)
    end
end
