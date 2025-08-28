using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins

@testset "GLM basics: AME/MEM/MER and APE/APM/APR" begin
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

    # AME on eta and mu
    res_eta = ame(m, df; dydx=[:x], target=:eta)
    res_mu  = ame(m, df; dydx=[:x], target=:mu)
    @test nrow(res_eta.table) == 1
    @test nrow(res_mu.table) == 1
    @test all(isfinite, res_eta.table.dydx)
    @test all(isfinite, res_mu.table.dydx)

    # MEM equals MER at means
    mem_mu = mem(m, df; dydx=[:x], target=:mu)
    mer_mu = mer(m, df; dydx=[:x], target=:mu, at=:means)
    @test nrow(mem_mu.table) == nrow(mer_mu.table) == 1
    @test isapprox(mem_mu.table.dydx[1], mer_mu.table.dydx[1]; rtol=1e-6, atol=1e-8)

    # APE/APR
    ape_mu = ape(m, df; target=:mu)
    @test nrow(ape_mu.table) == 1
    apr_mu = apr(m, df; target=:mu, at=Dict(:x=>[-2.0,0.0,2.0]))
    @test nrow(apr_mu.table) == 3
    # average_profiles collapses grid
    apr_avg = apr(m, df; target=:mu, at=Dict(:x=>[-2.0,0.0,2.0]), average_profiles=true)
    @test nrow(apr_avg.table) == 1

    # Grouping: over and by
    over_res = ame(m, df; dydx=[:x], target=:mu, over=:g)
    @test nrow(over_res.table) == length(levels(df.g))
    @test haskey(over_res.table, :g)

    by_res = ame(m, df; dydx=[:x], target=:mu, by=:g)
    @test nrow(by_res.table) == length(levels(df.g))
    @test haskey(by_res.table, :g)

    # Weights and asbalanced
    w = rand(n)
    ame_w = ame(m, df; dydx=[:x], target=:mu, weights=w)
    @test nrow(ame_w.table) == 1
    ame_bal = ame(m, df; dydx=[:x], target=:mu, asbalanced=true)
    @test nrow(ame_bal.table) == 1
    # asbalanced over subset (only g)
    ame_bal_g = ame(m, df; dydx=[:x], target=:mu, asbalanced=[:g])
    @test nrow(ame_bal_g.table) == 1

    # vcov overrides (matrix and function) should not change point estimates
    Σ = vcov(m)
    ame_vΣ = ame(m, df; dydx=[:x], target=:mu, vcov=Σ)
    ame_vf = ame(m, df; dydx=[:x], target=:mu, vcov = m->vcov(m))
    @test isapprox(ame_vΣ.table.dydx[1], ame_vf.table.dydx[1]; rtol=1e-12)
end

