using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins

@testset "GLM basics: Population and Profile Margins" begin
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

    # Population marginal effects on eta and mu scales
    res_eta = population_margins(m, df; type=:effects, vars=[:x], target=:eta)
    res_mu  = population_margins(m, df; type=:effects, vars=[:x], target=:mu)
    @test nrow(res_eta.table) == 1
    @test nrow(res_mu.table) == 1
    @test all(isfinite, res_eta.table.dydx)
    @test all(isfinite, res_mu.table.dydx)

    # Profile effects at means
    profile_mu = profile_margins(m, df; type=:effects, vars=[:x], target=:mu, at=:means)
    @test nrow(profile_mu.table) == 1
    @test all(isfinite, profile_mu.table.dydx)

    # Population and Profile predictions
    pop_pred = population_margins(m, df; type=:predictions, scale=:response)
    @test nrow(pop_pred.table) == 1
    profile_pred = profile_margins(m, df; type=:predictions, scale=:response, at=Dict(:x=>[-2.0,0.0,2.0]))
    @test nrow(profile_pred.table) == 3
    # averaged profiles
    profile_avg = profile_margins(m, df; type=:predictions, scale=:response, at=Dict(:x=>[-2.0,0.0,2.0]), average=true)
    @test nrow(profile_avg.table) == 1

    # Grouping: over and by
    over_res = population_margins(m, df; type=:effects, vars=[:x], target=:mu, over=:g)
    @test nrow(over_res.table) == length(levels(df.g))
    @test "g" in names(over_res.table)

    by_res = population_margins(m, df; type=:effects, vars=[:x], target=:mu, by=:g)
    @test nrow(by_res.table) == length(levels(df.g))
    @test "g" in names(by_res.table)

    # Weights and balance
    w = rand(n)
    ame_w = population_margins(m, df; type=:effects, vars=[:x], target=:mu, weights=w)
    @test nrow(ame_w.table) == 1
    ame_bal = population_margins(m, df; type=:effects, vars=[:x], target=:mu, balance=:all)
    @test nrow(ame_bal.table) == 1
    # balance over subset (only g)
    ame_bal_g = population_margins(m, df; type=:effects, vars=[:x], target=:mu, balance=[:g])
    @test nrow(ame_bal_g.table) == 1

    # vcov overrides (matrix and function) should not change point estimates
    Σ = vcov(m)
    ame_vΣ = population_margins(m, df; type=:effects, vars=[:x], target=:mu, vcov=Σ)
    ame_vf = population_margins(m, df; type=:effects, vars=[:x], target=:mu, vcov = m->vcov(m))
    @test isapprox(ame_vΣ.table.dydx[1], ame_vf.table.dydx[1]; rtol=1e-12)
end

