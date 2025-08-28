using Test
using DataFrames, CategoricalArrays, GLM
using Margins

@testset "Margins basic GLM" begin
    df = DataFrame(y = rand(Bool, 300), x = randn(300), z = randn(300), g = categorical(rand(["A","B"], 300)))
    m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())

    # AME (η/μ) basic
    res_eta = ame(m, df; dydx=[:x], target=:eta)
    res_mu  = ame(m, df; dydx=[:x], target=:mu)
    @test nrow(res_eta.table) == 1
    @test haskey(res_eta.table, :dydx)
    @test nrow(res_mu.table) == 1

    # MER
    res_mer = mer(m, df; dydx=[:x], target=:mu, at=Dict(:x=>[-1,0,1], :g=>["A","B"]))
    @test nrow(res_mer.table) == 6  # 3 x values × 1 var × 2 g levels

    # APE/APR
    res_ape = ape(m, df; target=:mu)
    @test nrow(res_ape.table) == 1
    res_apr = apr(m, df; target=:mu, at=Dict(:x=>[-2,0,2]))
    @test nrow(res_apr.table) == 3

    # asbalanced weighting
    res_bal = ape(m, df; target=:mu, asbalanced=true)
    @test nrow(res_bal.table) == 1

    # vcov override (use model vcov explicitly)
    Σ = vcov(m)
    res_vcov = ame(m, df; dydx=[:x], vcov=Σ)
    @test nrow(res_vcov.table) == 1
end

