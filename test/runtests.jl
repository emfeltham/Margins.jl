# runtests.jl

using Revise
using Margins

begin
    using Test
    using Random
    using DataFrames, CategoricalArrays
    using Distributions, Statistics, GLM, MixedModels
    using RDatasets
    import LinearAlgebra.dot
    import LinearAlgebra.diag
    using StandardizedPredictors
end

# Load data
iris = dataset("datasets", "iris") |> DataFrame;
iris.Species = categorical(iris.Species);

Revise.retry()

@testset "Margins.jl" begin
    include("bool_test.jl") # linear models with booleans
    include("lm_tests.jl") # linear models
    include("glm_tests.jl") # general linear models
    include("mm_tests.jl") # general mixed models
    include("additional_tests.jl")
    include("df_tests.jl") # DataFrame conversion
    include("contrast_tests.jl") # contrasts
    include("large_tests.jl") # larger data with repvals
end;


@testset "3-way interaction OLS (no repvals)" begin
    # simulate data
    Random.seed!(42)
    n = 500
    x = randn(n)
    d = rand(n) .> 0.5       # Bool
    z = randn(n)
    # true model: y = β0 + βx x + βd d + βz z + βxd x*d + βxz x*z + βdz d*z + βxdz x*d*z + ε
    β = (β0=1.0, βx=2.0, βd=-1.5, βz=0.5, βxd=0.8, βxz=1.2, βdz=-0.7, βxdz=0.4)
    μ = β.β0 .+ β.βx*x .+ β.βd*(d .== true) .+ β.βz*z .+
        β.βxd*(x .* (d .== true)) .+ β.βxz*(x .* z) .+ β.βdz*((d .== true) .* z) .+
        β.βxdz*(x .* (d .== true) .* z)
    y = μ .+ randn(n)*0.1
    df = DataFrame(y=y, x=x, d=CategoricalArray(d), z=z)

    # fit model
    m = lm(@formula(y ~ x * d * z), df)

    # analytic AME (average derivative) and SE via delta method
    # derivative for each obs: ∂μ/∂x = βx + βxd*d + βxz*z + βxdz*d*z
    d_num = Float64.(d)           # 1 for true, 0 for false
    dz = d_num .* z
    # average derivative
    ame_closed = β.βx + β.βxd*mean(d_num) + β.βxz*mean(z) + β.βxdz*mean(dz)

    # SE: var(c'β̂) with c = [0, 1, mean(d), mean(z), mean(dz), ...] matching coef order
    cn = coefnames(m)
    coefs = coef(m)
    V = vcov(m)
    # build contrast vector
    c = zeros(length(coefs))
    c[findfirst(isequal("x"), cn)]      = 1
    c[findfirst(isequal("x & d: true"), cn)]  = mean(d_num)
    c[findfirst(isequal("x & z"), cn)]  = mean(z)
    c[findfirst(isequal("x & d: true & z"), cn)] = mean(dz)
    se_closed = sqrt(c' * V * c)

    # compute margins without repvals (defaults to sample average)
    ame_out = margins(m, :x, df)

    # tests
    @test isapprox(ame_out.effects[:x], ame_closed; atol=1e-6)
    @test isapprox(ame_out.ses[:x], se_closed; atol=1e-6)
end
