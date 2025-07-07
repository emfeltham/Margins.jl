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
end

# Load data
iris = dataset("datasets", "iris") |> DataFrame;
iris.Species = categorical(iris.Species);

@testset "Margins.jl" begin
    include("bool_test.jl") # linear models with booleans
    include("lm_tests.jl") # linear models
    include("glm_tests.jl") # general linear models
    include("mm_tests.jl") # general mixed models
    include("additional_tests.jl")
    include("df_tests.jl") # DataFrame conversion
    include("contrast_tests.jl") # contrasts
end

@testset "check modelmatrix!" begin
    using DataFrames, StatsModels, Random

    # ------------------------------------------------ toy data
    Random.seed!(42)
    df = DataFrame(x = randn(1_000), z = randn(1_000))
    df.y = 1 .+ 2df.x .- 0.5df.z .+ randn(1_000)

    f  = @formula(y ~ x + z + x & z)
    ml = lm(f, df)

    # ------------------------------------------------ one-time build
    mf   = ModelFrame(f, df)          # parses formula & stores the schema
    mm   = ModelMatrix(mf)            # allocates the design matrix once
    Xbuf = copy(mm.m)                 # working buffer you will keep overwriting
    # frhs = mf.f.rhs (this or that, but check)
    frhs  = formula(ml).rhs   # the baked RHS term

    Xbuf .= 0

    # ------------------------------------------------ fast loop
    # ... mutate some columns in `df` here ...
    modelmatrix!(Xbuf, frhs, Tables.columntable(df))
    @test Xbuf == mm.m
end

#######

# categorical_with_repvals_test.jl

using Test, RDatasets, DataFrames, CategoricalArrays
using GLM, Margins

@testset "Categorical main effect with continuous repvals" begin
    # 1) load & prepare
    df = dataset("datasets", "iris")
    df.Outcome = df.SepalLength .> mean(df.SepalLength)
    df.Species = categorical(df.Species)

    # 2) fit a simple logit with a continuous covariate
    m = glm(@formula(Outcome ~ Species * SepalWidth),
            df, Binomial(), LogitLink())

    # 3) try to get Species margins at two SepalWidth values
    rep = Dict(:SepalWidth => [quantile(df.SepalWidth, .25),
                                quantile(df.SepalWidth, .75)])
    mg = margins(m, :Species, df; repvals = rep)
    DataFrame(mg)
end

# fix this text, the logic is broken, likely the numbers are correct
# @testset "Categorical main effect with continuous repvals" begin
#     # 1) load & prepare
#     df = dataset("datasets", "iris")
#     df.Outcome = df.SepalLength .> mean(df.SepalLength)
#     df.Species = categorical(df.Species)

#     # 2) fit a simple logit with a continuous covariate
#     m = glm(@formula(Outcome ~ Species * SepalWidth),
#             df, Binomial(), LogitLink())

#     # 3) try to get Species margins at two SepalWidth values
#     rep = Dict(:SepalWidth => [quantile(df.SepalWidth, .25),
#                                 quantile(df.SepalWidth, .75)])
#     mg = margins(m, :Species, df; repvals = rep)
#     mg1 = margins(m, :Species, df)
    
#     # 4) Manual calculation for verification
#     coef_vals = coef(m)
#     repvals = [quantile(df.SepalWidth, 0.25), quantile(df.SepalWidth, 0.75)]
#     species_levels = levels(df.Species)
    
#     # Design matrix function for each species and sepalwidth combination
#     function design_matrix(species, sepalwidth, species_levels)
#         # Intercept
#         intercept = 1.0
#         # Species dummies (reference is first level - setosa)
#         sp_versicolor = species == species_levels[2] ? 1.0 : 0.0
#         sp_virginica = species == species_levels[3] ? 1.0 : 0.0
#         # SepalWidth
#         sw = sepalwidth
#         # Interaction terms
#         sp_versicolor_sw = sp_versicolor * sw
#         sp_virginica_sw = sp_virginica * sw
#         return [intercept, sp_versicolor, sp_virginica, sw, sp_versicolor_sw, sp_virginica_sw]
#     end
    
#     # Calculate expected margins manually
#     expected_margins = Dict()
#     for sp in species_levels
#         expected_margins[sp] = []
#         for sw in repvals
#             X = design_matrix(sp, sw, species_levels)
#             linear_pred = dot(coef_vals, X)
#             prob = 1 / (1 + exp(-linear_pred))  # logistic function
#             push!(expected_margins[sp], prob)
#         end
#     end
    
#     # Test that computed margins match expected values
#     for (i, sp) in enumerate(species_levels)
#         for (j, sw) in enumerate(repvals)
#             @test mg.margins[i, j] ≈ expected_margins[sp][j] atol=1e-6
#         end
#     end
# end

@testset "large data OLS" begin
    # simulate
    Random.seed!(42)
    n = 2_000_000
    x = randn(n)
    d = rand(n) .> 0.5       # Bool
    z = randn(n)
    # true model: y = β0 + βx x + βd d + βz z + βxd x*d + βxz x*z + βdz d*z + βxdz x*d*z + ε
    β = (β0=1.0, βx=2.0, βd= -1.5, βz=0.5, βxd=0.8, βxz=1.2, βdz=-0.7, βxdz=0.4)
    μ = β.β0 .+ β.βx*x .+ β.βd*(d .== true) .+ β.βz*z .+
        β.βxd*(x .* (d .== true)) .+ β.βxz*(x .* z) .+ β.βdz*((d .== true) .* z) .+
        β.βxdz*(x .* (d .== true) .* z)
    y = μ .+ randn(n)*0.1
    df = DataFrame(y=y, x=x, d=CategoricalArray(d), z=z)

    m = lm(@formula(y ~ x * d * z), df)
    # AME of x at d=false, z= mean(z)
    zv = mean(z)
    # closed-form derivative: ∂μ/∂x = βx + βxd*d + βxz*z + βxdz*d*z
    # at d=false => d=0: deriv = βx + βxz*zv
    ame_closed = β.βx + β.βxz*zv
    # SE via delta method: g = ∂β/∂x vector; compute var(g'β̂)
    cn = coefnames(m); coefs = coef(m); V = vcov(m)
    # build contrast c
    c = zeros(length(coefs))
    # intercept excluded
    iβx  = findfirst(isequal("x"), cn)
    iβxz = findfirst(isequal("x & z"), cn)
    c[iβx]  = 1
    c[iβxz] = zv
    se_closed = sqrt(c' * V * c)

    ame   = margins(m, :x, df; repvals=Dict(:d => categorical([false]), :z => [zv]))

    # fails at high tolerance (above 1e-2)
    # @test isapprox(ame.effects[:x][(false, zv)], ame_closed; atol=1e-6)

    # instead use estimated coefficients
    β̂ = coef(m)
    ame_closed_est = β̂[iβx] + β̂[iβxz] * zv
    @test isapprox(ame.effects[:x][(false, zv)], ame_closed_est; atol=1e-6)

    @test isapprox(ame.ses[:x][(false, zv)], se_closed; atol=1e-6)
end
