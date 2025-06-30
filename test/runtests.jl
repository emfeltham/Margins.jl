# runtests.jl

using Revise
using Margins
using Test
using Random
using DataFrames, CategoricalArrays
using Distributions, Statistics, GLM, MixedModels
using RDatasets
import LinearAlgebra.dot
import LinearAlgebra.diag

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
