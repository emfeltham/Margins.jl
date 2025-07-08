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

@testset "Margins.jl" begin
    include("bool_test.jl") # linear models with booleans
    include("lm_tests.jl") # linear models
    include("glm_tests.jl") # general linear models
    include("mm_tests.jl") # general mixed models
    include("additional_tests.jl")
    include("df_tests.jl") # DataFrame conversion
    include("contrast_tests.jl") # contrasts
    include("large_tests.jl") # larger data
end;
