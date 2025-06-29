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

@testset "Margins.jl" begin
    include("bool_test.jl") # linear models with booleans
    include("lm_tests.jl") # linear models
    include("glm_tests.jl") # general linear models
    include("mm_tests.jl") # general mixed models
end
