using Margins
using Test
using DataFrames, CategoricalArrays
using Statistics, GLM, MixedModels
using RDatasets
import LinearAlgebra.dot

@testset "Margins.jl" begin
    # Write your tests here.
    include("lm_tests.jl")
end

