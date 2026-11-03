# test/regression/test_int_column_conversion.jl
#
# Continuous Int (and Float32) columns are converted to a Float64 copy on every call
# (`_convert_numeric_to_float64`). Results must equal those on Float64 data, be identical on
# repeated calls, and pick up in-place edits of the source column. (Through 3.0 development
# the copy was memoized per source array to keep the removed engine cache hitting; with the
# cache gone the copy is per call.)

using Test
using DataFrames, GLM, StatsModels, Random, Statistics, Tables
using Margins

@testset "continuous Int/Float32 columns" begin
    Random.seed!(3); n = 2_000
    d = DataFrame(x = randn(n), k = rand(0:20, n), f = Float32.(randn(n)))
    d.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.3 .* d.x .+ 0.05 .* d.k .- 0.2 .* d.f)))
    m = glm(@formula(y ~ x + k + f), d, Binomial(), LogitLink())

    r1 = DataFrame(population_margins(m, d))
    r2 = DataFrame(population_margins(m, d))
    @test r1.estimate == r2.estimate
    @test r1.se == r2.se

    # Reference: the same data with Float64 columns
    dd = transform(d, :k => ByRow(Float64) => :k, :f => ByRow(Float64) => :f)
    rf = DataFrame(population_margins(m, dd))
    @test r1.estimate ≈ rf.estimate rtol=1e-12
    @test r1.se ≈ rf.se rtol=1e-12

    # In-place mutation of the source is picked up (and matches a fresh Float64 frame)
    d.k[1:200] .= 20
    dd.k[1:200] .= 20.0
    r4 = DataFrame(population_margins(m, d))
    @test r4.estimate != r1.estimate
    @test r4.estimate ≈ DataFrame(population_margins(m, dd)).estimate rtol=1e-12
    d.f[1:200] .= 2.5f0
    dd.f[1:200] .= 2.5
    @test DataFrame(population_margins(m, d)).estimate ≈ DataFrame(population_margins(m, dd)).estimate rtol=1e-12

    # Values that do not round-trip through Float64 (|k| > 2^53) are converted afresh
    big = [2^53 + 1, 3]
    c1 = Margins._convert_numeric_to_float64((k = big,)).k
    c2 = Margins._convert_numeric_to_float64((k = big,)).k
    @test c1 == Float64.(big) && c2 == Float64.(big)
end
