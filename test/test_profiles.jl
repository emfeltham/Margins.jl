using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins

@testset "Profile margins (at) semantics" begin
    Random.seed!(123)
    n = 200
    df = DataFrame(
        y = randn(n),
        x = Float64.(randn(n)),  # Ensure Float64 for compatibility
        z = Float64.(randn(n)),  # Ensure Float64 for compatibility
        g = categorical(rand(["A","B","C"], n))
    )
    m = lm(@formula(y ~ x + z + g), df)

    # Profile predictions at means
    profile1 = profile_margins(m, df; type=:predictions, at=:means)
    profile2 = profile_margins(m, df; type=:predictions, at=Dict(:all=>:mean))
    @test nrow(profile1.table) == 1
    @test nrow(profile2.table) == 1

    # general→specific precedence: :all then x override
    atspec = Dict(:all=>:mean, :x=>[-1.0, 0.0, 1.0])
    profile_spec = profile_margins(m, df; type=:predictions, at=atspec)
    @test nrow(profile_spec.table) == 3
    @test haskey(profile_spec.table, Symbol("at_", :x))

    # numlist parsing e.g., "-2(2)2" becomes [-2,0,2]
    profile_num = profile_margins(m, df; type=:predictions, at=Dict(:x=>"-2(2)2"))
    @test nrow(apr_num.table) == 3

    # multiple at blocks concatenation
    apr_multi = apr(m, df; at=[Dict(:x=>[-1.0]), Dict(:x=>[1.0])])
    @test nrow(apr_multi.table) == 2

    # average_profiles collapses profiles to a single summary
    apr_avg = apr(m, df; at=Dict(:x=>[-2.0,0.0,2.0]), average_profiles=true)
    @test nrow(apr_avg.table) == 1
end

