using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins

@testset "Profiles (at) semantics" begin
    Random.seed!(123)
    n = 200
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        z = randn(n),
        g = categorical(rand(["A","B","C"], n))
    )
    m = lm(@formula(y ~ x + z + g), df)

    # at=:means vs at=Dict(:all=>:mean)
    apm1 = apm(m, df)  # uses at=:means
    apm2 = apr(m, df; at=Dict(:all=>:mean))
    @test nrow(apm1.table) == 1
    @test nrow(apm2.table) == 1

    # general→specific precedence: :all then x override
    atspec = Dict(:all=>:mean, :x=>[-1.0, 0.0, 1.0])
    apr_spec = apr(m, df; at=atspec)
    @test nrow(apr_spec.table) == 3
    @test haskey(apr_spec.table, Symbol("at_", :x))

    # numlist parsing e.g., "-2(2)2" becomes [-2,0,2]
    apr_num = apr(m, df; at=Dict(:x=>"-2(2)2"))
    @test nrow(apr_num.table) == 3

    # multiple at blocks concatenation
    apr_multi = apr(m, df; at=[Dict(:x=>[-1.0]), Dict(:x=>[1.0])])
    @test nrow(apr_multi.table) == 2

    # average_profiles collapses profiles to a single summary
    apr_avg = apr(m, df; at=Dict(:x=>[-2.0,0.0,2.0]), average_profiles=true)
    @test nrow(apr_avg.table) == 1
end

