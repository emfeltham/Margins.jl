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
    @test nrow(DataFrame(profile1)) == 1
    @test nrow(DataFrame(profile2)) == 1

    # general→specific precedence: :all then x override
    atspec = Dict(:all=>:mean, :x=>[-1.0, 0.0, 1.0])
    profile_spec = profile_margins(m, df; type=:predictions, at=atspec)
    @test nrow(DataFrame(profile_spec)) == 3
    df_result = DataFrame(profile_spec)
    @test any(contains.(string.(names(df_result)), "x"))

    # numlist parsing e.g., "-2(2)2" becomes [-2,0,2]
    profile_num = profile_margins(m, df; type=:predictions, at=Dict(:x=>"-2(2)2"))
    @test nrow(DataFrame(profile_num)) == 3

    # multiple at blocks concatenation  
    profile_multi = profile_margins(m, df; type=:predictions, at=[Dict(:x=>[-1.0]), Dict(:x=>[1.0])])
    @test nrow(DataFrame(profile_multi)) == 2

    # multiple profiles without averaging
    profile_multiple = profile_margins(m, df; type=:predictions, at=Dict(:x=>[-2.0,0.0,2.0]))
    @test nrow(DataFrame(profile_multiple)) == 3
end

