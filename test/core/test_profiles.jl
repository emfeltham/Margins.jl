using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins
using Statistics

@testset "Profile margins reference grid semantics" begin
    Random.seed!(06515)
    n = 200
    df = DataFrame(
        y = randn(n),
        x = Float64.(randn(n)),  # Ensure Float64 for compatibility
        z = Float64.(randn(n)),  # Ensure Float64 for compatibility
        g = categorical(rand(["A","B","C"], n))
    )
    m = lm(@formula(y ~ x + z + g), df)

    # Profile predictions at means
    profile1 = profile_margins(m, df, means_grid(df); type=:predictions)  # At sample means
    profile2 = profile_margins(m, df, means_grid(df); type=:predictions)  # Equivalent explicit call
    @test nrow(DataFrame(profile1)) == 1
    @test nrow(DataFrame(profile2)) == 1

    # Using cartesian grid to specify x values, others use typical values
    profile_spec = profile_margins(m, df, cartesian_grid(df; x=[-1.0, 0.0, 1.0]); type=:predictions)
    @test nrow(DataFrame(profile_spec)) == 3
    df_result = DataFrame(profile_spec)
    @test any(contains.(string.(names(df_result)), "x"))

    # explicit values (equivalent to numlist "-2(2)2" which becomes [-2,0,2])
    profile_num = profile_margins(m, df, cartesian_grid(df; x=[-2.0, 0.0, 2.0]); type=:predictions)
    @test nrow(DataFrame(profile_num)) == 3

    # explicit profiles using DataFrame with proper categorical values
    profile_multi = profile_margins(m, df, DataFrame(x=[-1.0, 1.0], z=[mean(df.z), mean(df.z)], g=[df.g[1], df.g[1]]); type=:predictions)
    @test nrow(DataFrame(profile_multi)) == 2

    # multiple profiles using cartesian grid
    profile_multiple = profile_margins(m, df, cartesian_grid(df; x=[-2.0, 0.0, 2.0]); type=:predictions)
    @test nrow(DataFrame(profile_multiple)) == 3
end

