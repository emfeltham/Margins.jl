# test/regression/test_pvalues_and_version.jl
#
# Regression tests:
#  * p-values were computed as 2(1 - Φ(|z|)), which is exactly 0 once Φ(|z|) rounds to 1
#    (|z| ≳ 8.3). They now use the complementary CDF, 2·ccdf(Normal(), |z|), which stays
#    accurate down to |z| ≈ 38.
#  * `Margins.VERSION` was a hard-coded v"2.0.0" while Project.toml said 2.2.0; it is now
#    read from Project.toml.

using Test
using DataFrames, GLM, Distributions, Statistics
using Margins

@testset "p-values use the complementary CDF" begin
    i = 1:400
    d = DataFrame(x = sin.(i .* 0.37) .* 2, z = cos.(i .* 1.1))
    d.y = 1.0 .+ 0.5 .* d.x .+ 0.3 .* d.z .+ 0.9 .* sin.(i .* 2.9)   # |z| for x is ≈ 9–37
    m = lm(@formula(y ~ x + z), d)

    # (the default profile-effects table has no p-value column; it is skipped)
    pz(r) = (df = DataFrame(r); hasproperty(df, :p_value) ?
             [(abs(e / s), p) for (e, s, p) in zip(df.estimate, df.se, df.p_value)] : [])
    results = [population_margins(m, d; vars=[:x]),
               population_margins(m, d; type=:predictions),
               profile_margins(m, d, cartesian_grid(x=[1.0]); vars=[:x]),
               profile_margins(m, d, cartesian_grid(x=[1.0]); type=:predictions)]
    @test count(r -> !isempty(pz(r)), results) >= 3
    for r in results, (z, p) in pz(r)
        @test p ≈ 2 * ccdf(Normal(), z) rtol=1e-10
    end
    (zx, px) = only(pz(results[1]))
    @test 8.5 < zx < 37        # in the range where 1 - cdf underflows to exactly 0
    @test px > 0
end

@testset "Margins.VERSION matches Project.toml" begin
    @test Margins.VERSION == pkgversion(Margins)
    @test Margins.VERSION >= v"2.2.0"
end
