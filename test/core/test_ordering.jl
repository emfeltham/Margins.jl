# test_ordering.jl
# Deterministic row and column order (3.0 plan, step 9): grid builders, `groups=` strata and
# `scenarios=` combinations. Convention: the first variable varies slowest; categorical levels in
# level order; Bool/strings/numbers ascending; columns in the user's (keyword) or the data's order.

using Test
using DataFrames, CategoricalArrays, GLM
using Margins

@testset "deterministic ordering" begin
    n = 48
    d = DataFrame(
        y = sin.(1:n) .+ 0.1 .* (1:n),
        x = [mod(7i, 13) / 3 for i in 1:n],
        # levels deliberately not alphabetical, and the data starts with "a" (the last level)
        g = categorical(["a", "c", "b", "a"][1 .+ mod.(0:n-1, 4)]; levels=["c", "b", "a"]),
        s = ["q", "p", "r"][1 .+ mod.(0:n-1, 3)],
        t = [true, false][1 .+ mod.(div.(0:n-1, 4), 2)],
        z = [mod(5i, 11) / 2 for i in 1:n],
    )
    m = lm(@formula(y ~ x + g + s + t + z), d)

    @testset "cartesian_grid: keyword order, first variable slowest" begin
        cg = cartesian_grid(z=[2, 1], x=[5, 6, 7])
        @test names(cg) == ["z", "x"]
        @test cg.z == [2, 2, 2, 1, 1, 1]
        @test cg.x == [5, 6, 7, 5, 6, 7]
        cg3 = cartesian_grid(b=[true, false], a=1:2, c="x")
        @test names(cg3) == ["b", "a", "c"]
        @test collect(zip(cg3.b, cg3.a)) == [(true, 1), (true, 2), (false, 1), (false, 2)]
        @test cg3.c == fill("x", 4)
        # A Dict has no order: alphabetical
        cd = cartesian_grid(Dict(:z => [1, 2], :x => [5, 6]))
        @test names(cd) == ["x", "z"]
        @test collect(zip(cd.x, cd.z)) == [(5, 1), (5, 2), (6, 1), (6, 2)]
    end

    @testset "builders over all columns follow the data's column order" begin
        @test names(means_grid(d)) == names(d)
        @test names(balanced_grid(d; g=:all)) == names(d)
        @test names(quantile_grid(d; z=[0.5], x=[0.25, 0.75])) == names(d)
        qg = quantile_grid(d; z=[0.25, 0.75], x=[0.25, 0.75])
        @test qg.z == repeat([quantile(d.z, 0.25), quantile(d.z, 0.75)]; inner=2)   # first keyword slowest
        @test qg.x == repeat([quantile(d.x, 0.25), quantile(d.x, 0.75)]; outer=2)
        @test names(hierarchical_grid(d, :g)) == names(d)
        # complete_reference_grid: the grid's columns, then the fill-ins in the data's order
        cr = complete_reference_grid(cartesian_grid(z=[1.0], x=[0.0]), m, d)
        @test names(cr) == ["z", "x", "g", "s", "t"]
    end

    @testset "hierarchical_grid rows: level order, ascending, first slowest" begin
        @test string.(hierarchical_grid(d, :g).g) == ["c", "b", "a"]
        @test hierarchical_grid(d, :s).s == ["p", "q", "r"]
        h = hierarchical_grid(d, [:t, :g]; warn_large=false)
        @test collect(zip(h.t, string.(h.g))) ==
              [(false, "c"), (false, "b"), (false, "a"), (true, "c"), (true, "b"), (true, "a")]
        hn = hierarchical_grid(d, :g => :s; warn_large=false)
        @test string.(hn.g) == repeat(["c", "b", "a"]; inner=3)
        @test hn.s == repeat(["p", "q", "r"]; outer=3)
    end

    @testset "groups= strata order" begin
        r = DataFrame(population_margins(m, d; vars=[:x], groups=:g))
        @test string.(r.over_g) == ["c", "b", "a"]                       # level order, not first appearance
        @test DataFrame(population_margins(m, d; vars=[:x], groups=:s)).over_s == ["p", "q", "r"]
        @test DataFrame(population_margins(m, d; vars=[:x], groups=:t)).over_t == [false, true]
        rc = DataFrame(population_margins(m, d; vars=[:x], groups=[:t, :g]))
        @test collect(zip(rc.over_t, string.(rc.over_g))) ==
              [(false, "c"), (false, "b"), (false, "a"), (true, "c"), (true, "b"), (true, "a")]
        rn = DataFrame(population_margins(m, d; vars=[:x], groups=:g => :s))
        @test string.(rn.over_g) == repeat(["c", "b", "a"]; inner=3)
        @test rn.over_s == repeat(["p", "q", "r"]; outer=3)
        # Bins are in bin order
        rb = DataFrame(population_margins(m, d; vars=[:x], groups=(:z, 3)))
        @test rb.over_z == ["T1", "T2", "T3"]
        # Values are unchanged: each stratum's estimate is the one computed on that stratum alone
        for row in eachrow(r)
            sub = d[d.g .== row.over_g, :]
            @test row.estimate ≈ DataFrame(population_margins(m, sub; vars=[:x])).estimate[1] rtol=1e-12
        end
    end

    @testset "scenarios= order: first variable slowest, values as given" begin
        r = DataFrame(population_margins(m, d; type=:predictions, scenarios=(z=[2.0, 1.0], t=[true, false])))
        @test collect(zip(r.at_z, r.at_t)) == [(2.0, true), (2.0, false), (1.0, true), (1.0, false)]
        # With groups: scenarios slowest, then strata
        rg = DataFrame(population_margins(m, d; type=:predictions, scenarios=(z=[2.0, 1.0],), groups=:g))
        @test collect(zip(rg.at_z, string.(rg.over_g))) ==
              [(2.0, "c"), (2.0, "b"), (2.0, "a"), (1.0, "c"), (1.0, "b"), (1.0, "a")]
    end

    @testset "profile_margins keeps the grid's order" begin
        grid = cartesian_grid(t=[true, false], x=[2.0, 1.0])
        r = DataFrame(profile_margins(m, d, grid; type=:predictions))
        @test collect(zip(r.t, r.x)) == [(true, 2.0), (true, 1.0), (false, 2.0), (false, 1.0)]
        @test findfirst(==("t"), names(r)) < findfirst(==("x"), names(r))
    end

    @testset "second differences: strata in level order" begin
        mi = lm(@formula(y ~ x * z + g), d)
        sd = second_differences(mi, d, :x, :z; groups=:g, at=:mean)
        @test string.(sd.over_g) == ["c", "b", "a"]
        sdp = second_differences(mi, d, :x, :z; at=:mean, profile=(g=["a", "c"],))
        @test string.(sdp.g) == ["a", "c"]
    end
end
