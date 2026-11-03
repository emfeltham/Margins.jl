# test_typical_values.jl
# Weighted typical values (3.0 plan, step 9): `Margins.typical_value`, the weighted median and
# mean, weighted shares and mixtures, and the data-level `weights=` of the grid builders.

using Test, Random, Statistics
using DataFrames, CategoricalArrays, GLM
using Margins
using Margins: typical_value, CategoricalMixture

# A mixture as a Dict(level string => share), independent of the mixture's level order
_shares(m::CategoricalMixture) = Dict(string(l) => w for (l, w) in zip(m.levels, m.weights))
_approx(a::Dict, b::Dict) = keys(a) == keys(b) && all(isapprox(a[k], b[k]; rtol=1e-12) for k in keys(a))
# Grids equal cell by cell (a mixture by its levels and shares, in order: bitwise the same grid)
_cell_eq(a::CategoricalMixture, b::CategoricalMixture) = a.levels == b.levels && a.weights == b.weights
_cell_eq(a, b) = isequal(a, b)
_grid_eq(a, b) = names(a) == names(b) && size(a) == size(b) &&
                 all(_cell_eq(a[i, j], b[i, j]) && typeof(a[!, j]) == typeof(b[!, j]) for i in 1:nrow(a), j in 1:ncol(a))

@testset "typical values (weighted)" begin
    @testset "weighted median: definition" begin
        # Hand-computed: sorted values with cumulative weights C, total W
        @test typical_value([1.0, 2, 3, 4], median; weights=[1.0, 3, 1, 1]) == 2.0   # C = 1,4 ≥ 3 at x=2
        @test typical_value([1.0, 2, 3, 4], median; weights=[1.0, 2, 2, 1]) == 2.5   # C₂ = 3 = W/2: midpoint
        @test typical_value([1.0, 2, 3], median; weights=[1.0, 1, 2]) == 2.5         # = median([1,2,3,3])
        @test typical_value([10.0, 1, 5], median; weights=[0.2, 0.5, 0.3]) == 3.0    # unsorted; tie at 0.5
        @test typical_value([10.0, 1, 5], median; weights=[0.2, 0.6, 0.2]) == 1.0
        @test typical_value([1, 5, 9], median; weights=[1.0, 1, 5]) === 9.0          # Int column → Float64

        # Property checks over 2000 random cases (ties included), counted so each property is one test
        rng = MersenneTwister(20261029)
        bad = Dict(:equal => 0, :equal_general => 0, :repeated => 0, :scale => 0, :zero => 0)
        for _ in 1:2000
            n = rand(rng, 1:12)
            x = round.(randn(rng, n); digits=rand(rng, 0:2))
            # Equal weights: exactly `median` (also the general path, without the equal-weights shortcut)
            for c in (1.0, 0.1, 3.7, 1e-6, 2.0^20)
                typical_value(x, median; weights=fill(c, n)) === median(x) || (bad[:equal] += 1)
                Margins._weighted_median(x, fill(c, n)) === median(x) || (bad[:equal_general] += 1)
            end
            # Integer weights: the median of the sample with each row repeated wᵢ times
            wi = Float64.(rand(rng, 1:4, n))
            m = typical_value(x, median; weights=wi)
            m == median(reduce(vcat, [fill(x[i], Int(wi[i])) for i in 1:n])) || (bad[:repeated] += 1)
            # Invariant to the weights' scale
            for s in (0.1, 0.37, 1e-3, 7.13)
                typical_value(x, median; weights=wi .* s) === m || (bad[:scale] += 1)
            end
            # Zero weights: as if the row were dropped
            w0 = copy(wi); w0[rand(rng, 1:n)] = 0.0
            if any(>(0), w0)
                keep = w0 .> 0
                typical_value(x, median; weights=w0) === typical_value(x[keep], median; weights=w0[keep]) ||
                    (bad[:zero] += 1)
            end
        end
        for (k, v) in bad
            @test (k, v) == (k, 0)
        end
        # The exact tie survives rescaling (0.1·[1,2,2,1]: the cumulative weight 0.30000000000000004 is W/2)
        @test typical_value([1.0, 2, 3, 4], median; weights=0.1 .* [1.0, 2, 2, 1]) == 2.5
    end

    @testset "weighted mean, shares and mixtures" begin
        x = [1.0, 2.0, 4.0]
        @test typical_value(x, mean; weights=[1.0, 1.0, 2.0]) == 11 / 4
        @test typical_value(x; weights=[1.0, 1.0, 2.0]) == 11 / 4          # mean is the default
        rng = MersenneTwister(7)
        @test all(1:200) do _
            n = rand(rng, 1:20); v = randn(rng, n)
            all(c -> typical_value(v, mean; weights=fill(c, n)) === mean(v), (1.0, 0.1, 3.7))
        end
        # Bool: weighted share of true
        @test typical_value([true, false, true], mean; weights=[1.0, 2.0, 3.0]) == 4 / 6
        @test typical_value([true, false, true]; weights=[2.0, 2.0, 2.0]) === mean([true, false, true])
        # Categorical and string mixtures: weighted shares
        g = categorical(["a", "b", "a", "c"])
        m = typical_value(g; weights=[1.0, 2.0, 3.0, 4.0])
        @test m isa CategoricalMixture
        @test _approx(_shares(m), Dict("a" => 0.4, "b" => 0.2, "c" => 0.4))
        ms = typical_value(["p", "q", "q"]; weights=[3.0, 0.5, 0.5])
        @test _approx(_shares(ms), Dict("p" => 0.75, "q" => 0.25))
        # Equal weights: the unweighted mixture, levels in the same order
        mu = typical_value(g); me = typical_value(g; weights=fill(0.3, 4))
        @test mu.levels == me.levels && mu.weights == me.weights
        # Zero weights drop rows: a level seen only in zero-weight rows is not in the mixture
        mz = typical_value(g; weights=[1.0, 0.0, 1.0, 2.0])
        @test _shares(mz) == Dict("a" => 0.5, "c" => 0.5)
        @test typical_value([true, false], mean; weights=[1.0, 0.0]) == 1.0
    end

    @testset "invalid weights and custom typical" begin
        x = [1.0, 2.0, 3.0]
        @test_throws ArgumentError typical_value(x; weights=[1.0, -1.0, 1.0])
        @test_throws ArgumentError typical_value(x; weights=[1.0, NaN, 1.0])
        @test_throws ArgumentError typical_value(x; weights=[1.0, Inf, 1.0])
        @test_throws ArgumentError typical_value(x; weights=[0.0, 0.0, 0.0])
        @test_throws ArgumentError typical_value(x; weights=[1.0, missing, 1.0])
        @test_throws ArgumentError typical_value(x; weights=[1.0, 1.0])
        # A custom typical function cannot be weighted: an error, not silently unweighted
        q90 = v -> quantile(v, 0.9)
        @test typical_value(x, q90) == quantile(x, 0.9)
        err = try typical_value(x, q90; weights=[1.0, 2.0, 3.0]) catch e e end
        @test err isa ArgumentError && occursin("cannot be weighted", err.msg)
        @test_throws ArgumentError typical_value(x, q90; weights=ones(3))        # also for equal weights
        @test_throws ArgumentError typical_value(["a"], q90; weights=[1.0])        # also where unused
        d = DataFrame(x=x, w=[1.0, 2.0, 3.0])
        @test_throws ArgumentError means_grid(d; typical=q90, weights=:w)
    end

    # Data for the grid builders: w is not equal across rows, w0 has zeros
    n = 60
    d = DataFrame(
        x = [mod(7i, 13) + 0.25i for i in 1:n],
        z = [Float64(mod(5i, 11)) for i in 1:n],
        g = categorical(["b", "a", "c"][1 .+ mod.(1:n, 3)]; levels=["c", "b", "a"]),
        s = ["q", "p"][1 .+ mod.(div.(1:n, 4), 2)],
        t = isodd.(div.(1:n, 3)),
        w = [0.5 + mod(3i, 7) / 4 for i in 1:n],
    )
    d.w0 = [mod(i, 5) == 0 ? 0.0 : d.w[i] for i in 1:n]
    d.y = 1.0 .+ 0.3 .* d.x .- 0.2 .* d.z .+ 0.4 .* (d.g .== "a") .+ 0.1 .* d.t .+ sin.(1:n)
    wsum = sum(d.w)
    wmean(v) = sum(d.w .* v) / wsum

    @testset "means_grid(…; weights=)" begin
        g0 = means_grid(d)
        @test _grid_eq(means_grid(d; weights=nothing), g0)
        @test _grid_eq(means_grid(d; weights=:none), g0)
        gw = means_grid(d; weights=:w)
        @test gw.x[1] ≈ wmean(d.x) rtol=1e-14
        @test gw.z[1] ≈ wmean(d.z) rtol=1e-14
        @test gw.t[1] ≈ wmean(d.t) rtol=1e-14
        @test _approx(_shares(gw.g[1]), Dict(l => sum(d.w[d.g .== l]) / wsum for l in ["a", "b", "c"]))
        @test _approx(_shares(gw.s[1]), Dict(l => sum(d.w[d.s .== l]) / wsum for l in ["p", "q"]))
        # A vector is the same as the column
        gv = means_grid(d; weights=d.w)
        @test gv.x == gw.x && gv.z == gw.z && gv.t == gw.t
        # Weighted medians
        gm = means_grid(d; typical=median, weights=:w)
        @test gm.x[1] == Margins._weighted_median(d.x, d.w)
        # Zero weights drop rows
        keep = d.w0 .> 0
        gz = means_grid(d; weights=:w0)
        gk = means_grid(d[keep, :]; weights=:w0)
        @test gz.x == gk.x && gz.z == gk.z
        # Invalid
        @test_throws ArgumentError means_grid(d; weights=:nope)
        @test_throws ArgumentError means_grid(d; weights=ones(3))
        @test_throws ArgumentError means_grid(d; weights=-d.w)
        err = try means_grid(d; weights=:model) catch e e end
        @test err isa ArgumentError && occursin("complete_reference_grid", err.msg)
    end

    @testset "balanced_grid, quantile_grid, hierarchical_grid (weights=)" begin
        b0 = balanced_grid(d; g=:all)
        @test _grid_eq(balanced_grid(d; g=:all, weights=nothing), b0)
        bw = balanced_grid(d; g=:all, weights=:w)
        @test _shares(bw.g[1]) == _shares(b0.g[1])               # the balanced variable keeps equal shares
        @test bw.x[1] ≈ wmean(d.x) rtol=1e-14                     # others are weighted
        @test _approx(_shares(bw.s[1]), Dict(l => sum(d.w[d.s .== l]) / wsum for l in ["p", "q"]))

        q0 = quantile_grid(d; x=[0.25, 0.5])
        @test _grid_eq(quantile_grid(d; x=[0.25, 0.5], weights=nothing), q0)
        # Equal weights: the unweighted quantiles, bit for bit
        @test quantile_grid(d; x=[0.25, 0.5], weights=fill(2.5, n)).x == q0.x
        qw = quantile_grid(d; x=[0.25, 0.5], weights=:w)
        @test qw.x == Margins._weighted_quantiles(d.x, d.w, [0.25, 0.5])
        @test qw.z == fill(typical_value(d.z; weights=d.w), 2)

        h0 = hierarchical_grid(d, :g => (:x, :median))
        @test _grid_eq(hierarchical_grid(d, :g => (:x, :median); weights=nothing), h0)
        hw = hierarchical_grid(d, :g => (:x, :median); weights=:w)
        for r in eachrow(hw)
            rows = d.g .== r.g
            @test r.x == Margins._weighted_median(d.x[rows], d.w[rows])   # within each group
        end
        @test all(hw.z .≈ wmean(d.z))                                       # fill-ins weighted, whole data
        hq = hierarchical_grid(d, (:x, :quartiles); weights=fill(1.0, n))
        @test hq.x == hierarchical_grid(d, (:x, :quartiles)).x
        hmix = hierarchical_grid(d, (:g, :mix_proportional); weights=:w, warn_large=false)
        @test _approx(_shares(only(hmix.g)), Dict(l => sum(d.w[d.g .== l]) / wsum for l in ["a", "b", "c"]))
        @test_throws ArgumentError hierarchical_grid(d, :g; weights=:model)
    end

    @testset "complete_reference_grid(…; weights=), weighted MEM" begin
        m = lm(@formula(y ~ x + z + g + t), d; wts=d.w)
        part = cartesian_grid(x=[1.0, 2.0])
        c0 = complete_reference_grid(part, m, d)
        @test _grid_eq(complete_reference_grid(part, m, d; weights=nothing), c0)
        cw = complete_reference_grid(part, m, d; weights=:w)
        @test all(cw.z .≈ wmean(d.z))
        @test all(cw.t .≈ wmean(d.t))
        @test _approx(_shares(cw.g[1]), Dict(l => sum(d.w[d.g .== l]) / wsum for l in ["a", "b", "c"]))
        # :model = the model's estimation weights; an unweighted model gives the unweighted fill
        @test _grid_eq(complete_reference_grid(part, m, d; weights=:model), cw)
        mu = lm(@formula(y ~ x + z + g + t), d)
        @test _grid_eq(complete_reference_grid(part, mu, d; weights=:model), c0)
        @test_throws ArgumentError complete_reference_grid(part, m, d; typical=v -> quantile(v, 0.9), weights=:w)

        # profile_margins evaluates the grid it is given: weighted MEM = MEM at the hand-weighted point
        mem_w = DataFrame(profile_margins(m, d, means_grid(d; weights=:w); vars=[:x]))
        hand = DataFrame(x=[wmean(d.x)], z=[wmean(d.z)], t=[wmean(d.t)],
                         g=[CategoricalMixture(["c", "b", "a"], [sum(d.w[d.g .== l]) / wsum for l in ["c", "b", "a"]])])
        mem_h = DataFrame(profile_margins(m, d, hand; vars=[:x]))
        @test mem_w.estimate ≈ mem_h.estimate rtol=1e-12
        @test mem_w.se ≈ mem_h.se rtol=1e-12
        # ... and a partial grid completed with weights equals the explicit weighted grid
        pp_w = DataFrame(profile_margins(m, d, cw; type=:predictions))
        pp_h = DataFrame(profile_margins(m, d, DataFrame(x=[1.0, 2.0], z=fill(wmean(d.z), 2), t=fill(wmean(d.t), 2),
                                                         g=fill(hand.g[1], 2)); type=:predictions))
        @test pp_w.estimate ≈ pp_h.estimate rtol=1e-12
        # profile_margins' own completion stays unweighted
        pp_0 = DataFrame(profile_margins(m, d, part; type=:predictions))
        @test pp_0.estimate ≈ DataFrame(profile_margins(m, d, c0; type=:predictions)).estimate rtol=1e-14
        @test_throws MethodError profile_margins(m, d, part; weights=:w)
    end

    @testset "second differences at=:mean/:median use typical_value" begin
        m = lm(@formula(y ~ x * z + g), d)
        for at in (:mean, :median)
            sd = second_differences(m, d, :x, :z; at, weights=:w)
            @test only(sd.eval_point) == Float64(typical_value(d.z, at === :mean ? mean : median; weights=d.w))
            sdg = second_differences(m, d, :x, :z; at, weights=:w, groups=:s)
            for r in eachrow(sdg)
                rows = d.s .== r.over_s
                @test r.eval_point == Float64(typical_value(d.z[rows], at === :mean ? mean : median; weights=d.w[rows]))
            end
        end
    end
end
