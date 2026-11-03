# Regression tests for: continuous `groups` binning silently returned wrong numbers
#
# Five distinct defects, all silent (no error, no warning), all uncovered by the suite:
#
#   1. Quantile bins used [lo, hi) for bin 1 but (lo, hi] for the rest, so every row sitting
#      exactly on an interior boundary belonged to no bin at all. With tied values this lost
#      a large share of the sample: groups=(:income, 4) on income with a mass at zero kept
#      333 of 500 rows and reported Q1 as 0.0 +/- 0.0.
#   2. Tied quantile boundaries produced empty bins reported as 0.0 +/- 0.0 -- readable as a
#      precisely estimated null rather than "no data".
#   3. The group spec carried a companion `Symbol("$(var)_bin")` key. When the data happened
#      to contain a real column of that name -- which docs/src/stata_migration.md and
#      docs/src/examples.md instructed users to create -- it was matched against that column,
#      every context came back empty, and all estimates were 0.0 +/- 0.0.
#   4. Threshold bin labels contradicted the filtering: a bin labelled "[25000, 50000)"
#      actually selected (25000, 50000]. Rows at the first threshold were dropped entirely
#      and rows at later thresholds landed one bin too low.
#   5. A bare numeric `groups=:x` silently auto-quartiled. On an integer count column this
#      returned quartiles of it instead of its levels, with degenerate bins losing rows.
#      Stata's over() and R marginaleffects' by() both require a discrete grouping variable.
#
# The unifying invariant asserted throughout: **the bins must partition the sample exactly**.

using Test
using Margins, GLM, DataFrames, CategoricalArrays
using Random
using Statistics

@testset "Continuous group binning" begin

    @testset "Quantile bins partition the sample exactly" begin
        Random.seed!(20260822)
        n = 400
        df = DataFrame(y = randn(n), x = randn(n), z = randn(n))
        m = lm(@formula(y ~ x + z), df)

        res = DataFrame(population_margins(m, df; type=:effects, vars=[:z], groups=(:x, 4)))
        @test nrow(res) == 4
        @test sum(res.n) == n                      # no row belongs to zero bins
        @test all(res.n .> 0)

        # Membership matches the half-open convention: [q0,q1), [q1,q2), [q2,q3), [q3,q4]
        q = [quantile(df.x, i/4) for i in 0:4]
        expected = [count(v -> q[i] <= v < q[i+1], df.x) for i in 1:3]
        push!(expected, count(v -> q[4] <= v <= q[5], df.x))
        @test res.n == expected
        @test sum(expected) == n                   # the convention itself partitions
    end

    @testset "Rows sitting exactly on an interior boundary are not lost" begin
        # Every value is one of five repeated numbers, so interior quantile boundaries fall
        # exactly on observed values -- the case the old mixed convention dropped.
        Random.seed!(11)
        vals = Float64.(repeat([1.0, 2.0, 3.0, 4.0, 5.0], inner = 40))
        n = length(vals)
        df = DataFrame(y = randn(n), x = randn(n), w = vals)
        m = lm(@formula(y ~ x + w), df)

        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x], groups=(:w, 5)))
        @test sum(res.n) == n
        @test all(res.n .> 0)
    end

    @testset "Tied quantile boundaries are refused, not silently emptied" begin
        Random.seed!(3)
        n = 500
        # 30% of incomes are exactly zero: the 0th and 1st quartile boundaries coincide.
        income = [rand() < 0.3 ? 0.0 : 20000 + 30000*rand() for _ in 1:n]
        df = DataFrame(y = randn(n), x = randn(n), income = income)
        m = lm(@formula(y ~ x + income), df)

        @debug "Testing tied quantile boundaries" expected_error=Margins.MarginsError
        @test_throws Margins.MarginsError population_margins(
            m, df; type=:effects, vars=[:x], groups=(:income, 4)
        )

        # Fewer bins avoid the tie and must then partition exactly.
        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x], groups=(:income, 2)))
        @test sum(res.n) == n
        @test all(res.n .> 0)
    end

    @testset "A column named <var>_bin does not corrupt the result" begin
        Random.seed!(1)
        n = 400
        base = DataFrame(y = randn(n), x = randn(n), z = randn(n))
        m = lm(@formula(y ~ x + z), base)

        withbin = copy(base)
        withbin.x_bin = cut(withbin.x, 4)   # exactly what the migration docs told users to create

        r_without = DataFrame(population_margins(m, base;    type=:effects, vars=[:z], groups=(:x, 4)))
        r_with    = DataFrame(population_margins(m, withbin; type=:effects, vars=[:z], groups=(:x, 4)))

        @test sum(r_with.n) == n
        @test r_with.n == r_without.n
        @test r_with.estimate ≈ r_without.estimate
        @test r_with.se ≈ r_without.se
        @test !any(iszero, r_with.se)        # the failure mode was 0.0 +/- 0.0 everywhere
    end

    @testset "Threshold bins match their own labels" begin
        Random.seed!(4)
        # Mass sits exactly ON both cutpoints -- the policy-threshold case (tax brackets,
        # poverty lines) where this matters most.
        income = Float64.(repeat([10000, 25000, 40000, 50000, 60000], inner = 40))
        n = length(income)
        df = DataFrame(y = randn(n), x = randn(n), income = income)
        m = lm(@formula(y ~ x + income), df)

        res = DataFrame(population_margins(
            m, df; type=:effects, vars=[:x], groups=(:income, [25000, 50000])
        ))

        @test nrow(res) == 3
        @test sum(res.n) == n
        # "< 25000" holds only 10000; "[25000, 50000)" holds 25000 and 40000;
        # ">= 50000" holds 50000 and 60000 -- each label read literally.
        @test res.n == [40, 80, 80]
        @test res.over_income == ["< 25000", "[25000, 50000)", ">= 50000"]
    end

    @testset "Bare numeric groups is refused" begin
        Random.seed!(2)
        n = 300
        df = DataFrame(y = randn(n), x = randn(n), nkids = rand(0:3, n))
        m = lm(@formula(y ~ x + nkids), df)

        @debug "Testing bare numeric groups" test_input=(groups=:nkids,) expected_error=Margins.MarginsError
        @test_throws Margins.MarginsError population_margins(m, df; type=:effects, vars=[:x], groups=:nkids)
        # Same rejection through the vector and nested spellings
        @test_throws Margins.MarginsError population_margins(m, df; type=:effects, vars=[:x], groups=[:nkids])
        @test_throws Margins.MarginsError population_margins(m, df; type=:predictions, groups=:nkids)

        # The explicit spellings work and partition exactly
        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x], groups=(:nkids, 2)))
        @test sum(res.n) == n

        # A categorical column of the same values is grouped by level, as requested
        df.nkids_cat = categorical(string.(df.nkids))
        m2 = lm(@formula(y ~ x + nkids_cat), df)
        res2 = DataFrame(population_margins(m2, df; type=:effects, vars=[:x], groups=:nkids_cat))
        @test nrow(res2) == 4
        @test sum(res2.n) == n
    end

    @testset "Bare categorical groups still works" begin
        Random.seed!(6)
        n = 200
        df = DataFrame(y = randn(n), x = randn(n), g = categorical(rand(["A", "B", "C"], n)))
        m = lm(@formula(y ~ x + g), df)

        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x], groups=:g))
        @test nrow(res) == 3
        @test sum(res.n) == n
        @test all(isfinite, res.estimate)
    end

    @testset "Empty cross-tab cells are omitted, not reported as zero" begin
        Random.seed!(5)
        n = 200
        region = rand(["A", "B"], n)
        edu = rand(["HS", "BA"], n)
        region[1:20] .= "C"        # region C is only ever observed with edu "HS"
        edu[1:20] .= "HS"
        df = DataFrame(y = randn(n), x = randn(n),
                       region = categorical(region), edu = categorical(edu))
        m = lm(@formula(y ~ x + region + edu), df)

        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x], groups=[:region, :edu]))
        @test nrow(res) == 5                    # 6 combinations, one unobserved
        @test !any(res.n .== 0)                 # the C x BA row is absent, not 0.0 +/- 0.0
        @test sum(res.n) == n
        @test !any(iszero, res.se)

        # Predictions take the same path
        res_p = DataFrame(population_margins(m, df; type=:predictions, groups=[:region, :edu]))
        @test nrow(res_p) == 5
        @test sum(res_p.n) == n
    end
end

@testset "Threshold validation and decimal numlist" begin
    using Random
    rng = MersenneTwister(7)
    n = 400
    df = DataFrame(x = randn(rng, n), k = rand(rng, 1:5, n))
    df.y = df.x .+ randn(rng, n)
    m = lm(@formula(y ~ x + k), df)

    # Unsorted or repeated cutpoints would give overlapping bins; they must be rejected
    @test_throws Margins.MarginsError population_margins(m, df; type=:predictions, groups=(:k, [4, 2]))
    @test_throws Margins.MarginsError population_margins(m, df; type=:predictions, groups=(:k, [2, 2]))
    res = DataFrame(population_margins(m, df; type=:predictions, groups=(:k, [2, 3])))
    @test sum(res.n) == n

    # The numlist gate and parser must agree: decimal and negative steps parse
    for (s, expected) in (("-2(0.5)2", collect(-2.0:0.5:2.0)), ("5(-1)1", [5.0, 4, 3, 2, 1]))
        g = DataFrame(profile_margins(m, df, cartesian_grid(x=s); type=:predictions))
        @test g.x ≈ expected
    end
end

@testset "Scenario/group overlap and duplicate group variables" begin
    rng = MersenneTwister(8)
    n = 300
    df = DataFrame(x = randn(rng, n), z = randn(rng, n),
                   g = categorical(rand(rng, ["a", "b", "c"], n)))
    df.y = df.x .+ randn(rng, n)
    m = lm(@formula(y ~ x + z + g), df)

    # A categorical group that is also a scenario variable made the scenario label unreadable
    @test_throws ArgumentError population_margins(m, df; type=:predictions,
        scenarios=(g=["a", "b"],), groups=:g)
    # Different variables are fine
    res = DataFrame(population_margins(m, df; type=:predictions, scenarios=(z=[0, 1],), groups=:g))
    @test nrow(res) == 6

    # Repeating a grouping variable duplicated or overwrote specs
    @test_throws ArgumentError population_margins(m, df; type=:predictions, groups=[:g, :g])
    @test_throws ArgumentError population_margins(m, df; type=:predictions,
        groups=[(:x, 2), (:x, [0.0])])
end

@testset "Unused scenario variables and missing group values are refused" begin
    rng = MersenneTwister(9)
    n = 200
    df = DataFrame(x = randn(rng, n), z = randn(rng, n), wt = rand(rng, n) .+ 0.5,
                   g = categorical(rand(rng, ["a", "b"], n)))
    df.y = df.x .+ randn(rng, n)
    m = lm(@formula(y ~ x + z + g), df)

    # Typo, non-model column, and the response would all silently return the baseline
    for sc in ((zz=[0, 1],), (wt=[0, 1],), (y=[0, 1],))
        @test_throws ArgumentError population_margins(m, df; type=:predictions, scenarios=sc)
    end
    # A real predictor still works
    @test nrow(DataFrame(population_margins(m, df; type=:predictions, scenarios=(z=[0, 1],)))) == 2

    # Missing in a grouping variable gave an opaque TypeError
    dfm = copy(df)
    dfm.h = allowmissing(string.(dfm.g)); dfm.h[1:5] .= missing
    @test_throws ArgumentError population_margins(m, dfm; type=:predictions, groups=:h)
    dfm.gm = allowmissing(df.g); dfm.gm[1:5] .= missing
    @test_throws ArgumentError population_margins(m, dfm; type=:predictions, groups=:gm)
end

@testset "Nested bins are computed within each outer group" begin
    rng = MersenneTwister(10)
    n = 600
    h = rand(rng, ["u", "v"], n)
    # Different income scales per group, so global and within-group cutpoints differ
    x = [hh == "u" ? randn(rng) : 10 + 3 * randn(rng) for hh in h]
    df = DataFrame(z = randn(rng, n), x = x, w = randn(rng, n), h = categorical(h))
    df.y = df.z .+ randn(rng, n)
    m = lm(@formula(y ~ z + x + w + h), df)

    res = DataFrame(population_margins(m, df; type=:effects, vars=[:z], groups=:h => (:x, 3)))
    @test nrow(res) == 6
    @test sum(res.n) == n
    for lvl in ("u", "v")
        sub = res[res.over_h .== lvl, :]
        nh = count(==(lvl), h)
        @test sum(sub.n) == nh
        # terciles of the group's own x: sizes within one row of nh/3
        @test all(abs.(sub.n .- nh / 3) .<= 1)
        @test all(sub.estimate .≈ coef(m)[2])  # linear model: AME of z is its coefficient
    end

    # Sample-wide cutpoints remain available via the crossed vector form
    res2 = DataFrame(population_margins(m, df; type=:effects, vars=[:z], groups=[:h, (:x, 3)]))
    @test sum(res2.n) == n
    @test res2.n != res.n

    # Parallel inner specs and a failure inside a group name the group
    res3 = DataFrame(population_margins(m, df; type=:effects, vars=[:z],
                                        groups=:h => [(:x, 2), (:w, 2)]))
    @test nrow(res3) == 8
    @test sum(res3.n) == 2n
end

@testset "Tuple scenario values" begin
    rng = MersenneTwister(12)
    n = 300
    df = DataFrame(x = randn(rng, n), z = randn(rng, n),
                   g = categorical(rand(rng, ["a", "b", "c"], n)))
    df.y = df.x .+ randn(rng, n)
    m = lm(@formula(y ~ x + z + g), df)

    # (0, 1) used to raise a MethodError; it means the same as [0, 1]
    r_tuple = DataFrame(population_margins(m, df; type=:predictions, scenarios=(z=(0, 1),)))
    r_vec   = DataFrame(population_margins(m, df; type=:predictions, scenarios=(z=[0, 1],)))
    @test nrow(r_tuple) == 2
    @test r_tuple.estimate ≈ r_vec.estimate

    # Weights that are zero everywhere are refused up front
    @test_throws ArgumentError population_margins(m, df; type=:predictions,
        groups=:g, weights=zeros(n))
end

@testset "Combination limit has one honest message" begin
    rng = MersenneTwister(13)
    n = 100
    df = DataFrame(x = randn(rng, n), z = randn(rng, n))
    df.y = df.x .+ randn(rng, n)
    m = lm(@formula(y ~ x + z), df)
    err = try
        population_margins(m, df; type=:predictions, scenarios=(z=collect(1.0:300.0),)); nothing
    catch e
        e
    end
    @test err isa Margins.MarginsError
    msg = sprint(showerror, err)
    @test occursin("300", msg) && occursin("250", msg) && !occursin("1000", msg)
    # At the limit it still runs
    @test nrow(DataFrame(population_margins(m, df; type=:predictions,
        scenarios=(z=collect(1.0:250.0),)))) == 250
end

@testset "Profile grids reject non-predictor columns" begin
    rng = MersenneTwister(14)
    n = 200
    df = DataFrame(x = randn(rng, n), z = randn(rng, n), wt = rand(rng, n))
    df.y = df.x .+ randn(rng, n)
    m = lm(@formula(y ~ x + z), df)

    # An unused column or the response used to return identical rows for every grid point
    for g in (cartesian_grid(wt=[0, 1]), cartesian_grid(y=[0, 1]), DataFrame(x=[0.0, 1.0], wt=[1.0, 2.0]))
        @test_throws ArgumentError profile_margins(m, df, g; type=:predictions)
    end
    res = DataFrame(profile_margins(m, df, cartesian_grid(x=[0, 1]); type=:predictions))
    @test res.estimate[1] != res.estimate[2]
end

@testset "Quantile bins use the weights" begin
    rng = MersenneTwister(15)
    n = 800
    x = randn(rng, n)
    # Heavy weights on the right tail, so weighted and unweighted quartiles differ clearly
    w = exp.(0.8 .* x)
    df = DataFrame(x = x, z = randn(rng, n), w = w)
    df.y = df.z .+ randn(rng, n)
    m = lm(@formula(y ~ x + z), df)
    bins(; kw...) = DataFrame(population_margins(m, df; type=:effects, vars=[:z], groups=(:x, 4), kw...))

    # Unit weights reproduce the unweighted bins exactly; the weights' scale does not matter
    r0 = bins()
    @test bins(weights=ones(n)).n == r0.n
    @test bins(weights=7.5 .* ones(n)).n == r0.n
    rw = bins(weights=w)
    @test bins(weights=w ./ sum(w)).n == rw.n

    # Each weighted bin carries ~25% of the total weight (to within one observation's share),
    # whereas the unweighted bins do not
    function bin_weight_shares(res)
        cuts = cumsum(res.n)
        order = sortperm(x)
        [sum(w[order][(i == 1 ? 1 : cuts[i-1] + 1):cuts[i]]) / sum(w) for i in eachindex(cuts)]
    end
    @test all(abs.(bin_weight_shares(rw) .- 0.25) .< 2 * maximum(w) / sum(w))
    @test maximum(abs.(bin_weight_shares(r0) .- 0.25)) > 0.05
    @test sum(rw.n) == n

    # Zero-weight rows do not influence the cutpoints
    w0 = copy(w); w0[1:100] .= 0
    keep = 101:n
    # Cutpoints equal those from the positively weighted rows alone
    cuts_full = Margins._weighted_quantiles(x, w0, [0, .25, .5, .75, 1.0])
    cuts_sub  = Margins._weighted_quantiles(x[keep], w0[keep], [0, .25, .5, .75, 1.0])
    @test cuts_full == cuts_sub

    # Nested bins use the within-group weights
    df.h = categorical(rand(rng, ["u", "v"], n))
    m3 = lm(@formula(y ~ x + z + h), df)
    r_nested = DataFrame(population_margins(m3, df; type=:effects, vars=[:z],
                                            groups=:h => (:x, 2), weights=w))
    @test sum(r_nested.n) == n
    for lvl in ("u", "v")
        rows = findall(==(lvl), string.(df.h))
        cut = Margins._weighted_quantiles(x[rows], w[rows], [0, .5, 1.0])[2]
        lo = sum(w[rows][x[rows] .< cut]) / sum(w[rows])
        @test abs(lo - 0.5) < 2 * maximum(w[rows]) / sum(w[rows])
    end
end

@testset "Weights are validated the same way as column or vector" begin
    rng = MersenneTwister(16)
    n = 100
    df = DataFrame(x = randn(rng, n), w = rand(rng, n) .+ 0.5)
    df.y = df.x .+ randn(rng, n)
    m = lm(@formula(y ~ x), df)
    run(w) = population_margins(m, df; type=:predictions, weights=w)

    for (label, f) in (("negative", w -> (w[1] = -1.0; w)),
                       ("NaN",      w -> (w[1] = NaN; w)),
                       ("Inf",      w -> (w[1] = Inf; w)),
                       ("all zero", w -> zero(w)))
        v = f(copy(df.w))
        @test_throws ArgumentError run(v)
        d2 = copy(df); d2.w = f(copy(df.w))
        @test_throws ArgumentError population_margins(m, d2; type=:predictions, weights=:w)
    end
    dm = copy(df); dm.w = allowmissing(df.w); dm.w[1] = missing
    @test_throws ArgumentError population_margins(m, dm; type=:predictions, weights=:w)
    @test_throws ArgumentError run(allowmissing(df.w)[[1; 1:n-1]] |> v -> (v[1] = missing; v))

    # Weighted grouped/scenario results report that they are weighted
    r = population_margins(m, df; type=:predictions, weights=:w, scenarios=(x=[0, 1],))
    @test r.metadata[:weighted] == true
end
