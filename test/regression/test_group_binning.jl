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
        @test res.income == ["< 25000", "[25000, 50000)", ">= 50000"]
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
