using Test
using DataFrames, CategoricalArrays, GLM, StatsModels, Tables
using Margins
using Random
using Statistics: mean, quantile
using LinearAlgebra: dot

# Continuous and categorical effects under scenarios, with weights and within groups
# (categorical and binned), are checked against closed forms in test_feature_matrix_oracle.jl.

@testset "population_margins: multi-valued scenarios form Cartesian contexts" begin
    # Deterministic dataset so every group is populated
    g = categorical(repeat(["A", "B"], inner=8))
    segment = categorical(repeat(["S1", "S2"], inner=8))
    x = repeat([-1.0, -0.5, 0.5, 1.0], inner=4)
    z = repeat([-0.8, -0.2, 0.2, 0.8], inner=4)
    y = 0.3 .+ 1.5 .* x .- 0.9 .* z .+ 0.4 .* (g .== "B")
    df = DataFrame(y=y, x=x, z=z, g=g, segment=segment)
    m = lm(@formula(y ~ x + z + g), df)   # 4 distinct rows: rank deficient (z dropped), which these shape checks do not depend on

    scenarios = (x=[-1.0, 1.0], z=[-0.5, 0.5])
    expected_pairs = Set([(xv, zv) for xv in scenarios.x for zv in scenarios.z])

    for backend in (:fd, :ad)
        res = @test_logs (:warn, r"rank deficient") population_margins(m, df; type=:effects, vars=[:g], scenarios=scenarios,
                                 scale=:link, backend=backend)
        dfres = DataFrame(res)

        @test size(dfres, 1) == length(expected_pairs)
        @test eltype(dfres.at_x) <: Real
        @test eltype(dfres.at_z) <: Real

        observed = Set([(dfres.at_x[i], dfres.at_z[i]) for i in 1:size(dfres, 1)])
        @test observed == expected_pairs
    end

    # With grouping: one row per (scenario, group) combination
    for backend in (:fd, :ad)
        res = @test_logs (:warn, r"rank deficient") population_margins(m, df; type=:effects, vars=[:g], scenarios=scenarios,
                                 groups=:segment, scale=:link, backend=backend)
        dfres = DataFrame(res)

        unique_segments = unique(df.segment)
        expected = Set([(xv, zv, seg) for xv in scenarios.x for zv in scenarios.z for seg in unique_segments])

        @test size(dfres, 1) == length(expected)
        @test eltype(dfres.at_x) <: Real
        @test eltype(dfres.at_z) <: Real
        observed = Set([(dfres.at_x[i], dfres.at_z[i], dfres.over_segment[i]) for i in 1:size(dfres, 1)])
        @test observed == expected
    end
end

@testset "population_margins: scenario overrides preserve column storage" begin
    g = categorical(repeat(["A", "B"], inner=8))
    flag = repeat([false, true], inner=8)
    x = repeat([-1.0, -0.5, 0.5, 1.0], inner=4)
    z = repeat([-0.8, -0.2, 0.2, 0.8], inner=4)
    y = 0.2 .+ 1.2 .* x .- 0.5 .* z .+ 0.3 .* (g .== "B") .+ 0.1 .* (flag .== true)
    df = DataFrame(y=y, x=x, z=z, g=g, flag=flag)
    m = lm(@formula(y ~ x + z + g + flag), df)

    # Warm run to ensure overall API still works under scenario overrides
    cat_res = @test_logs (:warn, r"rank deficient") population_margins(m, df; type=:effects, vars=[:x], scenarios=(g="A",), scale=:link)
    df_cat = DataFrame(cat_res)
    @test all(df_cat.at_g .== "A")

    bool_res = @test_logs (:warn, r"rank deficient") population_margins(m, df; type=:effects, vars=[:g], scenarios=(flag=false,), scale=:link)
    df_bool = DataFrame(bool_res)
    @test all(df_bool.at_flag .== false)

    cn = coefnames(m)
    idxB = findfirst(s -> occursin("g: B", String(s)) || occursin("g:B", String(s)), cn)
    @test !isnothing(idxB)
    coeffB = coef(m)[idxB]
    @test length(df_bool.estimate) == 1
    @test df_bool.estimate[1] ≈ coeffB atol=1e-6

    # Directly inspect override application: categorical should remain pooled, boolean stays Bool
    data_nt = Tables.columntable(df)
    engine = Margins.build_engine(Margins.PopulationUsage, m, data_nt, [:x, :g, :flag], GLM.vcov, :fd)

    context_cat, _, _ = Margins._create_context_data_with_weights(engine.data_nt, Dict(:g => "A"), Dict(), nothing)
    @test eltype(context_cat.g) <: CategoricalValue
    @test all(x -> x == context_cat.g[1], context_cat.g)

    context_bool, _, _ = Margins._create_context_data_with_weights(engine.data_nt, Dict(:flag => false), Dict(), nothing)
    @test eltype(context_bool.flag) <: Bool
    @test all(x -> x == context_bool.flag[1], context_bool.flag)
end

@testset "population_margins: Pair syntax for scenarios" begin
    # Test that Pair syntax (:var => value) works identically to NamedTuple syntax (var=value)
    n = 100
    x = randn(n)
    z = randn(n)
    β0, β1, β2 = 0.5, 2.0, -1.0
    y = β0 .+ β1 .* x .+ β2 .* z .+ 0.01 .* randn(n)
    df = DataFrame(y=y, x=x, z=z)
    m = lm(@formula(y ~ x + z), df)

    # Test 1: Single scenario - both syntaxes should produce identical results
    res_nt = population_margins(m, df; type=:effects, vars=[:x], scenarios=(z=0.7,), scale=:link)
    res_pair = population_margins(m, df; type=:effects, vars=[:x], scenarios=(:z => 0.7,), scale=:link)

    df_nt = DataFrame(res_nt)
    df_pair = DataFrame(res_pair)

    @test length(df_nt.estimate) == length(df_pair.estimate)
    @test df_nt.estimate ≈ df_pair.estimate
    @test df_nt.se ≈ df_pair.se

    # Check that scenario columns exist and match (if they're created)
    if "at_z" in names(df_nt)
        @test df_nt.at_z == df_pair.at_z
    end

    # Test 2: Multi-valued scenarios with Cartesian expansion (using effects, not predictions)
    # Create a categorical variable to compute effects on
    g = categorical(rand(["A","B"], n))
    y_cat = β0 .+ β1 .* x .+ β2 .* z .+ 0.5 .* (g .== "B") .+ 0.01 .* randn(n)
    df_cat = DataFrame(y=y_cat, x=x, z=z, g=g)
    m_cat = lm(@formula(y ~ x + z + g), df_cat)

    scenarios_nt = (x=[-1.0, 1.0], z=[-0.5, 0.5])
    scenarios_pair = (:x => [-1.0, 1.0], :z => [-0.5, 0.5])

    res_nt = population_margins(m_cat, df_cat; type=:effects, vars=[:g], scenarios=scenarios_nt, scale=:link)
    res_pair = population_margins(m_cat, df_cat; type=:effects, vars=[:g], scenarios=scenarios_pair, scale=:link)

    df_nt = DataFrame(res_nt)
    df_pair = DataFrame(res_pair)

    # Both should produce 4 combinations (2 x values × 2 z values) × 1 categorical contrast
    @test size(df_nt, 1) == 4
    @test size(df_pair, 1) == 4
    @test df_nt.estimate ≈ df_pair.estimate
    @test df_nt.se ≈ df_pair.se

    # Verify Cartesian product was computed correctly (scenario columns should exist for effects with scenarios)
    expected_pairs = Set([(xv, zv) for xv in [-1.0, 1.0] for zv in [-0.5, 0.5]])
    observed_nt = Set([(df_nt.at_x[i], df_nt.at_z[i]) for i in 1:size(df_nt, 1)])
    observed_pair = Set([(df_pair.at_x[i], df_pair.at_z[i]) for i in 1:size(df_pair, 1)])

    @test observed_nt == expected_pairs
    @test observed_pair == expected_pairs

    # Test 3: Boolean variable with Pair syntax
    flag = repeat([false, true], inner=div(n, 2))
    y2 = 0.5 .+ 1.5 .* x .+ 0.3 .* (g .== "B") .+ 0.2 .* flag
    df2 = DataFrame(y=y2, x=x, g=g, flag=flag)
    m2 = lm(@formula(y ~ x + g + flag), df2)

    res_nt = population_margins(m2, df2; type=:effects, vars=[:g], scenarios=(flag=true,), scale=:link)
    res_pair = population_margins(m2, df2; type=:effects, vars=[:g], scenarios=(:flag => true,), scale=:link)

    df_nt = DataFrame(res_nt)
    df_pair = DataFrame(res_pair)

    @test df_nt.estimate ≈ df_pair.estimate
    @test df_nt.se ≈ df_pair.se

    # Check scenario columns if they exist
    if "at_flag" in names(df_nt)
        @test all(df_nt.at_flag .== true)
        @test all(df_pair.at_flag .== true)
    end
end

@testset "population_margins: effects within bins of the differentiated variable" begin
    # Binning a variable does not pin it: x still varies inside every bin, so the average
    # of dy/dx over the bin's rows is well defined. Validated against a manual computation
    # over exactly those rows, on a logit so the bins genuinely differ.
    Random.seed!(20260822)
    n = 600
    x = randn(n)
    z = randn(n)
    η_true = -0.3 .+ 1.2 .* x .- 0.7 .* z
    y = rand(n) .< (1 ./ (1 .+ exp.(-η_true)))
    df = DataFrame(y = y, x = x, z = z)
    m = glm(@formula(y ~ x + z), df, Binomial(), LogitLink())

    X = modelmatrix(m)
    β = coef(m)
    ix = findfirst(==("x"), coefnames(m))
    η = X * β

    # Bin membership, mirroring the documented convention: [lo, hi) except the last, closed.
    q = [quantile(df.x, i/4) for i in 0:4]
    bin_rows(i) = i < 4 ? findall(v -> q[i] <= v < q[i+1], df.x) :
                          findall(v -> q[4] <= v <= q[5], df.x)

    @testset "response scale matches a manual within-bin average" begin
        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x], groups=(:x, 4)))
        @test nrow(res) == 4
        @test sum(res.n) == n                     # the bins partition the sample

        for i in 1:4
            rows = bin_rows(i)
            @test res.n[i] == length(rows)
            # AME on the response scale: mean over the bin of dmu/deta * beta_x
            manual = mean(GLM.mueta(LogitLink(), η[r]) * β[ix] for r in rows)
            @test res.estimate[i] ≈ manual atol=1e-10
        end
        # A logit's within-bin AMEs genuinely differ, so this is not a degenerate check
        @test length(unique(round.(res.estimate; digits=8))) > 1
    end

    @testset "link scale equals the coefficient in every bin" begin
        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x], groups=(:x, 4), scale=:link))
        @test sum(res.n) == n
        @test all(isapprox.(res.estimate, β[ix]; atol=1e-10))
    end

    @testset "standard errors match central differences of the estimate" begin
        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x], groups=(:x, 4)))
        Σ = GLM.vcov(m)
        for i in 1:4
            rows = bin_rows(i)
            est(b) = mean(GLM.mueta(LogitLink(), dot(view(X, r, :), b)) * b[ix] for r in rows)
            g = similar(β)
            for j in eachindex(β)
                h = 1e-6 * max(abs(β[j]), 1.0)
                bp = copy(β); bp[j] += h
                bm = copy(β); bm[j] -= h
                g[j] = (est(bp) - est(bm)) / (2h)
            end
            @test res.se[i] ≈ sqrt(dot(g, Σ * g)) rtol=1e-5
        end
    end

    @testset "threshold bins agree with a manual computation" begin
        res = DataFrame(population_margins(m, df; type=:effects, vars=[:x], groups=(:x, [-0.5, 0.5])))
        @test nrow(res) == 3
        @test sum(res.n) == n
        edges = [(minimum(df.x), -0.5), (-0.5, 0.5), (0.5, maximum(df.x))]
        for (i, (lo, hi)) in enumerate(edges)
            rows = i < 3 ? findall(v -> lo <= v < hi, df.x) : findall(v -> lo <= v <= hi, df.x)
            @test res.n[i] == length(rows)
            manual = mean(GLM.mueta(LogitLink(), η[r]) * β[ix] for r in rows)
            @test res.estimate[i] ≈ manual atol=1e-10
        end
    end

    @testset "weighted within-bin averages" begin
        Random.seed!(77)
        dfw = copy(df)
        dfw.w = rand(n) .+ 0.1
        res = DataFrame(population_margins(m, dfw; type=:effects, vars=[:x], groups=(:x, 4), weights=:w))
        @test sum(res.n) == n
        # Quantile bins are weighted quantiles when weights are given (weighted-quantile
        # properties are tested in regression/test_group_binning.jl)
        cuts = Margins._weighted_quantiles(df.x, dfw.w, [0, 0.25, 0.5, 0.75, 1.0])
        for i in 1:4
            rows = i < 4 ? findall(v -> cuts[i] <= v < cuts[i+1], df.x) :
                           findall(v -> cuts[4] <= v <= cuts[5], df.x)
            @test res.n[i] == length(rows)
            num = sum(dfw.w[r] * GLM.mueta(LogitLink(), η[r]) * β[ix] for r in rows)
            manual = num / sum(dfw.w[r] for r in rows)
            @test res.estimate[i] ≈ manual atol=1e-10
        end
    end

    @testset "agrees with the derived bin column spelling" begin
        # Grouping by a pre-cut categorical column was the documented workaround; binning
        # directly must give the same numbers.
        dfb = copy(df)
        dfb.xq = categorical([q[1] <= v < q[2] ? "Q1" :
                              q[2] <= v < q[3] ? "Q2" :
                              q[3] <= v < q[4] ? "Q3" : "Q4" for v in dfb.x])
        m2 = glm(@formula(y ~ x + z), dfb, Binomial(), LogitLink())

        direct = DataFrame(population_margins(m2, dfb; type=:effects, vars=[:x], groups=(:x, 4)))
        derived = DataFrame(population_margins(m2, dfb; type=:effects, vars=[:x], groups=:xq))
        sort!(direct, :over_x); sort!(derived, :over_xq)
        @test direct.n == derived.n
        @test direct.estimate ≈ derived.estimate
        @test direct.se ≈ derived.se
    end
end
