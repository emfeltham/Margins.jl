using Test
using DataFrames, CategoricalArrays, GLM, StatsModels, Tables
using Margins

@testset "population_margins: scenario-aware continuous effects" begin
    # Simple linear model: y = β0 + β1 x + β2 z + ε, Identity link
    n = 200
    x = randn(n)
    z = randn(n)
    β0, β1, β2 = 0.5, 2.0, -1.0
    y = β0 .+ β1 .* x .+ β2 .* z .+ 0.01 .* randn(n)
    df = DataFrame(y=y, x=x, z=z)
    m = lm(@formula(y ~ x + z), df)

    # Compute AME of x at a counterfactual z = 0.7 (should be ≈ β1 on identity/link scale)
    res = population_margins(m, df; type=:effects, vars=[:x], scenarios=(z=0.7,), scale=:link)
    dfres = DataFrame(res)
    @test length(dfres.estimate) == 1
    @test dfres.estimate[1] ≈ coef(m)[2] atol=1e-6
end

@testset "population_margins: categorical effects under scenarios (unweighted)" begin
    n = 100
    g = categorical(rand(["A","B"], n))
    x = randn(n)
    y = 1 .+ 0.5 .* x .+ (g .== "B") .* 0.3 .+ 0.01 .* randn(n)
    df = DataFrame(y=y, x=x, g=g)
    m = lm(@formula(y ~ x + g), df)

    # Under identity/link scale, the baseline contrast B vs A should equal coefficient for g:B
    res = population_margins(m, df; type=:effects, vars=[:g], scenarios=(x=0.0,), scale=:link)
    dfr = DataFrame(res)
    # One non-baseline level (B) ⇒ single row
    @test length(dfr.estimate) == 1
    # Coefficient for g:B is the third coefficient in lm (Intercept, x, g:B)
    @test dfr.estimate[1] ≈ coef(m)[3] atol=1e-6
end

@testset "population_margins: weighted categorical effects in contexts" begin
    n = 100
    g = categorical(rand(["A","B","C"], n))
    z = randn(n)
    y = 1 .+ (g .== "B") .* 0.3 .+ (g .== "C") .* (-0.2) .+ 0.2 .* z .+ 0.01 .* randn(n)
    w = rand(n) .+ 0.1
    df = DataFrame(y=y, g=g, z=z, w=w)
    m = lm(@formula(y ~ g + z), df)

    # Weighted categorical effects without groups: under identity link, contrasts equal coefficients
    res = population_margins(m, df; type=:effects, vars=[:g], weights=:w, scale=:link)
    dfr = DataFrame(res)
    # Expect two non-baseline levels B and C; order may vary; check set membership
    @test length(dfr.estimate) == 2
    # Pull coefficient positions for g levels
    cn = coefnames(m)
    idxB = findfirst(s -> occursin("g: B", String(s)) || occursin("g:B", String(s)), cn)
    idxC = findfirst(s -> occursin("g: C", String(s)) || occursin("g:C", String(s)), cn)
    @test !isnothing(idxB) && !isnothing(idxC)
    coeffB = coef(m)[idxB]
    coeffC = coef(m)[idxC]
    for i in 1:length(dfr.estimate)
        term = dfr.contrast[i]
        if occursin("B vs", term)
            @test dfr.estimate[i] ≈ coeffB atol=1e-6
        elseif occursin("C vs", term)
            @test dfr.estimate[i] ≈ coeffC atol=1e-6
        end
    end
end

@testset "population_margins: multi-valued scenarios form Cartesian contexts" begin
    # Deterministic dataset so every group is populated
    g = categorical(repeat(["A", "B"], inner=8))
    segment = categorical(repeat(["S1", "S2"], inner=8))
    x = repeat([-1.0, -0.5, 0.5, 1.0], inner=4)
    z = repeat([-0.8, -0.2, 0.2, 0.8], inner=4)
    y = 0.3 .+ 1.5 .* x .- 0.9 .* z .+ 0.4 .* (g .== "B")
    df = DataFrame(y=y, x=x, z=z, g=g, segment=segment)
    m = lm(@formula(y ~ x + z + g), df)

    scenarios = (x=[-1.0, 1.0], z=[-0.5, 0.5])
    expected_pairs = Set([(xv, zv) for xv in scenarios.x for zv in scenarios.z])

    for backend in (:fd, :ad)
        res = population_margins(m, df; type=:effects, vars=[:g], scenarios=scenarios,
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
        res = population_margins(m, df; type=:effects, vars=[:g], scenarios=scenarios,
                                 groups=:segment, scale=:link, backend=backend)
        dfres = DataFrame(res)

        unique_segments = unique(df.segment)
        expected = Set([(xv, zv, seg) for xv in scenarios.x for zv in scenarios.z for seg in unique_segments])

        @test size(dfres, 1) == length(expected)
        @test eltype(dfres.at_x) <: Real
        @test eltype(dfres.at_z) <: Real
        observed = Set([(dfres.at_x[i], dfres.at_z[i], dfres.segment[i]) for i in 1:size(dfres, 1)])
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
    cat_res = population_margins(m, df; type=:effects, vars=[:x], scenarios=(g="A",), scale=:link)
    df_cat = DataFrame(cat_res)
    @test all(df_cat.at_g .== "A")

    bool_res = population_margins(m, df; type=:effects, vars=[:g], scenarios=(flag=false,), scale=:link)
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
    engine = Margins.build_engine(Margins.PopulationUsage, Margins.HasDerivatives,
                                  m, data_nt, [:x, :g, :flag], GLM.vcov, :fd)

    context_cat, _, _ = Margins._create_context_data_with_weights(engine.data_nt, Dict(:g => "A"), Dict(), nothing)
    @test eltype(context_cat.g) <: CategoricalValue
    @test all(x -> x == context_cat.g[1], context_cat.g)

    context_bool, _, _ = Margins._create_context_data_with_weights(engine.data_nt, Dict(:flag => false), Dict(), nothing)
    @test eltype(context_bool.flag) <: Bool
    @test all(x -> x == context_bool.flag[1], context_bool.flag)
end

@testset "population_margins: weighted categorical effects with groups" begin
    n = 120
    g = categorical(rand(["A","B","C"], n))
    z = randn(n)
    w = rand(n) .+ 0.1
    y = 1 .+ (g .== "B") .* 0.4 .+ (g .== "C") .* (-0.25) .+ 0.3 .* z .+ 0.01 .* randn(n)
    df = DataFrame(y=y, g=g, z=z, w=w)
    m = lm(@formula(y ~ g + z), df)

    # Group by z quartiles; on identity/link scale, contrasts remain equal to coefficients within tolerance
    res = population_margins(m, df; type=:effects, vars=[:g], groups=(:z, 4), weights=:w, scale=:link)
    dfr = DataFrame(res)

    cn = coefnames(m)
    idxB = findfirst(s -> occursin("g: B", String(s)) || occursin("g:B", String(s)), cn)
    idxC = findfirst(s -> occursin("g: C", String(s)) || occursin("g:C", String(s)), cn)
    @test !isnothing(idxB) && !isnothing(idxC)
    coeffB = coef(m)[idxB]
    coeffC = coef(m)[idxC]

    # Expect at most one row per non-baseline level per group (2 contrasts × 4 groups = 8 rows)
    # Some groups may miss a level by random chance; allow fewer rows
    @test length(dfr.estimate) <= 8
    for i in 1:length(dfr.estimate)
        term = dfr.contrast[i]
        if occursin("B vs", term)
            @test dfr.estimate[i] ≈ coeffB atol=1e-6
        elseif occursin("C vs", term)
            @test dfr.estimate[i] ≈ coeffC atol=1e-6
        end
    end
end
