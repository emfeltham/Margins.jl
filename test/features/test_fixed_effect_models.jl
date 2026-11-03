using Test
using Margins
using FixedEffectModels
using DataFrames
using GLM
using Statistics
using CategoricalArrays

@testset "FixedEffectModels Extension" begin
    # Set up common test data with known state effects
    n = 1000
    state_effects = Dict("CA"=>1.0, "NY"=>-1.0, "TX"=>0.5, "FL"=>-0.5, "IL"=>0.0)
    year_effects = Dict(2020=>0.2, 2021=>0.1, 2022=>0.0, 2023=>-0.1, 2024=>-0.2)

    df = DataFrame(
        y = zeros(n),
        x1 = randn(n),
        x2 = randn(n),
        x3 = randn(n),
        state = repeat(["CA", "NY", "TX", "FL", "IL"], 200),
        year = repeat(2020:2024, 200),
    )
    # True coefficients: β₁=2.0, β₂=-1.5, β₃=0.5
    df.y .= 2.0 .* df.x1 .+ (-1.5) .* df.x2 .+ 0.5 .* df.x3 .+
             [state_effects[s] for s in df.state] .+
             [year_effects[y] for y in df.year] .+
             randn(n) * 0.3

    # Model WITHOUT save=:fe (for effects tests and error testing)
    m = reg(df, @formula(y ~ x1 + x2 + x3 + fe(state) + fe(year)))
    # Model WITH save=:fe (for prediction tests)
    m_fe = reg(df, @formula(y ~ x1 + x2 + x3 + fe(state) + fe(year)); save=:fe)

    # =====================================================================
    # Phase 1: Marginal Effects
    # =====================================================================

    @testset "Basic AME" begin
        result = population_margins(m, df; type=:effects)
        df_r = DataFrame(result)

        @test nrow(df_r) == 3
        @test Set(df_r.variable) == Set(["x1", "x2", "x3"])

        for row in eachrow(df_r)
            if row.variable == "x1"
                @test abs(row.estimate - 2.0) < 0.15
            elseif row.variable == "x2"
                @test abs(row.estimate - (-1.5)) < 0.15
            elseif row.variable == "x3"
                @test abs(row.estimate - 0.5) < 0.15
            end
        end

        @test all(df_r.se .> 0)
        @test all(df_r.se .< 0.1)
    end

    @testset "Specific vars" begin
        result = population_margins(m, df; type=:effects, vars=[:x1])
        df_r = DataFrame(result)
        @test nrow(df_r) == 1
        @test df_r.variable[1] == "x1"
        @test abs(df_r.estimate[1] - 2.0) < 0.15

        result2 = population_margins(m, df; type=:effects, vars=[:x1, :x2])
        df_r2 = DataFrame(result2)
        @test nrow(df_r2) == 2
    end

    @testset "Interaction model" begin
        df_int = copy(df)
        df_int.y .= 2.0 .* df.x1 .+ (-1.5) .* df.x2 .+ 0.8 .* df.x1 .* df.x2 .+ randn(n) * 0.3

        m_int = reg(df_int, @formula(y ~ x1 + x2 + x1 & x2 + fe(state)))
        m_glm = lm(@formula(y ~ x1 + x2 + x1 & x2), df_int)

        result_fe = population_margins(m_int, df_int; type=:effects, vars=[:x1, :x2])
        result_glm = population_margins(m_glm, df_int; type=:effects, vars=[:x1, :x2])

        df_fe = DataFrame(result_fe)
        df_glm = DataFrame(result_glm)

        for var in ["x1", "x2"]
            fe_est = df_fe[df_fe.variable .== var, :estimate][1]
            glm_est = df_glm[df_glm.variable .== var, :estimate][1]
            @test abs(fe_est - glm_est) < 0.05
        end
    end

    @testset "Profile margins (MEM)" begin
        grid = DataFrame(x1=[0.0], x2=[0.0], x3=[0.0])
        result = profile_margins(m, df, grid; type=:effects, vars=[:x1, :x2])
        df_r = DataFrame(result)
        @test nrow(df_r) == 2

        ame_result = population_margins(m, df; type=:effects, vars=[:x1, :x2])
        df_ame = DataFrame(ame_result)

        for var in ["x1", "x2"]
            mem_est = df_r[df_r.variable .== var, :estimate][1]
            ame_est = df_ame[df_ame.variable .== var, :estimate][1]
            @test abs(mem_est - ame_est) < 1e-10
        end
    end

    @testset "Scenarios (counterfactual effects)" begin
        df_int = copy(df)
        df_int.y .= 2.0 .* df.x1 .+ (-1.5) .* df.x2 .+ 0.8 .* df.x1 .* df.x2 .+ randn(n) * 0.3

        m_int = reg(df_int, @formula(y ~ x1 + x2 + x1 & x2 + fe(state)))
        β = coef(m_int)

        result = population_margins(m_int, df_int; type=:effects, vars=[:x1], scenarios=(x2=[0.0, 1.0],))
        df_r = DataFrame(result)

        @test nrow(df_r) == 2
        @test abs(df_r.estimate[1] - β[1]) < 0.05
        @test abs(df_r.estimate[2] - (β[1] + β[3])) < 0.05
    end

    @testset "FE variables in vars blocked" begin
        @test_throws Margins.MarginsError population_margins(m, df; type=:effects, vars=[:state])
        @test_throws Margins.MarginsError population_margins(m, df; type=:effects, vars=[:x1, :year])
    end

    @testset "IV model supported" begin
        df_iv = copy(df)
        df_iv.z = randn(n)
        # Make x1 correlated with instrument z (first stage relevance)
        df_iv.x1 .= 0.5 .* df_iv.z .+ randn(n) * 0.5
        df_iv.y .= 2.0 .* df_iv.x1 .+ randn(n) * 0.3

        m_iv = reg(df_iv, @formula(y ~ (x1 ~ z) + fe(state)))
        @test !isnan(m_iv.F_kp)  # Confirm it's actually an IV model

        # IV effects should work (uses structural coefficients)
        result = population_margins(m_iv, df_iv; type=:effects)
        df_r = DataFrame(result)
        @test nrow(df_r) == 1
        @test df_r.variable[1] == "x1"
        # IV estimate should be in the ballpark of true value (2.0)
        # but IV is noisier than OLS, so use wider tolerance
        @test abs(df_r.estimate[1] - 2.0) < 1.0
        @test df_r.se[1] > 0

        # IV predictions should also work if save=:fe
        m_iv_fe = reg(df_iv, @formula(y ~ (x1 ~ z) + fe(state)); save=:fe)
        result_pred = population_margins(m_iv_fe, df_iv; type=:predictions)
        df_pred = DataFrame(result_pred)
        @test nrow(df_pred) == 1
        @test isfinite(df_pred.estimate[1])
    end

    @testset "Both backends" begin
        for backend in [:ad, :fd]
            result = population_margins(m, df; type=:effects, vars=[:x1], backend=backend)
            df_r = DataFrame(result)
            @test abs(df_r.estimate[1] - 2.0) < 0.15
        end
    end

    # =====================================================================
    # Phase 2: Predictions (requires save=:fe)
    # =====================================================================

    @testset "Predictions blocked without save=:fe" begin
        @test_throws Margins.MarginsError population_margins(m, df; type=:predictions)

        grid = DataFrame(x1=[0.0], x2=[0.0], x3=[0.0])
        @test_throws Margins.MarginsError profile_margins(m, df, grid; type=:predictions)
    end

    @testset "Population predictions (AAP)" begin
        result = population_margins(m_fe, df; type=:predictions)
        df_r = DataFrame(result)

        @test nrow(df_r) == 1

        # Manual verification: mean(Xβ + Σ fe_k)
        β = coef(m_fe)
        fe_total = m_fe.fe.fe_state .+ m_fe.fe.fe_year
        xb = β[1] .* df.x1 .+ β[2] .* df.x2 .+ β[3] .* df.x3
        manual = mean(xb .+ fe_total)

        @test abs(df_r.estimate[1] - manual) < 1e-10
        @test df_r.se[1] > 0
    end

    # 3.0 (step 5, decision D3): a grid row's fixed effects are never filled in silently. Before,
    # the sample-average fixed effect was used; now the grid must set every FE variable, or the
    # sum of the fixed effects must be given as `offset=` (or an `offset` grid column).
    @testset "Profile predictions without FE in grid" begin
        grid = DataFrame(x1=[0.0, 1.0], x2=[0.0, 0.0], x3=[0.0, 0.0])
        err = try profile_margins(m_fe, df, grid; type=:predictions); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin(":state", sprint(showerror, err)) && occursin("offset=", sprint(showerror, err))

        β = coef(m_fe)
        avg_fe = mean(m_fe.fe.fe_state .+ m_fe.fe.fe_year)
        df_r = DataFrame(profile_margins(m_fe, df, grid; type=:predictions, offset=avg_fe))
        @test nrow(df_r) == 2
        @test abs(df_r.estimate[1] - (0.0 + avg_fe)) < 1e-10
        @test abs(df_r.estimate[2] - (β[1] + avg_fe)) < 1e-10
        gcol = copy(grid); gcol.offset = [avg_fe, 0.0]
        df_c = DataFrame(profile_margins(m_fe, df, gcol; type=:predictions))
        @test df_c.estimate ≈ [avg_fe, β[1]] atol = 1e-10
    end

    @testset "Profile predictions with FE in grid" begin
        # Build FE lookup tables
        fe_state_lookup = Dict{String, Float64}()
        fe_year_lookup = Dict{Int, Float64}()
        for row in eachrow(m_fe.fe)
            haskey(fe_state_lookup, row.state) || (fe_state_lookup[row.state] = row.fe_state)
            haskey(fe_year_lookup, row.year) || (fe_year_lookup[row.year] = row.fe_year)
        end

        β = coef(m_fe)

        # State only in grid: the year fixed effect is missing (it used to be averaged silently)
        grid = DataFrame(x1=[0.0, 0.0], x2=[0.0, 0.0], x3=[0.0, 0.0], state=["CA", "NY"])
        @test_throws ArgumentError profile_margins(m_fe, df, grid; type=:predictions)

        # Both FEs in grid
        grid2 = DataFrame(x1=[1.0, -1.0], x2=[0.0, 0.0], x3=[0.0, 0.0],
                          state=["CA", "TX"], year=[2020, 2024])
        result2 = profile_margins(m_fe, df, grid2; type=:predictions)
        df_r2 = DataFrame(result2)

        manual1 = β[1]*1.0 + fe_state_lookup["CA"] + fe_year_lookup[2020]
        manual2 = β[1]*(-1.0) + fe_state_lookup["TX"] + fe_year_lookup[2024]
        @test abs(df_r2.estimate[1] - manual1) < 1e-10
        @test abs(df_r2.estimate[2] - manual2) < 1e-10
        # (FixedEffectModels' own `predict` joins on the observed combination of all fixed-effect
        # levels, so it returns `missing` for (TX, 2024), which this data never pairs; Margins
        # looks each fixed effect up separately, as the additive model implies.)
    end

    @testset "Predictions have correct SEs" begin
        # SEs should match the Xβ-only SEs (FE offsets are constants)
        result = population_margins(m_fe, df; type=:predictions)
        df_r = DataFrame(result)

        @test df_r.se[1] > 0
        @test isfinite(df_r.se[1])

        # Profile SEs at non-zero covariate values
        grid = DataFrame(x1=[0.5, 1.0], x2=[0.3, -0.2], x3=[0.1, 0.5], state=["CA", "TX"], year=[2021, 2022])
        result2 = profile_margins(m_fe, df, grid; type=:predictions)
        df_r2 = DataFrame(result2)

        @test all(df_r2.se .> 0)
        @test all(isfinite.(df_r2.se))
    end

    @testset "Effects still work with save=:fe model" begin
        result = population_margins(m_fe, df; type=:effects, vars=[:x1])
        df_r = DataFrame(result)
        @test abs(df_r.estimate[1] - 2.0) < 0.15
    end

    @testset "Shared validation of the request layer (B7)" begin
        df_c = copy(df)
        df_c.c = categorical(ifelse.(df_c.x3 .> 0, "hi", "lo"))
        df_c.y .+= 0.3 .* (df_c.c .== "hi")
        # (save=:fe: elasticities divide by ŷ, which includes the absorbed fixed effects)
        mc = reg(df_c, @formula(y ~ x1 + x2 + c + fe(state)); save=:fe)
        # measure filter: a categorical's contrasts get no "elasticity" rows
        r = @test_logs (:info, r"Skipping measure=:elasticity") match_mode=:any population_margins(mc, df_c; measure=:elasticity)
        @test Set(DataFrame(r).variable) == Set(["x1", "x2"])
        @test_throws ArgumentError population_margins(mc, df_c; vars=[:c], measure=:elasticity)
        # vars/groups overlap and scenarios/groups conflict, as for GLM
        @test_throws ArgumentError population_margins(mc, df_c; vars=[:c], groups=:c)
        @test_throws ArgumentError population_margins(mc, df_c; vars=[:x1], scenarios=(x2=[0.0],), groups=(:x2, 2))
        # a scenario variable must be a predictor; the absorbed fixed effects count as predictors
        @test_throws ArgumentError population_margins(mc, df_c; vars=[:x1], scenarios=(y=[0.0],))
        @test_throws Margins.MarginsError population_margins(m_fe, df; type=:predictions, scenarios=(state=["CA"],))
        # alpha and contrasts checked
        @test_throws ArgumentError population_margins(m, df; vars=[:x1], alpha=1.5)
        @test_throws ArgumentError profile_margins(m, df, DataFrame(x1=[0.0]); vars=[:x2], alpha=0.0)
        @test_throws ArgumentError population_margins(m, df; vars=[:x1], contrasts=:foo)
        @test population_margins(m, df; vars=[:x1], alpha=1//10).inference.alpha == 0.1
        # a varying grid column that is not a predictor (FE columns are)
        @test_throws ArgumentError profile_margins(m_fe, df, DataFrame(x1=[0.0, 0.0], y=[0.0, 1.0]); type=:predictions)
        # an absorbed FE named in vars gets the FE message, also for a String column
        msg = try population_margins(m, df; vars=[:state]); "" catch e; sprint(showerror, e) end
        @test occursin("absorbed fixed effect", msg)
    end

    @testset "profile_margins interval keyword" begin
        grid = DataFrame(x1=[0.0, 1.0], x2=[0.0, 0.0], x3=[0.0, 0.0], state=["CA", "CA"], year=[2020, 2020])
        a = profile_margins(m_fe, df, grid; type=:predictions)
        b = profile_margins(m_fe, df, grid; type=:predictions, interval=:confidence)
        @test DataFrame(a) == DataFrame(b)
        @test_throws ArgumentError profile_margins(m_fe, df, grid; type=:predictions, interval=:prediction)
        @test_throws ArgumentError profile_margins(m_fe, df, grid; type=:effects, interval=:prediction)
    end

    @testset "Absorbed slopes (fe(g)&x): effects of x blocked, others exact" begin
        # ∂ŷ/∂x includes each state's slope, which is absorbed (not in coef or vcov): the
        # coefficient-based AME of x would leave it out, so it is an error, named or auto-detected
        i = 1:300
        ds = DataFrame(st = "s" .* string.(mod.(i, 12) .+ 1), x = 0.5 .+ mod.(i .* 37, 101) ./ 40,
                       z = cos.(i .* 0.7))
        ds.y = 1 .+ 0.5 .* ds.x .- 0.3 .* ds.z .+ 0.4 .* ds.x .* ds.z .+
               0.2 .* mod.(i, 12) .* ds.x .+ sin.(i .* 2.3)
        ms = reg(ds, @formula(y ~ z + x&z + fe(st)*x); save=:fe)
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        @test_throws Margins.MarginsError population_margins(ms, ds; type=:effects, vars=[:x])
        @test_throws Margins.MarginsError population_margins(ms, ds; type=:effects, vars=[:x, :z])
        auto = msg(() -> population_margins(ms, ds; type=:effects))
        @test occursin("group-specific slopes in :x", auto) && occursin("vars = [:z]", auto)
        @test occursin("group-specific slopes", msg(() -> population_margins(ms, ds; type=:effects, vars=[:x])))
        grid = DataFrame(x = [1.0, 2.0], z = [0.0, 0.0])
        @test_throws Margins.MarginsError profile_margins(ms, ds, grid; type=:effects, vars=[:x])
        @test_throws Margins.MarginsError second_differences(ms, ds, :x, :z)

        # The effect of z does not involve the slopes: β_z + β_xz x, averaged; SE from the gradient
        r = DataFrame(population_margins(ms, ds; type=:effects, vars=[:z]))
        cn = coefnames(ms)
        jz, jxz = findfirst(==("z"), cn), findfirst(==("x & z"), cn)
        # (only z and x&z: FixedEffectModels < 2 reports the dropped common x with NaN in vcov)
        g = [1.0, mean(ds.x)]
        @test r.estimate[1] ≈ coef(ms)[jz] + coef(ms)[jxz] * mean(ds.x) rtol = 1e-10
        @test r.se[1] ≈ sqrt(g' * vcov(ms)[[jz, jxz], [jz, jxz]] * g) rtol = 1e-8
        # x as the modifier of z's effect is fine: ∂AME(z)/∂x = β_xz
        sd = DataFrame(second_differences(ms, ds, :z, :x))
        @test sd.second_diff[1] ≈ coef(ms)[jxz] rtol = 1e-6
        # predictions were already blocked for these models
        @test_throws Margins.MarginsError population_margins(ms, ds; type=:predictions)
    end
end
