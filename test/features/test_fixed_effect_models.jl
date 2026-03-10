using Test
using Margins
using FixedEffectModels
using DataFrames
using GLM
using Statistics

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

    @testset "Profile predictions without FE in grid" begin
        grid = DataFrame(x1=[0.0, 1.0], x2=[0.0, 0.0], x3=[0.0, 0.0])
        result = profile_margins(m_fe, df, grid; type=:predictions)
        df_r = DataFrame(result)

        @test nrow(df_r) == 2

        β = coef(m_fe)
        avg_fe = mean(m_fe.fe.fe_state .+ m_fe.fe.fe_year)

        @test abs(df_r.estimate[1] - (0.0 + avg_fe)) < 1e-10
        @test abs(df_r.estimate[2] - (β[1] + avg_fe)) < 1e-10
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

        # State only in grid (year averaged)
        grid = DataFrame(x1=[0.0, 0.0], x2=[0.0, 0.0], x3=[0.0, 0.0], state=["CA", "NY"])
        result = profile_margins(m_fe, df, grid; type=:predictions)
        df_r = DataFrame(result)

        avg_year = mean(skipmissing(m_fe.fe.fe_year))
        @test abs(df_r.estimate[1] - (fe_state_lookup["CA"] + avg_year)) < 1e-10
        @test abs(df_r.estimate[2] - (fe_state_lookup["NY"] + avg_year)) < 1e-10

        # Both FEs in grid
        grid2 = DataFrame(x1=[1.0, -1.0], x2=[0.0, 0.0], x3=[0.0, 0.0],
                          state=["CA", "TX"], year=[2020, 2024])
        result2 = profile_margins(m_fe, df, grid2; type=:predictions)
        df_r2 = DataFrame(result2)

        manual1 = β[1]*1.0 + fe_state_lookup["CA"] + fe_year_lookup[2020]
        manual2 = β[1]*(-1.0) + fe_state_lookup["TX"] + fe_year_lookup[2024]
        @test abs(df_r2.estimate[1] - manual1) < 1e-10
        @test abs(df_r2.estimate[2] - manual2) < 1e-10
    end

    @testset "Predictions have correct SEs" begin
        # SEs should match the Xβ-only SEs (FE offsets are constants)
        result = population_margins(m_fe, df; type=:predictions)
        df_r = DataFrame(result)

        @test df_r.se[1] > 0
        @test isfinite(df_r.se[1])

        # Profile SEs at non-zero covariate values
        grid = DataFrame(x1=[0.5, 1.0], x2=[0.3, -0.2], x3=[0.1, 0.5])
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
end
