# sd_model_level.jl
# second_differences(model, data, variables, modifier; …): binary, categorical and continuous
# modifiers, the local derivative (`at` = :mean / :median / a value / points / a NamedTuple),
# `levels`, `profile` strata, `delta`, `scale`, weights, the integer-code warning and input errors

using Test
using Margins
using DataFrames
using GLM
using CategoricalArrays
using Random
using Statistics

@testset "second_differences(model, data, …)" begin
    Random.seed!(7)
    n = 800
    df = DataFrame(
        x = randn(n), w = randn(n), z = randn(n),
        t = rand([0, 1], n), g = rand(["a", "b", "c"], n),
    )
    df.y = 0.2 .+ 0.4df.x .+ 0.3df.w .+ 0.5df.t .+ 0.7df.x .* df.t .- 0.6df.x .* df.w .+
           0.3 .* (df.g .== "b") .+ 0.4 .* df.x .* (df.g .== "c") .+ 0.5df.z .+ randn(n)
    lmod = lm(@formula(y ~ x * t + x * w + x * g + z), df)
    β = coef(lmod); names = coefnames(lmod)
    b(nm) = β[findfirst(==(nm), names)]

    @testset "binary modifier: equals the interaction coefficient in a linear model" begin
        sd = second_differences(lmod, df, :x, :t)
        @test nrow(sd) == 1
        @test sd.second_diff[1] ≈ b("x & t") atol=1e-8
        @test sd.modifier_type[1] == :binary
        @test (sd.modifier_level1[1], sd.modifier_level2[1]) == (0, 1)
        @test sd.df[1] == dof_residual(lmod)
        # same answer as building it by hand
        ames = population_margins(lmod, df; vars=:x, scenarios=(t=[0, 1],), type=:effects)
        by_hand = second_differences(ames, :x, :t)
        @test sd.second_diff[1] ≈ by_hand.second_diff[1]
        @test sd.se[1] ≈ by_hand.se[1]
    end

    @testset "continuous modifier: local derivative and slope between levels" begin
        loc = second_differences(lmod, df, :x, :w)
        @test loc.second_diff[1] ≈ b("x & w") atol=1e-6
        @test isnan(loc.eval_point[1])                                   # default at = :observed: no single point
        @test isequal(loc, second_differences(lmod, df, :x, :w; at = :observed))
        @test second_differences(lmod, df, :x, :w; at = :mean).eval_point[1] ≈ sum(df.w) / n
        @test loc.modifier_type[1] == :continuous
        sl = second_differences(lmod, df, :x, :w; levels=[-1.0, 0.0, 1.0])
        @test nrow(sl) == 3
        @test all(isapprox.(sl.second_diff, b("x & w"); atol=1e-8))
        two = second_differences(lmod, df, :x, :w; levels=[0.0, 10.0])   # slope, not raw difference
        @test two.second_diff[1] ≈ b("x & w") atol=1e-8
        @test two.modifier_type[1] == :continuous
        @test second_differences(lmod, df, :x, :w; at=[-1.0, 1.0]).eval_point == [-1.0, 1.0]
    end

    @testset "categorical modifier, several focal variables" begin
        sd = second_differences(lmod, df, [:x, :z], :g; warn=false)
        @test nrow(sd) == 6                       # 2 variables × 3 level pairs
        @test Set(sd.variable) == Set([:x, :z])
        xr = sd[(sd.variable .== :x) .&& (sd.modifier_level1 .== "a") .&& (sd.modifier_level2 .== "c"), :]
        @test xr.second_diff[1] ≈ b("x & g: c") atol=1e-8
        zr = sd[sd.variable .== :z, :]
        @test all(abs.(zr.second_diff) .< 1e-8)    # no z × g interaction
    end

    @testset "categorical focal variable and categorical modifier array" begin
        d2 = copy(df); d2.gc = categorical(d2.g; levels=["c", "b", "a"])
        m2 = lm(@formula(y ~ x + t * gc + w + z), d2)
        sd = second_differences(m2, d2, :gc, :t)
        @test nrow(sd) == 2                       # baseline contrasts
        @test all(sd.modifier_level1 .== 0) && all(sd.modifier_level2 .== 1)
        sdm = second_differences(m2, d2, :x, :gc; warn=false)    # no x:gc term in m2
        @test (sdm.modifier_level1[1], sdm.modifier_level2[1]) == ("c", "b")   # level order, not alphabetical
    end

    @testset "profile strata and options" begin
        sd = second_differences(lmod, df, :x, :t; profile=(g=["a", "b"],))
        @test nrow(sd) == 2 && sd.g == ["a", "b"]
        # integer-coded categorical modifier
        d3 = copy(df); d3.k = rand(1:3, n)
        m3 = lm(@formula(y ~ x * k + t + w + z), d3)     # k enters the model, with an interaction
        auto = @test_logs (:warn, r"takes only 3 integer values \(1, 2, 3\)") second_differences(m3, d3, :x, :k)
        @test all(auto.modifier_type .== :continuous)
        # no warning when the user states what they mean
        @test_logs second_differences(m3, d3, :x, :k; at=2)
        @test_logs second_differences(m3, d3, :x, :k; levels=[1, 3])
        cat = @test_logs second_differences(m3, d3, :x, :k; modifier_type=:categorical)
        @test all(cat.modifier_type .== :categorical)
        # many distinct integers (an age-like variable) is continuous without a warning
        d3.age = rand(20:70, n)
        m3b = lm(@formula(y ~ x * age + t + z), d3)
        @test_logs second_differences(m3b, d3, :x, :age)
        # non-integer numeric modifier: no warning
        @test_logs second_differences(lmod, df, :x, :w)
    end

    @testset "logit: model-level equals the hand-built result" begin
        yb = df.y .> 0
        d4 = copy(df); d4.yb = yb
        gm = glm(@formula(yb ~ x * t + w), d4, Binomial(), LogitLink())
        a = second_differences(gm, d4, :x, :t)
        ames = population_margins(gm, d4; vars=:x, scenarios=(t=[0, 1],), type=:effects)
        h = second_differences(ames, :x, :t)
        @test a.second_diff[1] ≈ h.second_diff[1] && a.se[1] ≈ h.se[1]
        @test a.df[1] == Inf
    end

    @testset "errors" begin
        @test_throws ArgumentError second_differences(lmod, df, :x, :nope)
        @test_throws ArgumentError second_differences(lmod, df, :x, :w; at=0.0, levels=[0.0, 1.0])
        @test_throws ArgumentError second_differences(lmod, df, :x, :t; at=0.5)
        @test_throws ArgumentError second_differences(lmod, df, :x, :t; modifier_type=:nonsense)
        @test_throws ArgumentError second_differences(lmod, df, :x, :t; levels=[1])
        @test_throws Exception second_differences(lmod, df, :t, :t)
    end
end

@testset "second_differences(model, data, x, m; at=…) - Local Derivatives" begin
    Random.seed!(789)
    n = 200

    # Generate test data with interactions
    df = DataFrame(
        x1 = randn(n),
        x2 = randn(n),
        x3 = randn(n),
        age = rand(25:65, n) .+ randn(n),  # Continuous modifier
        income = rand(20000:80000, n) .+ randn(n) .* 1000,  # Another continuous modifier
        region = rand(["north", "south", "west"], n)
    )

    # Model with interactions
    df.y = 0.5 .+ 0.3 .* df.x1 .+ 0.2 .* df.x2 .+ 0.1 .* df.x3 .+
           0.01 .* df.age .+ 0.0001 .* df.income .+
           0.02 .* df.x1 .* df.age .+  # Interaction with age
           0.01 .* df.x2 .* df.age .+
           randn(n) .* 0.5

    model = lm(@formula(y ~ (x1 + x2 + x3) * age + income + region), df)
    Σ = vcov(model)

    @testset "Basic Functionality - At Mean" begin
        # Single variable at mean
        sd = second_differences(model, df, :x1, :age; vcov=Σ, modifier_type=:continuous, at=:mean)

        @test nrow(sd) == 1
        @test sd.variable[1] == :x1
        @test sd.modifier[1] == :age
        @test sd.eval_point[1] ≈ mean(df.age) rtol=0.01
        @test isfinite(sd.second_diff[1])
        @test isfinite(sd.se[1])
        @test sd.se[1] > 0
        @test isfinite(sd.statistic[1])
        @test 0 <= sd.p_value[1] <= 1
        @test sd.delta_used[1] > 0
    end

    @testset "At Median" begin
        sd = second_differences(model, df, :x1, :age; vcov=Σ, modifier_type=:continuous, at=:median)

        @test nrow(sd) == 1
        @test sd.eval_point[1] ≈ median(df.age) rtol=0.01
    end

    @testset "At Specific Value" begin
        sd = second_differences(model, df, :x1, :age; vcov=Σ, modifier_type=:continuous, at=45.0)

        @test nrow(sd) == 1
        @test sd.eval_point[1] == 45.0
    end

    @testset "Multiple Evaluation Points" begin
        sd = second_differences(model, df, :x1, :age; vcov=Σ, modifier_type=:continuous, at=[30.0, 45.0, 60.0])

        @test nrow(sd) == 3
        @test sd.eval_point == [30.0, 45.0, 60.0]
        @test all(sd.variable .== :x1)
        @test all(sd.modifier .== :age)
    end

    @testset "Multiple Variables - Single Eval Point" begin
        sd = second_differences(model, df, [:x1, :x2, :x3], :age; vcov=Σ, modifier_type=:continuous, at=:mean)

        @test nrow(sd) == 3
        @test Set(sd.variable) == Set([:x1, :x2, :x3])
        @test all(sd.modifier .== :age)
        @test all(sd.eval_point .≈ mean(df.age))
        @test all(isfinite.(sd.second_diff))
        @test all(sd.se .> 0)
    end

    @testset "Multiple Variables × Multiple Eval Points" begin
        sd = second_differences(model, df, [:x1, :x2], :age; vcov=Σ, modifier_type=:continuous, at=[30.0, 45.0, 60.0])

        @test nrow(sd) == 6  # 2 variables × 3 eval points
        @test count(sd.variable .== :x1) == 3
        @test count(sd.variable .== :x2) == 3

        # Check eval points are correct for each variable
        x1_rows = sd[sd.variable .== :x1, :]
        @test x1_rows.eval_point == [30.0, 45.0, 60.0]
    end

    @testset "Profile Support - Single Variable" begin
        # With profile holding income constant
        sd = second_differences(model, df, :x1, :age; vcov=Σ, modifier_type=:continuous,
                                  at=45.0,
                                  profile=(income=50000,))

        @test nrow(sd) == 1
        @test sd.eval_point[1] == 45.0
        @test sd.income[1] == 50000
    end

    @testset "Profile Support - Multiple Profile Variables" begin
        sd = second_differences(model, df, :x1, :age; vcov=Σ, modifier_type=:continuous,
                                  at=45.0,
                                  profile=(income=50000, region="north"))

        @test nrow(sd) == 1
        @test sd.income[1] == 50000
        @test sd.region[1] == "north"
    end

    @testset "Profile with Multiple Variables and Eval Points" begin
        sd = second_differences(model, df, [:x1, :x2], :age; vcov=Σ, modifier_type=:continuous,
                                  at=[30.0, 60.0],
                                  profile=(income=50000,))

        @test nrow(sd) == 4  # 2 variables × 2 eval points
        @test all(sd.income .== 50000)
        @test count(sd.eval_point .== 30.0) == 2  # x1 and x2
        @test count(sd.eval_point .== 60.0) == 2  # x1 and x2
    end

    @testset "NamedTuple for at Parameter" begin
        # Full profile specification via `at`
        sd = second_differences(model, df, :x1, :age; vcov=Σ, modifier_type=:continuous,
                                  at=(age=45.0, income=50000, region="north"))

        @test nrow(sd) == 1
        @test sd.eval_point[1] == 45.0
        @test sd.income[1] == 50000
        @test sd.region[1] == "north"
    end

    @testset "NamedTuple with Both at and profile" begin
        # Combine `at` NamedTuple with `profile` parameter
        sd = second_differences(model, df, :x1, :age; vcov=Σ, modifier_type=:continuous,
                                  at=(age=45.0, income=50000),
                                  profile=(region="south",))

        @test nrow(sd) == 1
        @test sd.eval_point[1] == 45.0
        @test sd.income[1] == 50000
        @test sd.region[1] == "south"
    end

    @testset "Custom Delta" begin
        sd1 = second_differences(model, df, :x1, :age; vcov=Σ, modifier_type=:continuous, at=:mean, delta=:auto)
        sd2 = second_differences(model, df, :x1, :age; vcov=Σ, modifier_type=:continuous, at=:mean, delta=1.0)

        @test sd1.delta_used[1] ≈ 0.01 * std(df.age) rtol=0.01
        @test sd2.delta_used[1] == 1.0
    end

    @testset "Scale Parameter" begin
        # Test both :link and :response scales
        sd_response = second_differences(model, df, :x1, :age; vcov=Σ, modifier_type=:continuous, scale=:response)
        sd_link = second_differences(model, df, :x1, :age; vcov=Σ, modifier_type=:continuous, scale=:link)

        # For linear models, they should be the same
        @test sd_response.second_diff[1] ≈ sd_link.second_diff[1]
    end

    @testset "Error Handling - Invalid Modifier" begin
        @test_throws ArgumentError second_differences(
            model, df, :x1, :nonexistent; vcov=Σ, modifier_type=:continuous
        )
    end

    @testset "Error Handling - Non-Numeric Modifier" begin
        @test_throws ArgumentError second_differences(
            model, df, :x1, :region; vcov=Σ, modifier_type=:continuous
        )
    end

    @testset "Error Handling - Invalid Variable" begin
        @test_throws ArgumentError second_differences(
            model, df, :nonexistent, :age; vcov=Σ, modifier_type=:continuous
        )
    end

    @testset "Error Handling - Invalid Profile Variable" begin
        @test_throws ArgumentError second_differences(
            model, df, :x1, :age; vcov=Σ, modifier_type=:continuous,
            profile=(nonexistent=100,))
    end

    @testset "Error Handling - Invalid at Parameter" begin
        @test_throws ArgumentError second_differences(
            model, df, :x1, :age; vcov=Σ, modifier_type=:continuous,
            at="invalid")
    end

    @testset "Error Handling - NamedTuple without Modifier" begin
        @test_throws ArgumentError second_differences(
            model, df, :x1, :age; vcov=Σ, modifier_type=:continuous,
            at=(income=50000,))  # Missing :age
    end

    @testset "Error Handling - Invalid Delta" begin
        @test_throws TypeError second_differences(
            model, df, :x1, :age; vcov=Σ, modifier_type=:continuous,
            delta="invalid")
    end

    @testset "Error Handling - Invalid Contrasts" begin
        # An ArgumentError from the request layer, as in every other entry point (3.0; was a
        # bare ErrorException here and silently accepted on the discrete path)
        @test_throws ArgumentError second_differences(
            model, df, :x1, :age; vcov=Σ, modifier_type=:continuous,
            contrasts=:invalid)
    end

    @testset "Real-World Example - Interaction Detection" begin
        # Model with strong interaction
        Random.seed!(999)
        n = 300
        df_interact = DataFrame(
            x = randn(n),
            m = randn(n) .* 10 .+ 50
        )
        df_interact.y = 1.0 .+ 0.5 .* df_interact.x .+
                        0.02 .* df_interact.m .+
                        0.1 .* df_interact.x .* df_interact.m .+  # Strong interaction
                        randn(n) .* 0.5

        model_interact = lm(@formula(y ~ x * m), df_interact)

        # Compute derivative at mean
        sd = second_differences(model_interact, df_interact, :x, :m; vcov=vcov(model_interact), modifier_type=:continuous)

        # Should detect significant interaction
        @test abs(sd.statistic[1]) > 2  # Statistically significant
        @test sd.p_value[1] < 0.05
    end

    @testset "Vector-Valued Profile Support" begin
        # Create fresh data frame for stratification tests with categorical variables
        Random.seed!(888)
        df_strat = DataFrame(
            x1 = randn(n),
            x2 = randn(n),
            x3 = randn(n),
            age = rand(25:65, n) .+ randn(n),
            income = rand(20000:80000, n) .+ randn(n) .* 1000,
            socio4 = rand([false, true], n),
            gender = rand(["male", "female"], n)
        )

        # Generate y with interactions
        df_strat.y = 0.5 .+ 0.3 .* df_strat.x1 .+ 0.2 .* df_strat.x2 .+ 0.1 .* df_strat.x3 .+
                     0.01 .* df_strat.age .+ 0.0001 .* df_strat.income .+
                     0.02 .* df_strat.x1 .* df_strat.age .+
                     randn(n) .* 0.5

        # Refit model with categorical variables
        model_strat = lm(@formula(y ~ (x1 + x2 + x3) * age + income + socio4 + gender), df_strat)
        Σ_strat = vcov(model_strat)

        @testset "Single Vector Profile" begin
            # Vector-valued profile: separate estimation for each socio4 level
            sd = second_differences(model_strat, df_strat, :x1, :age; vcov=Σ_strat, modifier_type=:continuous,
                                      at=45.0,
                                      profile=(socio4=[false, true],))

            @test nrow(sd) == 2  # One row per socio4 level
            @test sd.variable == [:x1, :x1]
            @test sd.modifier == [:age, :age]
            @test sd.eval_point == [45.0, 45.0]
            @test hasproperty(sd, :socio4)
            @test Set(sd.socio4) == Set([false, true])
            @test all(isfinite.(sd.second_diff))
            @test all(sd.se .> 0)
        end

        @testset "Vector Profile with Scalar Profile" begin
            # Mixed: vector socio4 + scalar income
            sd = second_differences(model_strat, df_strat, :x1, :age; vcov=Σ_strat, modifier_type=:continuous,
                                      at=45.0,
                                      profile=(socio4=[false, true],
                                              income=50000.0))

            @test nrow(sd) == 2
            @test hasproperty(sd, :socio4)
            @test hasproperty(sd, :income)
            @test all(sd.income .== 50000.0)  # Scalar stays constant
            @test Set(sd.socio4) == Set([false, true])  # Vector varies
        end

        @testset "Complex Combination: All Dimensions" begin
            # Combine everything: multiple variables × multiple eval points × vector profiles
            sd = second_differences(model_strat, df_strat, [:x1, :x2], :age; vcov=Σ_strat, modifier_type=:continuous,
                                      at=[30.0, 60.0],
                                      profile=(socio4=[false, true],
                                              gender=["male", "female"]))

            @test nrow(sd) == 16  # 2 variables × 2 eval points × 2 socio4 × 2 gender

            # Every combination appears once (the full Cartesian product)
            @test Set(zip(sd.variable, sd.eval_point, sd.socio4, sd.gender)) ==
                  Set(Iterators.product([:x1, :x2], [30.0, 60.0], [false, true], ["male", "female"]))
            @test all(sd.modifier .== :age)

            @test all(isfinite.(sd.second_diff))
            @test all(sd.se .> 0)
        end
    end
end

@testset "second differences: `at` forms, profile and modifier checks, warnings, weights" begin
    Random.seed!(21)
    n = 600
    df = DataFrame(x = randn(n), w = randn(n), z = rand([false, true], n), t = rand([0, 1], n),
                   g = rand(["a", "b", "c"], n))
    df.y = 0.2 .+ 0.4df.x .+ 0.3df.w .+ 0.5df.z .+ 0.6df.x .* df.z .- 0.4df.x .* df.w .+ randn(n)
    df.yb = (0.3 .+ 0.6df.x .+ 0.5df.z .+ 0.4df.x .* df.z .+ randn(n)) .> 0
    lmod = lm(@formula(y ~ x * z + x * w + t + g), df)
    lg = glm(@formula(yb ~ x * z + w), df, Binomial(), LogitLink())

    @testset "`at` forms" begin
        a1 = second_differences(lmod, df, :x, :w; at = [0.0, 0.5, 1.0])
        a2 = second_differences(lmod, df, :x, :w; at = 0.0:0.5:1.0)
        @test a1.second_diff ≈ a2.second_diff
        t = second_differences(lmod, df, :x, :w; at = (w = [0.0, 1.0], t = 1))
        @test nrow(t) == 2 && t.eval_point == [0.0, 1.0]
    end

    @testset "profile must not contain the modifier" begin
        @test_throws ArgumentError second_differences(lmod, df, :x, :w; profile = (w = 1.0,))
        @test_throws ArgumentError second_differences(lmod, df, :x, :z; profile = (z = true,))
        err = try second_differences(lmod, df, :x, :w; profile = (w = 1.0,)) catch e; sprint(showerror, e) end
        @test occursin("contains the modifier", err)
    end

    @testset "a Bool modifier cannot be forced continuous" begin
        err = try second_differences(lg, df, :x, :z; modifier_type = :continuous) catch e; sprint(showerror, e) end
        @test occursin("Bool", err)
        @test second_differences(lg, df, :x, :z).modifier_type == [:binary]
    end

    @testset "integer codes stored as floats are flagged; warn=false silences it" begin
        d = copy(df); d.k = Float64.(rand(1:3, n))
        m = lm(@formula(y ~ x * k + w), d)
        @test_logs (:warn, r"takes only 3 integer values") second_differences(m, d, :x, :k)
        @test_logs second_differences(m, d, :x, :k; warn = false)
        @test_logs second_differences(m, d, :x, :k; at = :mean)
    end

    @testset "warn=false silences every warning" begin
        d = copy(df); d.k = rand([10.0, 20.0], n)
        m = lm(@formula(y ~ x * k + w), d)
        ames = population_margins(m, d; vars = :x, scenarios = (k = [10.0, 20.0],), type = :effects)
        @test_logs (:warn, r"treated as :binary") second_differences(ames, :x, :k)
        @test_logs second_differences(ames, :x, :k; warn = false)
        @test_logs second_differences(m, d, :x, :k; levels = [10.0, 20.0], delta = 0.1, warn = false)
    end

    @testset "binary route warns like the pairwise route" begin
        d = copy(df); d.k = rand([10.0, 20.0], n)
        d.cg = categorical(d.g)
        m = lm(@formula(y ~ x * k + cg * k + w), d)
        ames = population_margins(m, d; vars = :cg, scenarios = (k = [10.0, 20.0],), type = :effects)
        @test_logs (:warn, r"treated as :binary") second_differences(ames, :cg, :k)
    end

    @testset "weights" begin
        w = 1.0 .+ 2.0 .* (df.x .> 0)          # weight 3 for x > 0, else 1
        dup = vcat(df, df[df.x .> 0, :], df[df.x .> 0, :])  # same weighting as 2 extra copies
        un = second_differences(lg, df, :x, :z)
        wt = second_differences(lg, df, :x, :z; weights = w)
        @test !isapprox(wt.second_diff[1], un.second_diff[1]; rtol = 1e-6)
        # equals the unweighted calculation on the expanded data, and a weights column works too
        ex = second_differences(lg, dup, :x, :z)
        @test wt.second_diff ≈ ex.second_diff && wt.se ≈ ex.se
        dw = copy(df); dw.sw = w
        @test second_differences(lg, dw, :x, :z; weights = :sw).second_diff ≈ wt.second_diff
        @test second_differences(lg, df, :x, :z; weights = ones(n)).second_diff ≈ un.second_diff
        # same as the result-level form on a weighted result
        ames = population_margins(lg, df; vars = :x, scenarios = (z = [false, true],), type = :effects, weights = w)
        @test second_differences(ames, :x, :z).second_diff ≈ wt.second_diff
        # local derivative: weighted mean as the point for at = :mean, weighted effects
        lw = second_differences(lg, df, :x, :w; weights = w, delta = 0.01, at = :mean)
        @test lw.eval_point[1] ≈ sum(w .* df.w) / sum(w)
        lex = second_differences(lg, dup, :x, :w; at = lw.eval_point[1], delta = 0.01)
        @test lw.second_diff ≈ lex.second_diff
        lmed = second_differences(lg, df, :x, :w; weights = ones(n), at = :median, delta = 0.01)
        @test lmed.eval_point[1] ≈ sort(df.w)[n ÷ 2] / 2 + sort(df.w)[n ÷ 2 + 1] / 2
        # invalid weights are rejected
        @test_throws ArgumentError second_differences(lg, df, :x, :z; weights = ones(n - 1))
        @test_throws ArgumentError second_differences(lg, df, :x, :z; weights = :nope)
    end
end
