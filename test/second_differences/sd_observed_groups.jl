# sd_observed_groups.jl
# at = :observed (the average derivative over the observed modifier values) and groups= on
# second_differences(model, data, …), with backend = :ad vs :fd.
#
# - at = :observed: the interaction coefficient in a linear model, the row layout, and a
#   closed-form oracle (the average unit-level cross-derivative, estimate and SE).
# - groups=: the counterfactual second difference within each stratum. Checked (1) exactly against
#   the result-level route (population_margins with the same scenarios and groups, then
#   second_differences(result, …)), and (2) against an oracle: the ungrouped call on that stratum's
#   rows only, with the same Σ.
# - backend=:ad vs :fd agree on second differences.
# (The FixedEffectModels estimation sample under at = :mean is checked in
# validation/test_second_differences_mixed_fem.jl.)

using Test
using Margins
using DataFrames, CategoricalArrays, GLM, StatsModels, Tables, Statistics, Random, LinearAlgebra

# The schema columns two second-difference frames share, compared exactly (NaN == NaN)
const _SD_COLS = [:variable, :contrast, :modifier, :modifier_type, :modifier_level1, :modifier_level2,
                  :ame_at_level1, :ame_at_level2, :second_diff, :se, :statistic, :df, :p_value,
                  :ci_lower, :ci_upper, :significant]
_same(a, b) = isequal(a, b)
function _frames_equal(a::DataFrame, b::DataFrame, cols)
    nrow(a) == nrow(b) || return false
    return all(c -> all(_same.(a[!, c], b[!, c])), cols)
end


@testset "second differences: at = :observed" begin
    Random.seed!(41)
    n = 500
    df = DataFrame(x = randn(n), w = randn(n) .+ 0.3, z = randn(n), t = rand([0.0, 1.0], n), sw = 0.5 .+ rand(n))
    df.y = 0.2 .+ 0.4df.x .+ 0.3df.w .+ 0.2df.z .+ 0.5df.x .* df.w .+ randn(n)
    df.yb = (0.1 .+ 0.7df.x .+ 0.4df.w .+ 0.3df.x .* df.w .+ 0.3 .* df.x .^ 2 .+ randn(n)) .> 0
    lin = lm(@formula(y ~ x * w + z), df)
    lg = glm(@formula(yb ~ x * w + z + t + x^2), df, Binomial(), LogitLink())

    @testset "linear model: the interaction coefficient, like any evaluation point" begin
        k = findfirst(==("x & w"), coefnames(lin))
        ob = second_differences(lin, df, :x, :w; at = :observed)
        @test nrow(ob) == 1
        @test ob.second_diff[1] ≈ coef(lin)[k] rtol = 1e-6
        @test ob.se[1] ≈ sqrt(vcov(lin)[k, k]) rtol = 1e-6
        @test ob.df[1] == dof_residual(lin)
    end

    @testset "row layout" begin
        ob = second_differences(lg, df, :x, :w; at = :observed, warn = false)
        @test isnan(ob.eval_point[1]) && isnan(ob.modifier_level1[1]) && isnan(ob.modifier_level2[1])
        @test ob.delta_used[1] > 0 && ob.modifier_type == [:continuous]
        @test ob.variable == [:x] && ob.contrast == ["dy/dx"]
        # differs from the derivative at the mean of the modifier in a nonlinear model
        @test !isapprox(ob.second_diff[1], second_differences(lg, df, :x, :w; at = :mean, warn = false).second_diff[1]; rtol = 1e-3)
        # profile strata, several points, several variables, p_adjust, gradients, contrast selection
        pr = second_differences(lg, df, [:x, :z], :w; at = :observed, profile = (t = [0.0, 1.0],), p_adjust = :holm,
                                include_gradients = true, warn = false)
        @test nrow(pr) == 4 && sort(unique(pr.t)) == [0.0, 1.0] && all(isnan, pr.eval_point)
        @test hasproperty(pr, :p_adjusted) && hasproperty(pr, :gradient)
        @test second_differences(lg, df, :x, :w; at = :observed, contrast = "dy/dx", warn = false).second_diff == ob.second_diff
    end

    @testset "equals the average unit-level cross-derivative (closed-form oracle)" begin
        rhs = formula(lg).rhs
        X(dd) = modelcols(rhs, Tables.columntable(dd))
        function est(β, scale, w; δ = 0.01, h = 1e-5)
            function ame(shift)
                dd = copy(df); dd.w .+= shift
                dp = copy(dd); dp.x .+= h
                dm = copy(dd); dm.x .-= h
                pred(Xs) = scale === :response ? 1 ./ (1 .+ exp.(-(Xs * β))) : Xs * β
                g = (pred(X(dp)) .- pred(X(dm))) ./ (2h)
                isnothing(w) ? mean(g) : sum(w .* g) / sum(w)
            end
            (ame(δ) - ame(-δ)) / (2δ)
        end
        β = coef(lg)
        for (scale, w) in ((:response, nothing), (:link, nothing), (:response, df.sw))
            grad = map(eachindex(β)) do k
                st = 1e-5 * max(abs(β[k]), 1e-2)
                bp = copy(β); bp[k] += st; bm = copy(β); bm[k] -= st
                (est(bp, scale, w) - est(bm, scale, w)) / (2st)
            end
            sd = second_differences(lg, df, :x, :w; at = :observed, delta = 0.01, scale, weights = w, warn = false)
            @test sd.second_diff[1] ≈ est(β, scale, w) rtol = 1e-5
            @test sd.se[1] ≈ sqrt(grad' * vcov(lg) * grad) rtol = 2e-3
        end
    end

    @testset "a modifier the model does not use is a structural zero" begin
        m = lm(@formula(y ~ x + z), df)
        sd = @test_logs (:warn, r"zero by construction") second_differences(m, df, :x, :w; at = :observed)
        @test sd.second_diff[1] == 0.0 && isnan(sd.p_value[1])
    end

    @testset "errors" begin
        @test_throws ArgumentError second_differences(lg, df, :x, :t; at = :observed)   # 0/1 modifier is discrete
        @test_throws ArgumentError second_differences(lg, df, :x, :w; at = :nonsense)
    end
end

@testset "second differences: groups=, backend=" begin
    Random.seed!(8008)
    n = 900
    df = DataFrame(x = randn(n), w = randn(n), t = rand(0:1, n),
                   g = categorical(rand(["a", "b", "c"], n)), k = categorical(rand(["p", "q"], n)),
                   z = rand(n) .* 4)
    df.wt = 0.5 .+ rand(n)
    η = 0.2 .+ 0.8 .* df.x .+ 0.5 .* df.x .* df.w .+ 0.6 .* df.t .* df.x .- 0.3 .* df.w .+
        0.4 .* (df.g .== "b") .- 0.3 .* (df.k .== "q") .* df.x .+ 0.1 .* df.z
    df.yb = rand(n) .< 1 ./ (1 .+ exp.(-η))
    m = glm(@formula(yb ~ x * w + x * t + g + k * x + z), df, Binomial(), LogitLink())
    V = vcov(m)

    @testset "discrete modifier: equals the result-level route, exactly" begin
        for G in (:g, [:g, :k], :g => :k, (:z, 3), [(:z, 2), :k])
            sd = second_differences(m, df, [:x, :w], :t; groups = G, include_gradients = true)
            r = population_margins(m, df; vars = [:x, :w], scenarios = (t = [0, 1],), groups = G)
            rl = second_differences(r, [:x, :w], :t; include_gradients = true)
            @test _frames_equal(sd, rl, [_SD_COLS; :gradient])
            over = [Symbol("over_", v) for v in Margins._extract_group_variables(G)]
            @test all(c -> c in propertynames(sd), over)
            @test names(sd)[end - length(over) + 1:end] == string.(over)   # over_ columns last, user order
            @test _frames_equal(sd, rl, over)
            # one row per (stratum, variable)
            @test nrow(sd) == 2 * length(unique(Tuple.(eachrow(sd[!, over]))))
            # every SE is the delta-method SE of its own gradient, with the model's Σ
            @test all(i -> sd.se[i] ≈ sqrt(dot(sd.gradient[i], V, sd.gradient[i])), 1:nrow(sd))
        end
        # user order of the over_ columns
        @test names(second_differences(m, df, :x, :t; groups = [:k, :g]))[end-1:end] == ["over_k", "over_g"]
        @test names(second_differences(m, df, :x, :t; groups = [:g, :k]))[end-1:end] == ["over_g", "over_k"]
    end

    @testset "discrete modifier: a stratum equals the ungrouped call on its rows (same Σ)" begin
        sd = second_differences(m, df, [:x, :k], :t; groups = :g, vcov = V)
        for lv in ("a", "b", "c")
            sub = df[df.g .== lv, :]
            o = second_differences(m, sub, [:x, :k], :t; vcov = V, levels = [0, 1])
            rows = sd[sd.over_g .== lv, :]
            @test nrow(rows) == nrow(o) == 2
            @test maximum(abs.(rows.second_diff .- o.second_diff)) <= 1e-12
            @test maximum(abs.(rows.se .- o.se)) <= 1e-12
            @test maximum(abs.(rows.ame_at_level1 .- o.ame_at_level1)) <= 1e-12
        end
    end

    @testset "at=:observed: within-stratum average cross-derivative" begin
        δ = 0.02
        sd = second_differences(m, df, :x, :w; groups = :g, delta = δ, include_gradients = true)
        @test all(isnan, sd.eval_point) && all(==(δ), sd.delta_used)
        @test Set(sd.over_g) == Set(["a", "b", "c"])
        for lv in ("a", "b", "c")
            o = second_differences(m, df[df.g .== lv, :], :x, :w; delta = δ, vcov = V)
            row = sd[sd.over_g .== lv, :]
            @test abs(row.second_diff[1] - o.second_diff[1]) <= 1e-12
            @test abs(row.se[1] - o.se[1]) <= 1e-12
        end
        # the strata differ (the sample composition differs) and their joint covariance is g₁'Σg₂
        @test length(unique(sd.second_diff)) == 3
        g1, g2 = sd.gradient[1], sd.gradient[2]
        se_diff = sqrt(dot(g1 .- g2, V, g1 .- g2))
        @test se_diff ≈ sqrt(sd.se[1]^2 + sd.se[2]^2 - 2 * dot(g1, V, g2))
        # the joint form agrees with the stratum-wise form only through this covariance
        @test se_diff < sd.se[1] + sd.se[2]
    end

    @testset "at=:mean / :median: each stratum's own (weighted) mean / median" begin
        δ = 0.02
        for at in (:mean, :median), wts in (:none, :wt)
            sd = second_differences(m, df, :x, :w; groups = :g, at, delta = δ, weights = wts)
            for lv in ("a", "b", "c")
                sub = df[df.g .== lv, :]
                ws = wts === :none ? nothing : sub.wt
                pt = at === :mean ? (isnothing(ws) ? mean(sub.w) : sum(ws .* sub.w) / sum(ws)) :
                                    (isnothing(ws) ? median(sub.w) : Margins._weighted_median(sub.w, ws))
                row = sd[sd.over_g .== lv, :]
                @test row.eval_point[1] == pt
                o = second_differences(m, sub, :x, :w; at, delta = δ, vcov = V, weights = wts)
                @test o.eval_point[1] == pt
                @test abs(row.second_diff[1] - o.second_diff[1]) <= 1e-12
                @test abs(row.se[1] - o.se[1]) <= 1e-12
            end
        end
    end

    @testset "at = a point / points, and levels: equal the result-level route" begin
        δ = 0.05
        for at in (0.3, [-0.5, 0.3])
            sd = second_differences(m, df, :x, :w; groups = [:k, :g], at, delta = δ)
            pts = at isa Real ? [at] : at
            @test nrow(sd) == length(pts) * 6
            for (j, a) in enumerate(pts)
                r = population_margins(m, df; vars = [:x], scenarios = (w = [a - δ, a + δ],), groups = [:k, :g])
                rl = second_differences(r, :x, :w; modifier_type = :continuous)
                block = sd[sd.eval_point .== a, :]
                @test _frames_equal(block, rl, [_SD_COLS; :over_k; :over_g])
            end
        end
        sd = second_differences(m, df, :x, :w; groups = :g, levels = [-1.0, 0.0, 1.0])
        r = population_margins(m, df; vars = [:x], scenarios = (w = [-1.0, 0.0, 1.0],), groups = :g)
        @test _frames_equal(sd, second_differences(r, :x, :w; modifier_type = :continuous), [_SD_COLS; :over_g])
        @test nrow(sd) == 3 * 3
        # an `at` NamedTuple: the modifier point and held variables, with groups on another variable
        sd = second_differences(m, df, :x, :w; groups = :g, at = (w = 0.3, z = [1.0, 2.0]), delta = δ)
        @test nrow(sd) == 6 && :z in propertynames(sd) && :over_g in propertynames(sd)
    end

    @testset "profile: counterfactual values within each stratum" begin
        sd = second_differences(m, df, :x, :t; groups = :g, profile = (k = ["p", "q"],))
        @test nrow(sd) == 6
        @test names(sd)[end-1:end] == ["over_g", "k"]   # over_ columns, then the profile columns
        for kv in ("p", "q")
            r = population_margins(m, df; vars = [:x], scenarios = (t = [0, 1], k = [kv]), groups = :g)
            @test _frames_equal(sd[sd.k .== kv, :], second_differences(r, :x, :t), [_SD_COLS; :over_g])
        end
        # local derivative with profile and groups
        sd = second_differences(m, df, :x, :w; groups = :k, profile = (t = [0, 1],), at = 0.0, delta = 0.05)
        @test nrow(sd) == 4 && Set(zip(sd.over_k, sd.t)) == Set([("p", 0), ("q", 0), ("p", 1), ("q", 1)])
    end

    @testset "p_adjust covers every stratum; weights; Tables source" begin
        sd = second_differences(m, df, [:x, :w], :t; groups = :g, p_adjust = :bonferroni)
        @test sd.p_adjusted ≈ min.(1, nrow(sd) .* sd.p_value)
        sw = second_differences(m, df, :x, :t; groups = :g, weights = :wt)
        rw = population_margins(m, df; vars = [:x], scenarios = (t = [0, 1],), groups = :g, weights = :wt)
        @test _frames_equal(sw, second_differences(rw, :x, :t), [_SD_COLS; :over_g])
        @test _frames_equal(second_differences(m, Tables.columntable(df), :x, :t; groups = :g),
                            second_differences(m, df, :x, :t; groups = :g), [_SD_COLS; :over_g])
    end

    @testset "errors: a variable the calculation sets cannot define strata" begin
        err = try second_differences(m, df, :x, :t; groups = :t) catch e; e end
        @test err isa ArgumentError
        @test occursin("result-level form", err.msg) && occursin("groups=:t", err.msg)
        @test_throws ArgumentError second_differences(m, df, :x, :w; groups = (:w, 3))            # binned modifier
        @test_throws ArgumentError second_differences(m, df, :x, :w; groups = [:g, (:w, 2)], at = :mean)
        @test_throws ArgumentError second_differences(m, df, :x, :t; groups = :k, profile = (k = ["p"],))
        @test_throws ArgumentError second_differences(m, df, :x, :w; groups = :k, at = (w = 0.0, k = "p"))
        @test_throws ArgumentError second_differences(m, df, :k, :t; groups = :k)                 # focal pinned
        @test_throws ArgumentError second_differences(m, df, :x, :t; groups = :nope)
        # a binned focal variable is allowed (it varies within each bin), as in population_margins
        @test nrow(second_differences(m, df, :x, :t; groups = (:x, 2))) == 2
        # the result-level form still rejects a result that varies over a second scenario variable
        r = population_margins(m, df; vars = [:x], scenarios = (t = [0, 1], k = ["p", "q"]))
        @test_throws ArgumentError second_differences(r, :x, :t)
    end

    @testset "subgroup vs counterfactual: the result-level route with groups = modifier" begin
        # the across-subgroup comparison the modifier-in-groups error points to: the AME of x among
        # the rows with k = p vs among the rows with k = q
        r = population_margins(m, df; vars = [:x], groups = :k)
        sub = second_differences(r, :x, :k)
        @test nrow(sub) == 1
        lv1, lv2 = sub.modifier_level1[1], sub.modifier_level2[1]
        ame(lv) = DataFrame(population_margins(m, df[df.k .== lv, :]; vars = [:x])).estimate[1]
        @test sub.ame_at_level1[1] ≈ ame(lv1) rtol = 1e-12
        @test sub.ame_at_level2[1] ≈ ame(lv2) rtol = 1e-12
        # the counterfactual one sets every row to p, then to q
        cf = second_differences(m, df, :x, :k)
        @test !(abs(sub.second_diff[1]) ≈ abs(cf.second_diff[1]))   # different estimands
    end

    @testset "backend=:ad and :fd agree on second differences" begin
        for (vars, mod, kw) in (([:x, :z], :w, (;)), ([:x, :z], :w, (; groups = :g)),
                                ([:x, :z], :w, (; at = :mean)), ([:x, :z], :w, (; groups = :k, at = 0.5)),
                                ([:x, :w, :g], :t, (;)), ([:x, :w], :t, (; groups = [:g, :k])))
            a = second_differences(m, df, vars, mod; kw..., backend = :ad, delta = 0.05, warn = false)
            f = second_differences(m, df, vars, mod; kw..., backend = :fd, delta = 0.05, warn = false)
            @test nrow(a) == nrow(f) > 0
            @test isapprox(a.second_diff, f.second_diff; rtol = 1e-6, atol = 1e-9)
            @test isapprox(a.se, f.se; rtol = 1e-6, atol = 1e-9)
        end
        a = second_differences(m, df, :x, :t; groups = :g, backend = :ad)
        f = second_differences(m, df, :x, :t; groups = :g, backend = :fd)
        @test isapprox(a.second_diff, f.second_diff; rtol = 1e-6) && isapprox(a.se, f.se; rtol = 1e-6)
        @test_throws ArgumentError second_differences(m, df, :x, :t; backend = :nope)
    end
end
