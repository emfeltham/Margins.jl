# test_second_differences_groups.jl
# Margins 3.0, step 8: the model-level second differences on the request layer.
#
# - `groups=` on second_differences(model, data, …): the counterfactual second difference within
#   each stratum. Checked (1) exactly against the result-level route (population_margins with the
#   same scenarios and groups, then second_differences(result, …)), and (2) against an oracle: the
#   ungrouped call on that stratum's rows only, with the same Σ.
# - backend=:ad vs :fd agree on second differences.
# - the FixedEffectModels estimation sample is used by at=:mean (the removed internal
#   _second_differences_local did not subset to it).
# - a categorical variable that appears only inside an interaction (no main effect), as a focal
#   variable of population_margins and of second_differences and as a modifier, against hand
#   computations from modelcols; the same pattern for a Bool and a continuous variable.

using Test
using Margins
using DataFrames, CategoricalArrays, GLM, StatsModels, Statistics, Random, LinearAlgebra
using FixedEffectModels

# The schema columns two second-difference frames share, compared exactly (NaN == NaN)
const _SD_COLS = [:variable, :contrast, :modifier, :modifier_type, :modifier_level1, :modifier_level2,
                  :ame_at_level1, :ame_at_level2, :second_diff, :se, :statistic, :df, :p_value,
                  :ci_lower, :ci_upper, :significant]
_same(a, b) = isequal(a, b)
function _frames_equal(a::DataFrame, b::DataFrame, cols)
    nrow(a) == nrow(b) || return false
    return all(c -> all(_same.(a[!, c], b[!, c])), cols)
end

@testset "second differences: groups=, backend=, interaction-only variables" begin
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

    @testset "bad alpha is an ArgumentError on both forms" begin
        for a in (0.0, 1.0, 1.5, -0.1, "0.05")
            @test_throws ArgumentError second_differences(m, df, :x, :t; alpha = a)
            @test_throws ArgumentError second_differences(m, df, :x, :w; alpha = a)
        end
        r = population_margins(m, df; vars = [:x], scenarios = (t = [0, 1],))
        @test_throws ArgumentError second_differences(r, :x, :t; alpha = 2)
        @test second_differences(m, df, :x, :t; alpha = 1 // 10).ci_lower[1] ≈
              second_differences(m, df, :x, :t; alpha = 0.1).ci_lower[1]
    end
end

@testset "second differences on FixedEffectModels: the estimation sample (at=:mean)" begin
    Random.seed!(77)
    n = 400
    d = DataFrame(x = randn(n), z = randn(n), h = rand(["p", "q", "r", "s"], n))
    d.y = d.x .+ 0.5 .* d.x .* d.z .+ 0.3 .* d.z .+ randn(n)
    d.h[1] = "singleton"
    d.z[1] = 50.0                     # a dropped row far from the others: it would move the mean
    fm = reg(d, @formula(y ~ x * z + fe(h)))
    @test sum(fm.esample) == n - 1
    sd = second_differences(fm, d, :x, :z; at = :mean, weights = :none)
    @test sd.eval_point[1] == mean(d.z[fm.esample])
    @test sd.eval_point[1] != mean(d.z)
    # identity link, linear in z: the derivative is the interaction coefficient
    @test sd.second_diff[1] ≈ coef(fm)[findfirst(==("x & z"), coefnames(fm))] rtol = 1e-6
    so = second_differences(fm, d, :x, :z; weights = :none)
    @test so.second_diff[1] ≈ sd.second_diff[1] rtol = 1e-6
end

# Hand computations for a categorical variable: average over rows of the prediction with every row
# set to `lv` minus every row set to `base` (other variables as observed, and `set` applied), with
# the delta-method SE from the modelcols gradients
function _hand_contrast(m, d, var, lv, base; set = (;), link = :identity)
    f = formula(m).rhs
    β = coef(m)
    function X(level)
        dd = copy(d)
        dd[!, var] = categorical(fill(level, nrow(dd)); levels = levels(d[!, var]))
        for (k, v) in pairs(set)
            dd[!, k] = fill(v, nrow(dd))
        end
        modelcols(f, dd)
    end
    Xl, Xb = X(lv), X(base)
    if link === :identity
        return mean(Xl * β .- Xb * β), vec(mean(Xl .- Xb; dims = 1))
    end
    μ(η) = 1 / (1 + exp(-η))
    dμ(η) = μ(η) * (1 - μ(η))
    el, eb = Xl * β, Xb * β
    return mean(μ.(el) .- μ.(eb)), vec(mean(dμ.(el) .* Xl .- dμ.(eb) .* Xb; dims = 1))
end

@testset "a variable that appears only inside an interaction" begin
    Random.seed!(4242)
    n = 600
    d = DataFrame(x = randn(n), w = 1 .+ rand(n), t = rand(0:1, n), b = rand(Bool, n),
                  g = categorical(rand(["a", "b", "c"], n)))
    d.y = d.x .+ 0.5 .* (d.g .== "b") .* d.w .- 0.4 .* (d.g .== "c") .* d.x .* d.w .+ 0.3 .* d.t .+ randn(n)
    ηb = 0.2 .+ d.x .+ 0.8 .* (d.g .== "b") .* d.w .- 0.5 .* (d.g .== "c") .* d.t .* d.w
    d.yb = rand(n) .< 1 ./ (1 .+ exp.(-ηb))

    @testset "AME of g (no main effect) against the hand computation: $(fm)" for (fm, link) in (
            (@formula(y ~ x + g & w), :identity),
            (@formula(y ~ x + g & x & w), :identity),
            (@formula(y ~ x + g & log(w)), :identity),
            (@formula(yb ~ x + g & w + g & t & w), :logit))
        m = link === :identity ? lm(fm, d) : glm(fm, d, Binomial(), LogitLink())
        r = DataFrame(population_margins(m, d; vars = [:g]))
        @test r.contrast == ["b - a", "c - a"]          # full-rank coding: the first level is the reference
        for row in eachrow(r)
            est, grad = _hand_contrast(m, d, :g, split(row.contrast, " - ")[1], "a"; link)
            @test row.estimate ≈ est atol = 1e-12
            @test row.se ≈ sqrt(dot(grad, vcov(m), grad)) atol = 1e-12
        end
        # auto-detected variables include it too
        @test "g" in DataFrame(population_margins(m, d)).variable
        # pairwise contrasts
        @test nrow(DataFrame(population_margins(m, d; vars = [:g], contrasts = :pairwise))) == 3
        # profile_margins (row-specific baseline at a grid row)
        # (a grid row that sets g gives the row-specific contrast of that level against the reference)
        @test DataFrame(profile_margins(m, d, cartesian_grid(g = ["b"]); type = :effects, vars = [:g])).contrast == ["b - a"]
    end

    @testset "full-rank main effect (y ~ 0 + g + x): contrasts are coefficient differences" begin
        m0 = lm(@formula(y ~ 0 + g + x), d)
        r = DataFrame(population_margins(m0, d; vars = [:g]))
        cn = coefnames(m0)
        @test r.estimate[1] ≈ coef(m0)[findfirst(==("g: b"), cn)] - coef(m0)[findfirst(==("g: a"), cn)] atol = 1e-12
    end

    m = glm(@formula(yb ~ x + g & w + g & t & w), d, Binomial(), LogitLink())
    @testset "as a focal variable of second differences" begin
        sd = second_differences(m, d, :g, :t)
        @test sd.contrast == ["b - a", "c - a"]
        for row in eachrow(sd)
            lv = split(row.contrast, " - ")[1]
            e0, g0 = _hand_contrast(m, d, :g, lv, "a"; set = (t = 0,), link = :logit)
            e1, g1 = _hand_contrast(m, d, :g, lv, "a"; set = (t = 1,), link = :logit)
            @test row.second_diff ≈ e1 - e0 atol = 1e-12
            @test row.se ≈ sqrt(dot(g1 .- g0, vcov(m), g1 .- g0)) atol = 1e-12
        end
        @test nrow(second_differences(m, d, :g, :w)) == 2          # continuous modifier, at=:observed
        @test nrow(second_differences(m, d, :g, :t; groups = :b)) == 4
    end

    @testset "as a modifier" begin
        sd = second_differences(m, d, :x, :g; warn = false)
        @test nrow(sd) == 3                                         # pairs of a, b, c
        r = population_margins(m, d; vars = [:x], scenarios = (g = ["a", "b", "c"],))
        @test _frames_equal(sd, second_differences(r, :x, :g; warn = false), _SD_COLS)
        sd = second_differences(m, d, :w, :g; warn = false)
        @test nrow(sd) == 3 && all(isfinite, sd.se)
    end

    @testset "Bool and continuous variables only inside an interaction" begin
        mb = glm(@formula(yb ~ x + b & w + x & w), d, Binomial(), LogitLink())
        rb = DataFrame(population_margins(mb, d; vars = [:b, :w]))
        @test Set(rb.variable) == Set(["b", "w"])
        # Bool: the true - false contrast, by hand
        f = formula(mb).rhs
        Xt = modelcols(f, transform(d, :b => ByRow(_ -> true) => :b))
        Xf = modelcols(f, transform(d, :b => ByRow(_ -> false) => :b))
        μ(η) = 1 / (1 + exp(-η)); dμ(η) = μ(η) * (1 - μ(η))
        β = coef(mb)
        est = mean(μ.(Xt * β) .- μ.(Xf * β))
        grad = vec(mean(dμ.(Xt * β) .* Xt .- dμ.(Xf * β) .* Xf; dims = 1))
        rowb = rb[rb.variable .== "b", :]
        @test rowb.estimate[1] ≈ est atol = 1e-12
        @test rowb.se[1] ≈ sqrt(dot(grad, vcov(mb), grad)) atol = 1e-12
        # second differences: Bool focal and Bool modifier; continuous w only in interactions
        @test nrow(second_differences(mb, d, :b, :x; warn = false)) == 1
        @test nrow(second_differences(mb, d, :x, :b)) == 1
        @test nrow(second_differences(mb, d, :w, :x)) == 1
    end
end
