# test/regression/test_hierarchical_partial_typical.jl
#
# Regression test: `hierarchical_grid` columns that only some combinations specify.
#
# `:g => [(:k, [1, 3]), (:z, :mean)]` builds rows where k is specified (and z typical) and
# rows where z is specified (and k typical). The typical cells of such a column were not
# marked, so for a variable the MODEL treats as categorical (an Int column fitted with
# `DummyCoding`) `complete_reference_grid` rejected the typical value (the mean code) as "not
# a level" instead of using the frequency mixture. Typical cells are now recorded per row;
# only those become the mixture and user-specified cells are still validated. And because
# FormulaCompiler compiles a column either as discrete or as one mixture, `profile_margins`
# now evaluates grids whose categorical columns mix levels and mixtures in homogeneous row
# blocks (this also affected true CategoricalArray columns).
#
# Oracle: design rows from `modelcols` on the model's schema; a categorical mixture is the
# frequency-weighted average of the design rows at each level; derivatives by central
# exact for the design rows (linear in x); SEs by the delta method with the analytic gradient.

using Test
using DataFrames, GLM, StatsModels, Statistics, LinearAlgebra, Random, Tables, CategoricalArrays
using Margins

@testset "hierarchical_grid: partly typical columns" begin
    Random.seed!(17); n = 1200
    d = DataFrame(x = randn(n), k = rand(1:3, n), z = randn(n), g = rand(["a", "b"], n),
                  h = categorical(rand(["p", "q", "r"], n)))
    d.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.4 .* d.x .+ [0.0, 0.7, -0.5][d.k] .+ 0.3 .* d.z .+
                                      0.2 .* (d.g .== "b") .- 0.4 .* (d.h .== "q"))))
    m = glm(@formula(y ~ x * k + z + g + h), d, Binomial(), LogitLink();
            contrasts = Dict(:k => DummyCoding()))
    β = coef(m); V = vcov(m); L = LogitLink(); rhs = formula(m).rhs
    X(dd) = vec(modelcols(rhs, Tables.columntable(dd)))
    kfreq = [mean(d.k .== l) for l in 1:3]
    hlev = ["p", "q", "r"]; hfreq = [mean(d.h .== l) for l in hlev]
    gfreq = Dict(l => mean(d.g .== l) for l in ["a", "b"])

    # Design row at a grid row: Dict var => value, where a value may be :kmix / :hmix / :gmix
    function xrow(vals)
        ks = vals[:k] === :kmix ? collect(zip(1:3, kfreq)) : [(Int(vals[:k]), 1.0)]
        hs = vals[:h] === :hmix ? collect(zip(hlev, hfreq)) : [(string(vals[:h]), 1.0)]
        gs = vals[:g] === :gmix ? collect(gfreq) : [(string(vals[:g]), 1.0)]
        sum(wk * wh * wg * X(DataFrame(x=[vals[:x]], k=[kv], z=[vals[:z]], g=[gv],
                                       h=categorical([hv]; levels=hlev), y=[false]))
            for (kv, wk) in ks, (hv, wh) in hs, (gv, wg) in gs)
    end
    pred(vals) = (xr = xrow(vals); η = dot(xr, β); μ = GLM.linkinv(L, η);
                  (μ, sqrt(dot(xr, V, xr)) * GLM.mueta(L, η)))
    # The design row is linear in x (x and x & k terms), so D = ∂x/∂x-row is exact by a
    # central difference; dμ/dx = μ'(η) D'β and its β-gradient is μ''(η)(D'β) x + μ'(η) D
    # (logit: μ' = μ(1-μ), μ'' = μ(1-μ)(1-2μ)).
    function dydx(vals)
        xr = xrow(vals)
        D = (xrow(merge(vals, Dict(:x => vals[:x] + 1.0))) - xrow(merge(vals, Dict(:x => vals[:x] - 1.0)))) / 2
        μ = GLM.linkinv(L, dot(xr, β)); d1 = μ * (1 - μ); d2 = d1 * (1 - 2μ)
        grad = d2 * dot(D, β) .* xr .+ d1 .* D
        (d1 * dot(D, β), sqrt(dot(grad, V, grad)))
    end
    x̄ = mean(d.x); z̄ = mean(d.z)

    @testset "Int column the model treats as categorical" begin
        grid = hierarchical_grid(d, :g => [(:k, [1, 3]), (:z, :mean)])
        @test nrow(grid) == 6
        # Typical cells are recorded per row, and only for the partly specified columns
        tk = Margins._typical_value_rows(grid, "k")
        @test tk == BitVector([0, 0, 1, 0, 0, 1])
        @test Margins._typical_value_rows(grid, "g") === nothing        # always specified
        @test Margins._typical_value_rows(grid, "x") == trues(6)        # never specified

        # Expected value of each grid row
        expect = map(eachrow(grid)) do r
            Dict(:x => x̄, :z => r.z, :g => r.g, :h => :hmix,
                 :k => (r.k == 1.0 || r.k == 3.0) && !ismissing(r.k) && !tk[rownumber(r)] ? r.k : :kmix)
        end
        res = DataFrame(profile_margins(m, d, grid; type=:predictions))
        @test nrow(res) == 6
        for (i, r) in enumerate(eachrow(res))
            (μ, se) = pred(expect[i])
            @test r.estimate ≈ μ rtol=1e-10
            @test r.se ≈ se rtol=1e-8
            @test r.g == grid.g[i]                     # grid-row order preserved
        end

        eff = DataFrame(profile_margins(m, d, grid; type=:effects, vars=[:x]))
        @test nrow(eff) == 6
        for (i, r) in enumerate(eachrow(eff))
            (est, se) = dydx(expect[i])
            @test r.estimate ≈ est rtol=1e-6
            @test r.se ≈ se rtol=1e-6
        end
    end

    @testset "CategoricalArray column (mixed levels and mixtures)" begin
        grid = hierarchical_grid(d, :g => [:h, (:x, [0.0, 1.0])])
        th = Margins._typical_value_rows(grid, "h")
        @test count(th) == 4 && count(.!th) == 6
        res = DataFrame(profile_margins(m, d, grid; type=:predictions))
        @test nrow(res) == nrow(grid)
        for (i, r) in enumerate(eachrow(res))
            vals = Dict(:x => grid.x[i], :z => z̄, :g => grid.g[i], :k => :kmix,
                        :h => th[i] ? :hmix : string(grid.h[i]))
            (μ, se) = pred(vals)
            @test r.estimate ≈ μ rtol=1e-10
            @test r.se ≈ se rtol=1e-8
        end
    end

    @testset "user-specified cells are still validated" begin
        # 9 is not a level of k: an error naming only the specified non-level value
        grid = hierarchical_grid(d, :g => [(:k, [1, 9]), (:z, :mean)])
        err = try profile_margins(m, d, grid; type=:predictions); nothing catch e; e end
        @test err isa Margins.MarginsError
        msg = sprint(showerror, err)
        @test occursin("[9.0]", msg) && occursin(":k", msg)
        @test !occursin(string(round(mean(d.k), digits=2)), msg)

        # Row-moving edits drop the per-row record (DataFrames drops :default-style
        # metadata), so the edited grid is treated as user-specified: an error, never a
        # mixture placed in the wrong rows
        grid = hierarchical_grid(d, :g => [(:k, [1, 3]), (:z, :mean)])
        sorted = sort(grid, :z)
        @test Margins._typical_value_rows(sorted, "k") === nothing
        @test_throws Margins.MarginsError profile_margins(m, d, sorted; type=:predictions)
        # ...while copies keep it
        @test Margins._typical_value_rows(copy(grid), "k") == Margins._typical_value_rows(grid, "k")
    end

    @testset "whole-column typical value equal to a level is still the mixture" begin
        # Balanced codes 1, 2, 3: the mean code is exactly 2.0, a level. It is a typical
        # value, not a request for level 2, so means_grid evaluates at the mixture.
        d2 = copy(d); d2.k = repeat(1:3, n ÷ 3)
        m2 = glm(@formula(y ~ x * k + z + g + h), d2, Binomial(), LogitLink();
                 contrasts = Dict(:k => DummyCoding()))
        mg = means_grid(d2)
        @test only(mg.k) == 2.0
        r = only(eachrow(DataFrame(profile_margins(m2, d2, mg; type=:predictions))))
        rhs2 = formula(m2).rhs; β2 = coef(m2)
        X2(dd) = vec(modelcols(rhs2, Tables.columntable(dd)))
        xm = sum((1/3) * wg * wh * X2(DataFrame(x=[mean(d2.x)], k=[kv], z=[mean(d2.z)], g=[gv],
                                                 h=categorical([hv]; levels=hlev), y=[false]))
                 for kv in 1:3, (gv, wg) in gfreq, (hv, wh) in zip(hlev, hfreq))
        @test r.estimate ≈ GLM.linkinv(L, dot(xm, β2)) rtol=1e-10
    end
end
