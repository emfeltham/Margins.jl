# test/validation/test_oracle_3_0.jl
#
# R oracles for paths added or changed in Margins 3.0 (release plan step 11). Reference values are
# fixtures committed under test/r_compare_marginaleffects/oracle_3_0/ (regenerate with the R
# scripts there; R is not needed to run this file):
#
#   observed_sd_reference.csv  second differences with the default `at = :observed`
#   weights_reference.csv      fits with estimation weights (integer and non-integer): AME, AAP, MEM, groups
#   offsets_reference.csv      Poisson with an exposure offset: AAP, AME, MEM and APM with the offset given
#   fixest_reference.csv       FixedEffectModels against fixest: predictions including the fixed
#                              effects, AME, by state, three covariance estimators, an esample-dropping fit
#
# R 4.6.1, marginaleffects 1.0.0, fixest 0.14.2. The README there lists the cases and the observed
# differences. Every tolerance below is a few times the largest difference observed, and the
# comment beside it says what limits it (R's numerical derivatives, never Margins').

using Test
using DataFrames, CategoricalArrays, GLM, CSV, Statistics
using FixedEffectModels
using Margins

include(joinpath(@__DIR__, "oracle_3_0_data.jl"))

const _ORACLE3_DIR = joinpath(@__DIR__, "..", "r_compare_marginaleffects", "oracle_3_0")

function _oracle3_csv(name)
    ref = CSV.read(joinpath(_ORACLE3_DIR, name), DataFrame; missingstring = "NA")
    for c in propertynames(ref)
        eltype(ref[!, c]) <: Union{Missing,AbstractString} && (ref[!, c] = String.(coalesce.(ref[!, c], "")))
    end
    return ref
end

# Tight IRLS convergence, matching `glm.control(epsilon = 1e-14)` in the R scripts: at the default
# tolerances GLM.jl and R stop at fits up to 1e-5 apart (probit), which would hide everything else
const _TIGHT = (rtol = 1e-15, atol = 0.0)

# R's label of a continuous effect
_ocontrast(c) = c == "dY/dX" ? "dy/dx" : c

@testset "R oracles (Margins 3.0)" begin
    d = _oracle3_data()

    @testset "the dataset is the one R built" begin
        ck = _oracle3_csv("data_checksums.csv")
        for r in eachrow(ck)
            if r.column == "g_b"
                @test count(==("b"), d.g) == r.sum
            else
                v = d[!, Symbol(r.column)]
                @test sum(v) ≈ r.sum rtol = 1e-13
                @test sum(abs2, v) ≈ r.sumsq rtol = 1e-13
            end
        end
    end

    @testset "at = :observed second differences vs R" begin
        # R: both shifted copies of the data in one newdata, avg_slopes(..., by = shift, hypothesis =
        # "(b2 - b1) / (2δ) = 0"), so the SE uses the joint Jacobian of the two AMEs
        models = Dict(
            "lm" => lm(@formula(yc ~ x * w + z + g * w), d),
            "logit" => glm(@formula(yb ~ x * w + z + g * w), d, Binomial(), LogitLink(); _TIGHT...),
            "probit" => glm(@formula(yb ~ x * w + z + g * w), d, Binomial(), ProbitLink(); _TIGHT...),
            "logit_add" => glm(@formula(yb ~ x + w + z + g), d, Binomial(), LogitLink(); _TIGHT...),
            "wlogit" => glm(@formula(yb ~ x * w + z + g * w), d, Binomial(), LogitLink(); wts = d.sw, _TIGHT...),
        )
        ref = _oracle3_csv("observed_sd_reference.csv")
        @test nrow(ref) == 32
        measure = Dict("dydx" => :effect, "eyex" => :elasticity)
        worst_est, worst_se = 0.0, 0.0
        for r in eachrow(ref)
            m = models[r.model]
            kw = (scale = Symbol(r.type), measure = measure[r.measure], warn = false)
            # the default `at = :observed` and `delta = :auto`; the wlogit uses the fit's weights
            # (`weights = :model`), as R's `wts = "sw"`
            sd = r.stratum == "all" ? second_differences(m, d, Symbol(r.focal), :w; kw...) :
                 second_differences(m, d, Symbol(r.focal), :w; groups = :g, kw...)
            sd = sd[sd.contrast .== r.contrast, :]
            r.stratum == "all" || (sd = sd[string.(sd.over_g) .== r.stratum, :])
            @test nrow(sd) == 1
            label = "$(r.model) $(r.focal) $(r.contrast) $(r.measure) $(r.type) $(r.stratum)"
            # δ = 0.01·sd(w) in both
            @test sd.delta_used[1] ≈ r.delta rtol = 1e-14
            # R's slope is a central difference with eps = 1e-4 divided by 2δ ≈ 0.0098 (observed ≤ 1.5e-9)
            @test isapprox(sd.second_diff[1], r.estimate; rtol = 1e-7, atol = 1e-12) ||
                  (println("EST ", label, " ", sd.second_diff[1], " vs ", r.estimate); false)
            # R's numerical Jacobian of a numerical slope: ≤ 2.3e-9 for effects and contrasts, 1.5e-5
            # for the elasticity (which also divides by ŷ inside the nested difference)
            tol = r.measure == "eyex" ? 1e-4 : 1e-6
            @test isapprox(sd.se[1], r.std_error; rtol = tol) ||
                  (println("SE  ", label, " ", sd.se[1], " vs ", r.std_error); false)
            worst_est = max(worst_est, abs(sd.second_diff[1] - r.estimate) / abs(r.estimate))
            worst_se = max(worst_se, abs(sd.se[1] - r.std_error) / r.std_error)
        end
        println("at=:observed vs R: max relative difference estimate ", worst_est, ", se ", worst_se)

        # Exact references, independent of R. In the linear model the cross-derivative of y ~ x*w is
        # the x:w coefficient at every row; on the logit link scale too.
        for (mn, term) in (("lm", "x & w"), ("logit", "x & w"), ("probit", "x & w"))
            m = models[mn]
            k = findfirst(==(term), coefnames(m))
            sd = second_differences(m, d, :x, :w; scale = :link, warn = false)
            @test sd.second_diff[1] ≈ coef(m)[k] rtol = 1e-9
            @test sd.se[1] ≈ sqrt(GLM.vcov(m)[k, k]) rtol = 1e-9
        end
    end

    @testset "model weights vs R (frequency vs precision convention)" begin
        fits = Dict{String,Any}()
        for wv in ("fw", "sw")
            wt = d[!, Symbol(wv)]
            fits["lm_$wv"] = lm(@formula(yc ~ x * z + g), d; wts = wt)
            fits["gauss_$wv"] = glm(@formula(yc ~ x * z + g), d, Normal(), IdentityLink(); wts = wt, _TIGHT...)
            fits["logit_$wv"] = glm(@formula(yb ~ x * z + g), d, Binomial(), LogitLink(); wts = wt, _TIGHT...)
            fits["poisson_$wv"] = glm(@formula(cnt ~ x * z + g), d, Poisson(), LogLink(); wts = wt, _TIGHT...)
        end
        ref = _oracle3_csv("weights_reference.csv")
        @test nrow(ref) == 216
        wmean(v, w) = sum(w .* v) / sum(w)
        function julia_value(r)
            m = fits[r.model]
            wv = Symbol(split(r.model, "_")[2])
            sc = Symbol(r.type)
            # weights = :model (the default) everywhere: the fit's own weights
            res = if r.quantity == "ame"
                r.by == "all" ? DataFrame(population_margins(m, d; scale = sc)) :
                    DataFrame(population_margins(m, d; vars = [:x, :z], groups = :g, scale = sc))
            elseif r.quantity == "aap"
                r.by == "all" ? DataFrame(population_margins(m, d; type = :predictions, scale = sc)) :
                    DataFrame(population_margins(m, d; type = :predictions, groups = :g, scale = sc))
            else
                x0, z0 = r.quantity == "mem_fixed" ? (1.5, 0.25) : (wmean(d.x, d[!, wv]), wmean(d.z, d[!, wv]))
                DataFrame(profile_margins(m, d, cartesian_grid(x = [x0], z = [z0], g = ["b"]);
                                          vars = [:x, :z], scale = sc))
            end
            r.by == "all" || (res = res[string.(res.over_g) .== r.by, :])
            r.quantity == "aap" || (res = res[(res.variable .== r.term) .& (res.contrast .== _ocontrast(r.contrast)), :])
            @test nrow(res) == 1
            return res.estimate[1], res.se[1]
        end
        worst_est, worst_se = 0.0, 0.0
        for r in eachrow(ref)
            est, se = julia_value(r)
            m = fits[r.model]
            label = "$(r.model) $(r.quantity) $(r.term) $(r.contrast) $(r.by) $(r.type)"
            # R's numerical slope (eps = 1e-4) and Jacobian: observed ≤ 2.2e-10
            @test isapprox(est, r.estimate; rtol = 1e-8, atol = 1e-12) ||
                  (println("EST ", label, " ", est, " vs ", r.estimate); false)
            worst_est = max(worst_est, abs(est - r.estimate) / abs(r.estimate))
            if startswith(r.model, "lm") || startswith(r.model, "gauss")
                # Estimated dispersion: GLM.jl's frequency weights give σ̂² = Σwr²/(Σw − p), R's
                # precision weights Σwr²/(n − p), so every SE differs by sqrt((n − p)/(Σw − p)):
                # 0.62959 for fw (Σw = 1000), 0.81361 for sw (Σw = 601.2), n = 400, p = 6
                n, p, W = nrow(d), length(coef(m)), sum(d[!, Symbol(split(r.model, "_")[2])])
                predicted = sqrt((n - p) / (W - p))
                @test isapprox(se / r.std_error, predicted; rtol = 1e-8) ||
                      (println("SE ratio ", label, " ", se / r.std_error, " vs ", predicted); false)
                worst_se = max(worst_se, abs(se / r.std_error / predicted - 1))
            else
                # Fixed dispersion (logit, Poisson): the conventions coincide. Observed ≤ 8.4e-10
                @test isapprox(se, r.std_error; rtol = 1e-7) ||
                      (println("SE  ", label, " ", se, " vs ", r.std_error); false)
                worst_se = max(worst_se, abs(se - r.std_error) / r.std_error)
            end
        end
        println("weights vs R: max relative difference estimate ", worst_est,
                ", se (after the Gaussian convention ratio) ", worst_se)
        # the Gaussian fits: residual df Σw − p (frequency weights), also with non-integer weights
        @test dof_residual(fits["lm_fw"]) == 1000 - 6
        @test dof_residual(fits["lm_sw"]) ≈ sum(d.sw) - 6
    end

    @testset "offsets vs R" begin
        fits = Dict(
            "poisson_off" => glm(@formula(cnt ~ x * z + g), d, Poisson(), LogLink(); offset = log.(d.expo), _TIGHT...),
            "poisson_off_w" => glm(@formula(cnt ~ x * z + g), d, Poisson(), LogLink(); offset = log.(d.expo),
                                   wts = d.sw, _TIGHT...),
        )
        ref = _oracle3_csv("offsets_reference.csv")
        @test nrow(ref) == 156
        function julia_value(r)
            m = fits[r.model]
            sc = Symbol(r.type)
            res = if r.quantity in ("aap", "ame")
                tp = r.quantity == "aap" ? :predictions : :effects
                gk = r.by == "all" ? (;) : (groups = :g,)
                vk = r.quantity == "ame" && r.by != "all" ? (vars = [:x, :z],) : (;)
                out = DataFrame(population_margins(m, d; type = tp, scale = sc, gk..., vk...))
                r.by == "all" ? out : out[string.(out.over_g) .== r.by, :]
            else
                # The offset at the grid row is log(exposure), given with `offset=` (R: `expo` in the
                # datagrid). A categorical pinned in the grid gets that row's contrast in Margins;
                # R gives every contrast at the row, so R's "c - a" at g = "b" is Margins' row at g = "c"
                gl = r.term == "g" ? string(r.contrast[1]) : "b"
                grid = DataFrame(x = [r.grid_x], z = [0.25], g = [gl])
                tp = r.quantity == "apm" ? :predictions : :effects
                vk = tp === :effects ? (vars = [:x, :z, :g],) : (;)
                DataFrame(profile_margins(m, d, grid; type = tp, scale = sc, offset = log(r.grid_expo), vk...))
            end
            r.quantity in ("ame", "mem") &&
                (res = res[(res.variable .== r.term) .& (res.contrast .== _ocontrast(r.contrast)), :])
            @test nrow(res) == 1
            return res.estimate[1], res.se[1]
        end
        worst_est, worst_se = 0.0, 0.0
        for r in eachrow(ref)
            est, se = julia_value(r)
            label = "$(r.model) $(r.quantity) $(r.term) $(r.contrast) $(r.by) $(r.type) $(r.grid_x) $(r.grid_expo)"
            # observed ≤ 3.1e-11
            @test isapprox(est, r.estimate; rtol = 1e-8, atol = 1e-12) ||
                  (println("EST ", label, " ", est, " vs ", r.estimate); false)
            # R's numerical Jacobian of a numerical slope at one row (MEM): observed ≤ 2.6e-5; AME
            # (averaged) ≤ 1.2e-6; predictions (no slope) ≤ 1.4e-9
            tol = r.quantity == "mem" ? 1e-4 : r.quantity == "ame" ? 1e-5 : 1e-7
            @test isapprox(se, r.std_error; rtol = tol) ||
                  (println("SE  ", label, " ", se, " vs ", r.std_error); false)
            worst_est = max(worst_est, abs(est - r.estimate) / abs(r.estimate))
            worst_se = max(worst_se, abs(se - r.std_error) / r.std_error)
        end
        println("offsets vs R: max relative difference estimate ", worst_est, ", se ", worst_se)

        m = fits["poisson_off"]
        # log link with an intercept: the AAP is mean(y) (and mean(predict)); R's is too (fixture)
        aap = DataFrame(population_margins(m, d; type = :predictions)).estimate[1]
        @test aap ≈ mean(d.cnt) rtol = 1e-12
        @test aap ≈ mean(predict(m)) rtol = 1e-12
        rref = ref[(ref.model .== "poisson_off") .& (ref.quantity .== "aap") .& (ref.by .== "all") .&
                   (ref.type .== "response"), :estimate]
        @test only(rref) ≈ mean(d.cnt) rtol = 1e-12
        # a grid `offset` column is the `offset=` keyword; without either the call is an error
        grid = DataFrame(x = [1.0, 1.0, 2.0, 2.0], z = 0.25, g = "b")
        off = log.([1.0, 5.0, 1.0, 5.0])
        a = DataFrame(profile_margins(m, d, transform(grid, [] => (() -> off) => :offset); type = :predictions))
        b = DataFrame(profile_margins(m, d, grid; type = :predictions, offset = off))
        @test a.estimate == b.estimate && a.se == b.se
        @test_throws ArgumentError profile_margins(m, d, grid; type = :predictions)
    end

    @testset "FixedEffectModels vs fixest" begin
        # The same panel as fixest_reference.R: 12 states × 11 years (coprime: every pair occurs, a
        # connected fixed-effects graph), plus three singleton states and a row with a missing z, all
        # outside the estimation sample
        i = 1:300
        fd = DataFrame(st = "s" .* string.(mod.(i, 12) .+ 1), yr = 2000 .+ mod.(i .* 7, 11),
                       x = 0.5 .+ mod.(i .* 37, 101) ./ 40, z = Union{Missing,Float64}[cos.(i .* 0.7)...])
        fd.y = 1 .+ 0.5 .* fd.x .- 0.3 .* fd.z .+ 0.4 .* fd.x .* fd.z .+ 0.2 .* mod.(i, 12) .+
               0.1 .* mod.(i .* 7, 11) .+ sin.(i .* 2.3)
        append!(fd, DataFrame(st = ["t1", "t2", "t3", "s1"], yr = [2001, 2003, 2005, 2002],
                              x = [1.1, 2.2, 0.7, 1.4], z = [0.3, -0.2, 0.5, missing], y = [3.1, 2.4, 4.0, 2.9]))
        V = Dict("simple" => Vcov.simple(), "robust" => Vcov.robust(), "cluster" => Vcov.cluster(:st))
        # tol = 1e-12 for the fixed effects (fixest: fixef.tol = 1e-11); at the defaults they agree to ~4e-8
        fits = Dict(k => reg(fd, @formula(y ~ x * z + fe(st) + fe(yr)), v; save = :fe, tol = 1e-12) for (k, v) in V)
        @test all(sum(m.esample) == 300 for m in values(fits))
        ref = _oracle3_csv("fixest_reference.csv")
        @test nrow(ref) == 129

        # Covariance conventions. Coefficient SEs differ by a constant per estimator, from the count
        # of absorbed degrees of freedom: fixest counts K = 3 + 12 + 11 − 1 = 25 (n − K = 275),
        # FixedEffectModels dof_fes = 12 + 11 = 23 (dof_residual 274), so simple and HC1 SEs differ by
        # sqrt(275/274); clustered by state, fixest uses n − K = 286 (state FE nested in the cluster)
        # and FixedEffectModels 285, a factor sqrt(286/285). Margins' SEs must carry the model's own
        # convention: the same ratio for every quantity.
        predicted = Dict("simple" => sqrt(275 / 274), "robust" => sqrt(275 / 274), "cluster" => sqrt(286 / 285))
        @test dof_residual(fits["simple"]) == 274
        @test dof_residual(fits["cluster"]) == 11
        for (k, m) in fits
            cr = ref[(ref.vcov .== k) .& (ref.quantity .== "coef"), :]
            for r in eachrow(cr)
                j = findfirst(==(replace(r.term, ":" => " & ")), coefnames(m))
                @test coef(m)[j] ≈ r.estimate rtol = 1e-12
                @test sqrt(vcov(m)[j, j]) / r.std_error ≈ predicted[k] rtol = 1e-9
            end
        end

        function julia_value(r)
            m = fits[r.vcov]
            res = if r.quantity == "apm"
                # the grid row's prediction includes α_s3 + γ_2004, looked up from the fit
                DataFrame(profile_margins(m, fd, DataFrame(st = ["s3"], yr = [2004], x = [1.5], z = [0.25]);
                                          type = :predictions))
            else
                tp = r.quantity == "aap" ? :predictions : :effects
                gk = r.by == "all" ? (;) : (groups = :st,)
                vk = tp === :effects ? (vars = [:x, :z],) : (;)
                # weights = :none: a FixedEffectModels fit does not store its weights (it is unweighted)
                out = DataFrame(population_margins(m, fd; type = tp, weights = :none, gk..., vk...))
                r.by == "all" || (out = out[string.(out.over_st) .== r.by, :])
                tp === :effects ? out[out.variable .== r.term, :] : out
            end
            @test nrow(res) == 1
            return res.estimate[1], res.se[1]
        end
        worst_est, worst_se = 0.0, 0.0
        for r in eachrow(ref)
            r.quantity == "coef" && continue
            est, se = julia_value(r)
            label = "$(r.vcov) $(r.quantity) $(r.term) $(r.by)"
            # fixed effects solved to 1e-11/1e-12, observed ≤ 4.1e-12
            @test isapprox(est, r.estimate; rtol = 1e-9) || (println("EST ", label, " ", est, " vs ", r.estimate); false)
            # the convention ratio; R's numerical Jacobian of the slope adds ≤ 1e-6 to AME SEs
            @test isapprox(se / r.std_error, predicted[r.vcov]; rtol = 1e-5) ||
                  (println("SE ratio ", label, " ", se / r.std_error, " vs ", predicted[r.vcov]); false)
            worst_est = max(worst_est, abs(est - r.estimate) / abs(r.estimate))
            worst_se = max(worst_se, abs(se / r.std_error / predicted[r.vcov] - 1))
        end
        println("FixedEffectModels vs fixest: max relative difference estimate ", worst_est,
                ", se (after the dof ratio) ", worst_se)
        # the AAP is the mean of the fitted values including the fixed effects
        m = fits["simple"]
        fes = fe(m)
        es = m.esample
        b = coef(m)
        xb = b[1] .* fd.x[es] .+ b[2] .* fd.z[es] .+ b[3] .* fd.x[es] .* fd.z[es]
        aap = DataFrame(population_margins(m, fd; type = :predictions, weights = :none)).estimate[1]
        @test aap ≈ mean(xb .+ fes.fe_st[es] .+ fes.fe_yr[es]) rtol = 1e-12
    end
end
