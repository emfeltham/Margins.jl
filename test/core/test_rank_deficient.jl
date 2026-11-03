# test/core/test_rank_deficient.jl
#
# Rank-deficient fits (ISSUES #10). GLM.jl (`dropcollinear=true`), MixedModels and
# FixedEffectModels drop a coefficient that is collinear with the others (or with the absorbed
# fixed effects) and report it as 0 with NaN rows and columns in `vcov`; g'Σg then gave NaN for
# every quantity. Margins now treats a dropped coefficient as fixed at 0 (its rows and columns of
# Σ are zeroed, as R does), so every quantity equals the one from the refit without the redundant
# term, and a quantity that depends only on dropped coefficients has a NaN SE (not identified).
# R's marginaleffects on the same fits: test/validation/test_oracle_3_0.jl.

using Test
using DataFrames, CategoricalArrays, GLM, Statistics, LinearAlgebra, Tables
using FixedEffectModels, MixedModels
import CovarianceMatrices
using CovarianceMatrices: HC0, HC1, HC2, HC3, HC4, HC5, CR1
using Margins

include(joinpath(@__DIR__, "..", "validation", "oracle_3_0_data.jl"))

# the population AME hot loop, measured inside a function (see test_ame_fd_loop_zero_alloc.jl)
function _ame_loop_allocs(engine, vars, data_nt, rows)
    Margins._compute_continuous_effects!(engine, vars, data_nt, rows, :response, :effect, nothing, Float64(length(rows)))
    return @allocated Margins._compute_continuous_effects!(engine, vars, data_nt, rows, :response, :effect,
                                                           nothing, Float64(length(rows)))
end

# every call on a rank-deficient fit warns once, naming the dropped coefficients, and nothing else
rd(f) = @test_logs (:warn, r"rank deficient.*dropped coefficient\(s\) z") f()
rdc(f) = @test_logs (:warn, r"rank deficient.*dropped coefficient\(s\) c") f()
const TIGHT = (rtol = 1e-15, atol = 0.0)

# rows of `a` (rank-deficient fit) and `b` (refit) agree, after dropping `a`'s rows for `skip`
function same_rows(a::DataFrame, b::DataFrame; skip = "z", rtol = 1e-12)
    "variable" in names(a) && (a = a[a.variable .!= skip, :])
    @test nrow(a) == nrow(b)
    nrow(a) == nrow(b) || return false
    ok = true
    for c in intersect(names(a), names(b))
        c in ("gradient",) && continue
        x, y = a[!, c], b[!, c]
        if eltype(x) <: Union{Missing, Real} && !(eltype(x) <: Bool)
            ok &= all(isapprox.(x, y; rtol, atol = 1e-14) .| (isnan.(x) .& isnan.(y)))
        else
            ok &= isequal(x, y)
        end
    end
    @test ok
    return ok
end

@testset "Rank-deficient fits: dropped coefficients fixed at 0" begin
    # the deterministic data of the R oracles (test/validation/oracle_3_0_data.jl), 400 rows
    d = _oracle3_data()
    i = 1:nrow(d)
    # collinear: v − 2x − 3z = 0. Not v = x + z: with unit coefficients the pivoted Cholesky that
    # GLM.jl uses meets an exact tie between x and z once v is pivoted (their residuals are
    # negatives of each other), and which one it drops then depends on rounding: Julia 1.10's
    # OpenBLAS dropped x, 1.13's z. With 2 and 3 no two of x, z, v tie, and z is dropped everywhere.
    d.v = 2 .* d.x .+ 3 .* d.z
    d.c = [Dict("a" => 0.7, "b" => 1.9, "c" => -0.4)[string(l)] for l in d.g]   # constant within g
    d.cl = mod.(i, 10)

    m_lm = lm(@formula(yc ~ x * w + g + v + z), d)
    r_lm = lm(@formula(yc ~ x * w + g + v), d)
    m_lg = glm(@formula(yb ~ x * w + g + v + z), d, Binomial(), LogitLink(); TIGHT...)
    r_lg = glm(@formula(yb ~ x * w + g + v), d, Binomial(), LogitLink(); TIGHT...)
    jz = findfirst(==("z"), coefnames(m_lm))
    grid = DataFrame(x = [1.0, 2.0], w = [1.2, 1.2], z = [0.1, 0.1], v = [2.3, 4.3], g = ["b", "b"])
    rgrid = select(grid, Not(:z))

    @testset "detection: the model's own rank information" begin
        @test Margins._dropped_coefficients(m_lm) == [jz]
        @test Margins._dropped_coefficients(m_lg) == [findfirst(==("z"), coefnames(m_lg))]
        @test Margins._dropped_coefficients(r_lm) == Int[]                    # full rank
        @test Margins._dropped_coefficients(lm(@formula(yc ~ x + w), d; dropcollinear = false)) == Int[]
        @test isnothing(Margins._dropped_coefficients(nothing))               # unknown model class
        # the fallback for a model class without rank information: estimate 0 and a non-finite variance
        V = vcov(m_lm)
        @test Margins._nonfinite_zero_coefs(V, coef(m_lm)) == [jz]
        @test Margins._nonfinite_zero_coefs(vcov(r_lm), coef(r_lm)) == Int[]
    end

    @testset "Σ: dropped rows and columns zeroed, the rest the model's" begin
        res = rd(() -> population_margins(m_lm, d; vars = [:x]))
        Σ = res.inference.vcov
        keep = setdiff(1:length(coef(m_lm)), jz)
        @test res.inference.dropped == [jz]
        @test all(iszero, Σ[jz, :]) && all(iszero, Σ[:, jz])
        @test Σ[keep, keep] == vcov(m_lm)[keep, keep]
        @test Σ[keep, keep] == vcov(r_lm)                                     # GLM.jl: the same numbers
        # full-rank fit: nothing dropped, no warning
        full = @test_logs population_margins(r_lm, d; vars = [:x])
        @test full.inference.dropped == Int[]
        @test full.inference.vcov == vcov(r_lm)
    end

    @testset "lm: every quantity equals the refit's (z: estimate 0, SE NaN)" begin
        a = DataFrame(rd(() -> population_margins(m_lm, d)))
        @test a[a.variable .== "z", :estimate] == [0.0]
        @test isnan(only(a[a.variable .== "z", :se]))
        @test all(isnan, a[a.variable .== "z", [:statistic, :p_value, :ci_lower, :ci_upper]] |> Matrix)
        same_rows(a, DataFrame(population_margins(r_lm, d)))
        # AAP, groups, scenarios, pairwise contrasts, MEM, APM
        same_rows(DataFrame(rd(() -> population_margins(m_lm, d; type = :predictions))),
                  DataFrame(population_margins(r_lm, d; type = :predictions)))
        same_rows(DataFrame(rd(() -> population_margins(m_lm, d; type = :predictions, groups = :g))),
                  DataFrame(population_margins(r_lm, d; type = :predictions, groups = :g)))
        a = DataFrame(rd(() -> population_margins(m_lm, d; groups = :g, contrasts = :pairwise)))
        @test all(isnan, a[a.variable .== "z", :se]) && all(iszero, a[a.variable .== "z", :estimate])
        same_rows(a, DataFrame(population_margins(r_lm, d; groups = :g, contrasts = :pairwise)))
        same_rows(DataFrame(rd(() -> population_margins(m_lm, d; vars = [:x, :g], scenarios = (w = [1.0, 2.0],)))),
                  DataFrame(population_margins(r_lm, d; vars = [:x, :g], scenarios = (w = [1.0, 2.0],))))
        a = DataFrame(rd(() -> profile_margins(m_lm, d, grid)))
        same_rows(a, DataFrame(profile_margins(r_lm, d, rgrid)))
        same_rows(DataFrame(rd(() -> profile_margins(m_lm, d, grid; type = :predictions))),
                  DataFrame(profile_margins(r_lm, d, rgrid; type = :predictions)))
    end

    @testset "logit: response and link scale, AD and FD" begin
        for scale in (:response, :link), backend in (:ad, :fd)
            a = DataFrame(rd(() -> population_margins(m_lg, d; scale, backend)))
            @test only(a[a.variable .== "z", :estimate]) == 0
            @test isnan(only(a[a.variable .== "z", :se]))
            same_rows(a, DataFrame(population_margins(r_lg, d; scale, backend)); rtol = 1e-10)
            same_rows(DataFrame(rd(() -> population_margins(m_lg, d; scale, backend, groups = :g, vars = [:x, :v]))),
                      DataFrame(population_margins(r_lg, d; scale, backend, groups = :g, vars = [:x, :v])); rtol = 1e-10)
            same_rows(DataFrame(rd(() -> profile_margins(m_lg, d, grid; scale, backend))),
                      DataFrame(profile_margins(r_lg, d, rgrid; scale, backend)); rtol = 1e-10)
        end
        same_rows(DataFrame(rd(() -> population_margins(m_lg, d; type = :predictions, groups = :g))),
                  DataFrame(population_margins(r_lg, d; type = :predictions, groups = :g)); rtol = 1e-10)
        # the effect of z loads on the dropped coefficient only, in every stratum: NaN
        a = DataFrame(rd(() -> population_margins(m_lg, d; vars = [:z], groups = :g)))
        @test all(iszero, a.estimate) && all(isnan, a.se)
    end

    @testset "contrast() and second differences" begin
        res = rd(() -> population_margins(m_lg, d; vars = [:x, :z], groups = :g))
        ref = population_margins(r_lg, d; vars = [:x], groups = :g)
        rows_x = findall(==("x"), res.variables)
        rows_z = findall(==("z"), res.variables)
        c = contrast(res, rows_x[2], rows_x[1])
        cr = contrast(ref, 2, 1)
        @test c.contrast ≈ cr.contrast rtol = 1e-10
        @test c.se ≈ cr.se rtol = 1e-10
        cz = contrast(res, rows_z[2], rows_z[1])                 # not identified
        @test isnan(cz.se) && isnan(cz.p_value) && isnan(cz.ci_lower)
        # model-level second differences, at = :observed and at a point, and the result-level form
        sd = rd(() -> second_differences(m_lg, d, :x, :w))
        @test sd.se ≈ second_differences(r_lg, d, :x, :w).se rtol = 1e-9
        sd = @test_logs (:warn, r"rank deficient") (:warn, r"zero by construction") second_differences(m_lm, d, [:x, :v], :w; at = 1.2)
        same_rows(sd, @test_logs (:warn, r"zero by construction") second_differences(r_lm, d, [:x, :v], :w; at = 1.2))
        sd = rd(() -> second_differences(m_lg, d, :x, :g))
        same_rows(sd, second_differences(r_lg, d, :x, :g); rtol = 1e-10)
        ames = rd(() -> population_margins(m_lg, d; vars = [:x, :z], scenarios = (w = [1.0, 2.0],)))
        sd = @test_logs second_differences(ames, [:x, :z], :w; modifier_type = :binary)  # no structural-zero warning for z
        ref_sd = second_differences(population_margins(r_lg, d; vars = [:x], scenarios = (w = [1.0, 2.0],)), :x, :w;
                                    modifier_type = :binary)
        @test sd.se[sd.variable .== :x] ≈ ref_sd.se rtol = 1e-10
        @test isnan(only(sd.se[sd.variable .== :z]))               # the effect of z: not identified
        @test isnan(only(sd.p_value[sd.variable .== :z]))
    end

    @testset "vcov= matrices, functions and CovarianceMatrices estimators" begin
        a = DataFrame(population_margins(r_lm, d))
        # the model's own matrix, NaN rows included, passed as a matrix or a function
        same_rows(DataFrame(rd(() -> population_margins(m_lm, d; vcov = vcov(m_lm)))), a)
        same_rows(DataFrame(rd(() -> population_margins(m_lm, d; vcov = m -> vcov(m)))), a)
        # a user matrix: any value in the dropped rows is ignored (the coefficient is fixed at 0)
        V = vcov(m_lm); V[jz, :] .= 1.0; V[:, jz] .= 1.0
        same_rows(DataFrame(rd(() -> population_margins(m_lm, d; vcov = V))), a)
        # non-finite values anywhere else are an error, not silently zeroed
        V = vcov(m_lm); V[1, 2] = V[2, 1] = NaN
        @test_throws r"non-finite values in the rows of coefficient\(s\) \(Intercept\), x" population_margins(m_lm, d; vcov = V)
        V = copy(vcov(r_lm)); V[2, 2] = Inf
        @test_throws ArgumentError population_margins(r_lm, d; vcov = V)
        @test_throws ArgumentError population_margins(r_lm, d; vcov = m -> fill(NaN, 7, 7))
        @test_throws r"3×3 matrix but the model has 7" population_margins(r_lm, d; vcov = m -> zeros(3, 3))
        # HC0: NaN rows for the dropped coefficient, the rest as for the refit. HC1 scales by
        # n/(n − p) with p counting the dropped coefficient (CovarianceMatrices 0.30); Margins
        # rescales it to n/(n − rank), the refit's (and R sandwich's) value
        for est in (HC0(), HC1())
            for (m, r) in ((m_lm, r_lm), (m_lg, r_lg))
                same_rows(DataFrame(rd(() -> population_margins(m, d; vcov = est))),
                          DataFrame(population_margins(r, d; vcov = est)); rtol = 1e-10)
            end
        end
        @test sqrt(CovarianceMatrices.vcov(HC1(), m_lm)[2, 2] / CovarianceMatrices.vcov(HC1(), r_lm)[2, 2]) ≈
              sqrt((400 - 7) / (400 - 8))                        # the convention being corrected
        # HC2-HC5: CovarianceMatrices' leverage of a rank-deficient GLM.jl fit is wrong (it takes the
        # first `rank` columns of X), so they are refused
        for est in (HC2(), HC3(), HC4(), HC5())
            @test_throws r"leverage-based" population_margins(m_lg, d; vcov = est)
        end
        keep = setdiff(1:length(coef(m_lg)), Margins._dropped_coefficients(m_lg))
        @test !isapprox(CovarianceMatrices.vcov(HC3(), m_lg)[keep, keep], CovarianceMatrices.vcov(HC3(), r_lg); rtol = 1e-3)  # still wrong upstream
        # (a full-rank fit is unaffected)
        @test_logs population_margins(r_lg, d; vcov = HC3())
        # CR estimators fail inside CovarianceMatrices on a rank-deficient GLM.jl fit: a clear error
        @test_throws r"rank deficient.*Refit without the collinear term" population_margins(m_lm, d; vcov = CR1(d.cl))
        @test_throws r"rank deficient" population_margins(m_lm, d; vcov = CR1(d.cl), warn = false)
    end

    @testset "FixedEffectModels: a regressor collinear with the fixed effects" begin
        for vc in (Vcov.simple(), Vcov.robust(), Vcov.cluster(:cl))
            f = reg(d, @formula(yc ~ x * w + z + c + fe(g)), vc; save = :fe, tol = 1e-12)
            f2 = reg(d, @formula(yc ~ x * w + z + fe(g)), vc; save = :fe, tol = 1e-12)
            @test coefnames(f)[Margins._dropped_coefficients(f)] == ["c"]
            a = DataFrame(rdc(() -> population_margins(f, d; weights = :none)))
            @test only(a[a.variable .== "c", :estimate]) == 0 && isnan(only(a[a.variable .== "c", :se]))
            same_rows(a, DataFrame(population_margins(f2, d; weights = :none)); skip = "c")
            same_rows(DataFrame(rdc(() -> population_margins(f, d; weights = :none, vars = [:x, :z], groups = :g))),
                      DataFrame(population_margins(f2, d; weights = :none, vars = [:x, :z], groups = :g)); skip = "c")
            same_rows(DataFrame(rdc(() -> population_margins(f, d; weights = :none, type = :predictions))),
                      DataFrame(population_margins(f2, d; weights = :none, type = :predictions)))
            fg = DataFrame(x = [1.5], w = [1.2], z = [0.1], c = [1.9], g = ["b"])
            a = DataFrame(rdc(() -> profile_margins(f, d, fg; vars = [:x, :w, :z, :c])))
            @test isnan(only(a[a.variable .== "c", :se]))
            same_rows(a, DataFrame(profile_margins(f2, d, select(fg, Not(:c)); vars = [:x, :w, :z])); skip = "c")
            same_rows(rdc(() -> second_differences(f, d, :x, :w; weights = :none)),
                      second_differences(f2, d, :x, :w; weights = :none); skip = "c")
        end
    end

    @testset "MixedModels: rank-deficient fixed effects" begin
        lmm = @test_logs (:warn, r"rank deficient") match_mode = :any fit(MixedModel,
            @formula(yc ~ x + w + v + z + (1 | g)), d; progress = false)
        # MixedModels pivots by its own rule (here it drops x); refit without the term it dropped
        dn = only(coefnames(lmm)[Margins._dropped_coefficients(lmm)])
        @test dn in ("x", "v", "z")
        keep = filter(!=(dn), ["x", "w", "v", "z"])
        lmm2 = fit(MixedModel, eval(Meta.parse("@formula(yc ~ " * join(keep, " + ") * " + (1 | g))")), d; progress = false)
        a = DataFrame(@test_logs (:warn, Regex("rank deficient.*coefficient\\(s\\) $dn ")) population_margins(lmm, d))
        @test only(a[a.variable .== dn, :estimate]) == 0 && isnan(only(a[a.variable .== dn, :se]))
        # the same fixed-effects fit, up to the optimizer's θ path
        same_rows(a, DataFrame(population_margins(lmm2, d)); skip = dn, rtol = 1e-6)
    end

    @testset "zero allocation on a rank-deficient fit" begin
        dnt = Margins._prepare_margins_data(Tables.columntable(d), m_lg)
        req = @test_logs (:warn, r"rank deficient") Margins._normalize_request(m_lg, d; vars = [:x, :z])
        engine = Margins.build_engine(Margins.PopulationUsage, m_lg, dnt, [:x, :z], req.Σ, :ad)
        @test _ame_loop_allocs(engine, [:x, :z], dnt, 1:nrow(d)) == 0
        alloc(f) = @allocated f()
        g = zeros(length(coef(m_lg))); g[req.dropped[1]] = 1.0
        h() = Margins._not_identified(g, req.dropped)
        h()
        @test h() && alloc(h) == 0
        se = rand(4); G = rand(4, length(g))
        k() = Margins._mark_not_identified!(se, G, req.dropped)
        k()
        @test alloc(k) == 0
    end
end
