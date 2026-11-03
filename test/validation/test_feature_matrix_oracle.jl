# test/validation/test_feature_matrix_oracle.jl
#
# Step 0 of the R6/R7 refactor ("oracles first"): an independent oracle over the feature
# matrix that the prediction kernel (R6) and the continuous-effects accumulator (R7) will be
# rewritten under.
#
#   paths    population (no context) | groups (categorical `h`, binned `(:z, 3)`) |
#            scenarios (continuous `z = 0.5`, categorical `g = "b"`) |
#            profile (`means_grid`, explicit `cartesian_grid`)
#   scale    :response, :link
#   weights  none | positive | positive with ~15% exact zeros       (population paths only)
#   measure  :effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx
#            (the last three for continuous variables only)
#   backend  :ad, :fd
#   links    Identity (lm), Logit, Probit, Cloglog, Log (Poisson), Inverse (Gamma),
#            NegativeBinomialLink(θ = 2)
#   type     :effects for every cell above; :predictions once per (link, scale, weights, path)
#
# Model: y ~ x * z + x * g + b   (continuous × continuous, continuous × categorical, a Bool,
# a 3-level categorical), so derivatives and contrasts both pass through interactions.
#
# Oracle (nothing from Margins is used):
#   * design rows from `modelcols(formula(m).rhs, data)`, `coef(m)`, `vcov(m)`, and
#     `GLM.linkinv`; ŷ = η on the link scale;
#   * unit derivatives ∂ŷᵢ/∂x by Richardson-extrapolated central differences in x
#     (h = 1e-3 and 5e-4, truncation O(h⁴) ≈ 1e-12);
#   * measures as (weighted) means of unit quantities: gᵢ, gᵢxᵢ/ŷᵢ, gᵢxᵢ, gᵢ/ŷᵢ;
#   * categorical and Bool contrasts from counterfactual design rows (level vs baseline,
#     true vs false); scenarios replace the column for every row; profile rows with a
#     frequency mixture are the probability-weighted sum of the per-level design rows (exact,
#     because every column is linear in each factor's indicators);
#   * SEs by the delta method, √(∇'V∇), with ∇ = ∂estimate/∂β by Richardson-extrapolated
#     central differences of the whole estimator in β (hβ = 1e-4·max(1, |βⱼ|)).
#
# Tolerances: estimates rtol = 1e-6, SEs rtol = 1e-5 (atol = 1e-9 for values that are
# genuinely ≈ 0). The oracle itself is accurate to ~1e-10 (estimates) and ~1e-8 (SEs);
# the budget is set by the FD backend, whose central step (~eps^(1/3)·max(1,|x|) ≈ 6e-6)
# gives O(h²) + O(eps/h) ≈ 1e-10 relative error in each unit derivative, amplified in the
# SE by the second-derivative term on the response scale and by 1/ŷ in the ratio measures.
# Real algebra errors (wrong chain rule, weights or rows) show orders of magnitude above
# this. The largest disagreements actually observed are logged at the end (@info).
# Do NOT loosen these to make a failure pass: mark it `@test_broken` with an explanation.
#
# Data: n = 400 per link, deterministic seed; x ∈ [0.5, 2.5] and the linear predictor is
# kept away from 0 so that link-scale ratio measures (which divide by η) are well posed —
# checked below. Each link gets its own DataFrame and model, so engine-cache entries never
# collide (the cache is keyed on the model and the column names).

module FeatureMatrixOracleTests

using Test
using Random, Statistics, LinearAlgebra
using DataFrames, CategoricalArrays, GLM, StatsModels, Tables
using Margins

const N = 400
const RTOL_EST = 1e-6
const RTOL_SE = 1e-5
const ATOL = 1e-9
const MEASURES = (:effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx)
const CONTINUOUS = (:x, :z)
const GLEVELS = ["a", "b", "c"]

# ---------------------------------------------------------------------------------------
# Data and models
# ---------------------------------------------------------------------------------------

function _poisson_draw(rng, λ)          # Knuth; λ is small here
    L = exp(-λ); k = 0; p = 1.0
    while true
        p *= rand(rng); p <= L && return k; k += 1
    end
end

"""Covariates shared by every link, plus `y` drawn from a family-appropriate model."""
function _make_data(linkname, seed)
    rng = MersenneTwister(seed)
    x = 0.5 .+ 2 .* rand(rng, N)
    z = 2 .* rand(rng, N) .- 1
    g = categorical(rand(rng, GLEVELS, N); levels=GLEVELS)
    b = rand(rng, N) .< 0.4
    h = categorical(rand(rng, ["p", "q", "r"], N); levels=["p", "q", "r"])
    w = 0.3 .+ rand(rng, N)
    w0 = copy(w); w0[rand(rng, N) .< 0.15] .= 0.0
    gb = Float64.(g .== "b"); gc = Float64.(g .== "c")
    s(c0, k) = c0 .+ k .* (0.3 .* x .+ 0.2 .* z .+ 0.15 .* x .* z .+ 0.25 .* gb .- 0.15 .* x .* gc .+ 0.3 .* b)
    u = rand(rng, N)
    y = if linkname == "identity"
        s(5.0, 1.0) .+ 0.5 .* randn(rng, N)
    elseif linkname == "logit"
        Float64.(u .< 1 ./ (1 .+ exp.(-s(2.0, 1.0))))
    elseif linkname == "probit"
        Float64.(u .< GLM.linkinv.(Ref(ProbitLink()), s(1.3, 0.5)))
    elseif linkname == "cloglog"
        Float64.(u .< 1 .- exp.(-exp.(s(-1.6, 0.5))))
    elseif linkname == "log"
        Float64.(_poisson_draw.(Ref(rng), exp.(s(1.3, 0.5))))
    elseif linkname == "inverse"
        μ = 1 ./ s(0.25, 0.1)
        μ .* [sum(-log(rand(rng)) for _ in 1:6) / 6 for _ in 1:N]   # Gamma(shape 6) noise
    elseif linkname == "negbin"
        Float64.(_poisson_draw.(Ref(rng), exp.(s(1.3, 0.5)) .* (0.5 .+ rand(rng, N))))
    end
    return DataFrame(y=y, x=x, z=z, g=g, b=b, h=h, w=w, w0=w0)
end

const F = @formula(y ~ x * z + x * g + b)

function _fit(linkname, d)
    linkname == "identity" && return (lm(F, d), IdentityLink())
    linkname == "logit"    && return (glm(F, d, Binomial(), LogitLink()), LogitLink())
    linkname == "probit"   && return (glm(F, d, Binomial(), ProbitLink()), ProbitLink())
    linkname == "cloglog"  && return (glm(F, d, Binomial(), CloglogLink()), CloglogLink())
    linkname == "log"      && return (glm(F, d, Poisson(), LogLink()), LogLink())
    linkname == "inverse"  && return (glm(F, d, Gamma(), InverseLink()), InverseLink())
    linkname == "negbin"   && return (glm(F, d, NegativeBinomial(2.0), NegativeBinomialLink(2.0)),
                                      NegativeBinomialLink(2.0))
    error("unknown link $linkname")
end

const LINKS = ["identity", "logit", "probit", "cloglog", "log", "inverse", "negbin"]

# ---------------------------------------------------------------------------------------
# Oracle primitives
# ---------------------------------------------------------------------------------------

struct OModel
    rhs
    link
    β::Vector{Float64}
    V::Matrix{Float64}
end
OModel(m, link) = OModel(formula(m).rhs, link, coef(m), Matrix(vcov(m)))

_X(om::OModel, d) = modelcols(om.rhs, Tables.columntable(d))
_pred(om, X, b, scale) = scale === :link ? X * b : GLM.linkinv.(Ref(om.link), X * b)

_wmean(v, rows, w) = isnothing(w) ? mean(v[rows]) : sum(w[rows] .* v[rows]) / sum(w[rows])

"""Richardson central difference of a scalar function of β, coordinatewise."""
function _grad_beta(est, β)
    map(eachindex(β)) do j
        h = 1e-4 * max(1.0, abs(β[j]))
        e = zeros(length(β)); e[j] = 1.0
        D(s) = (est(β .+ s .* e) - est(β .- s .* e)) / (2s)
        (4D(h / 2) - D(h)) / 3
    end
end

_se(om, est) = (gr = _grad_beta(est, om.β); sqrt(max(0.0, dot(gr, om.V, gr))))

"""
Design matrices for a context (a full-length counterfactual DataFrame) needed to evaluate
predictions, x-derivatives (four shifted copies per continuous variable) and contrasts.
"""
struct CtxMats
    X0::Matrix{Float64}
    shifts::Dict{Symbol, NTuple{4, Matrix{Float64}}}   # (+h, -h, +h/2, -h/2)
    cf::Dict{Tuple{Symbol, Any}, Matrix{Float64}}      # (var, level) => counterfactual X
    xvals::Dict{Symbol, Vector{Float64}}
end

const HX = 1e-3

function _with(d, var, val)
    dd = copy(d)
    if var === :g
        dd.g = categorical(fill(val, nrow(d)); levels=GLEVELS)
    elseif var === :b
        dd.b = fill(val, nrow(d))
    else
        dd[!, var] = fill(Float64(val), nrow(d))
    end
    dd
end

function CtxMats(om::OModel, d::DataFrame)
    shifts = Dict{Symbol, NTuple{4, Matrix{Float64}}}()
    for v in CONTINUOUS
        sh(δ) = (dd = copy(d); dd[!, v] = d[!, v] .+ δ; _X(om, dd))
        shifts[v] = (sh(HX), sh(-HX), sh(HX / 2), sh(-HX / 2))
    end
    cf = Dict{Tuple{Symbol, Any}, Matrix{Float64}}()
    for lev in GLEVELS
        cf[(:g, lev)] = _X(om, _with(d, :g, lev))
    end
    for bv in (true, false)
        cf[(:b, bv)] = _X(om, _with(d, :b, bv))
    end
    CtxMats(_X(om, d), shifts, cf, Dict(v => Vector{Float64}(d[!, v]) for v in CONTINUOUS))
end

"""Unit-level ∂ŷ/∂v (Richardson in x) for every row of the context."""
function _unit_slope(om, cm::CtxMats, v, b, scale)
    Xp, Xm, Xp2, Xm2 = cm.shifts[v]
    D1 = (_pred(om, Xp, b, scale) .- _pred(om, Xm, b, scale)) ./ (2HX)
    D2 = (_pred(om, Xp2, b, scale) .- _pred(om, Xm2, b, scale)) ./ HX
    return (4 .* D2 .- D1) ./ 3
end

function _unit_measure(om, cm, v, b, scale, measure)
    gi = _unit_slope(om, cm, v, b, scale)
    measure === :effect && return gi
    xi = cm.xvals[v]
    measure === :semielasticity_dyex && return gi .* xi
    ŷ = _pred(om, cm.X0, b, scale)
    measure === :elasticity && return gi .* xi ./ ŷ
    measure === :semielasticity_eydx && return gi ./ ŷ
    error("measure $measure")
end

"""
(estimate, se) for one result row: `var` with `contrast` ("dy/dx" or "L - R") over `rows`.
"""
function _oracle_effect(om, cm, var, contrast, rows, w, scale, measure)
    est = if var in CONTINUOUS
        b -> _wmean(_unit_measure(om, cm, var, b, scale, measure), rows, w)
    else
        lhs, rhs = strip.(split(contrast, " - "))
        conv(s) = var === :b ? parse(Bool, s) : String(s)
        X1, X0 = cm.cf[(var, conv(lhs))], cm.cf[(var, conv(rhs))]
        b -> _wmean(_pred(om, X1, b, scale) .- _pred(om, X0, b, scale), rows, w)
    end
    return est(om.β), _se(om, est)
end

function _oracle_prediction(om, cm, rows, w, scale)
    est = b -> _wmean(_pred(om, cm.X0, b, scale), rows, w)
    return est(om.β), _se(om, est)
end

# ---------------------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------------------

const COUNTS = Dict{String, Int}()
_count!(k) = (COUNTS[k] = get(COUNTS, k, 0) + 1)
const MAXREL = Dict{String, Float64}()   # largest observed relative disagreement
_rel(a, b) = abs(a - b) / max(abs(a), abs(b), 1e-300)
_maxrel!(k, v) = (MAXREL[k] = max(get(MAXREL, k, 0.0), v))

"""Context column: accept the current name and the planned 3.0 `over_` prefix."""
function _ctxcol(df::DataFrame, name::AbstractString)
    for c in (name, "over_" * name, "at_" * name)
        hasproperty(df, c) && return df[!, c]
    end
    error("no context column for $name in $(names(df))")
end

_is_dydx(c) = c in ("dy/dx", "derivative")

"""
Compare a Margins result row against an oracle (estimate, se). `known_broken` turns a
mismatch into `@test_broken` (it must then actually fail, so a later fix is noticed).
"""
function _check(label, r, est, se; known_broken=false)
    ok_e = isapprox(r.estimate, est; rtol=RTOL_EST, atol=ATOL)
    ok_s = isapprox(r.se, se; rtol=RTOL_SE, atol=ATOL)
    _count!("rows compared")
    bk = occursin("/fd", label) ? "fd" : occursin("/ad", label) ? "ad" : "predictions"
    _maxrel!("estimate ($bk)", _rel(r.estimate, est)); _maxrel!("se ($bk)", _rel(r.se, se))
    if known_broken
        @test_broken ok_e && ok_s
    else
        if !(ok_e && ok_s)
            @error "oracle mismatch" label variable=(hasproperty(r, :variable) ? r.variable : "-") contrast=(hasproperty(r, :contrast) ? r.contrast : "-") margins_est=r.estimate oracle_est=est margins_se=r.se oracle_se=se
        end
        @test ok_e
        @test ok_s
    end
end

# ---------------------------------------------------------------------------------------
# Known oracle failures
# ---------------------------------------------------------------------------------------
# A combination that disagrees with the oracle is a real finding: list it here (keyed on
# label/var/measure/scale) with an explanation, so it runs as `@test_broken`, rather than
# loosening a tolerance. None as of step 0 (Margins 2.3.0 + FormulaCompiler 1.5.0).

_known_broken(; label, var, measure, scale) = false

# ---------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------

_vars_for(measure, candidates) = measure === :effect ? candidates : filter(in(CONTINUOUS), candidates)

"""Population-path check of `res` (a DataFrame) for one context family."""
function _check_population_rows(label, om, res, ctx_rows_of, cm_of, w, scale, measure, vars;
                                ctxname=nothing)
    # every requested variable must be present (categoricals: one row per non-baseline level)
    nexp = sum(v === :g ? 2 : 1 for v in vars)
    nctx = ctxname === nothing ? 1 : length(unique(string.(_ctxcol(res, ctxname))))
    @test nrow(res) == nexp * nctx
    for r in eachrow(res)
        cval = ctxname === nothing ? nothing : string(_ctxcol(res, ctxname)[rownumber(r)])
        rows = ctx_rows_of(cval)
        var = Symbol(r.variable)
        if var in CONTINUOUS
            @test _is_dydx(r.contrast)
        end
        est, se = _oracle_effect(om, cm_of(cval), var, r.contrast, rows, w, scale, measure)
        _check("$label ctx=$cval", r, est, se;
               known_broken=_known_broken(; label, var, measure, scale))
    end
end

function _profile_X(om, pt)   # pt: Dict(:x, :z, :b => Float64/Bool, :g => level or Dict(level => p))
    gmix = pt[:g] isa AbstractDict ? pt[:g] : Dict(pt[:g] => 1.0)
    levs = collect(keys(gmix))
    dd = DataFrame(x=fill(Float64(pt[:x]), length(levs)), z=fill(Float64(pt[:z]), length(levs)),
                   g=categorical(levs; levels=GLEVELS), b=fill(pt[:b], length(levs)))
    X = _X(om, dd)
    return reshape(sum(gmix[l] .* X[k, :] for (k, l) in enumerate(levs)), 1, :)
end

"""Profile oracle at a single point `pt` for one result row."""
function _oracle_profile(om, pt, var, contrast, scale, measure)
    est = if var in CONTINUOUS
        Xs(δ) = (q = copy(pt); q[var] = pt[var] + δ; _profile_X(om, q))
        Xp, Xm, Xp2, Xm2 = Xs(HX), Xs(-HX), Xs(HX / 2), Xs(-HX / 2)
        X0 = _profile_X(om, pt)
        function (b)
            p(X) = only(_pred(om, X, b, scale))
            D1 = (p(Xp) - p(Xm)) / (2HX); D2 = (p(Xp2) - p(Xm2)) / HX
            gi = (4D2 - D1) / 3
            measure === :effect && return gi
            measure === :semielasticity_dyex && return gi * pt[var]
            ŷ = p(X0)
            measure === :elasticity && return gi * pt[var] / ŷ
            return gi / ŷ
        end
    else
        lhs, rhs = strip.(split(contrast, " - "))
        conv(s) = var === :b ? parse(Bool, s) : String(s)
        q1 = copy(pt); q1[var] = conv(lhs); q0 = copy(pt); q0[var] = conv(rhs)
        X1, X0 = _profile_X(om, q1), _profile_X(om, q0)
        b -> only(_pred(om, X1, b, scale)) - only(_pred(om, X0, b, scale))
    end
    return est(om.β), _se(om, est)
end

@testset "feature-matrix oracle" begin
    t0 = time()
    for (li, linkname) in enumerate(LINKS)
        d = _make_data(linkname, 20261016 + li)
        m, link = _fit(linkname, d)
        om = OModel(m, link)

        @testset "$linkname: setup" begin
            # the link-scale ratio measures divide by η; keep it away from 0
            @test minimum(abs, modelmatrix(m) * coef(m)) > 0.1
            @test all(isfinite, coef(m))
        end

        # Contexts (full-length counterfactual data + member rows)
        allrows = collect(1:N)
        cm_base = CtxMats(om, d)
        hrows = Dict(l => findall(==(l), string.(d.h)) for l in ("p", "q", "r"))
        # Tercile bins of z: with weights, the cutpoints are weighted quantiles (each bin
        # carries about 1/3 of the weight); the weighted-quantile definition itself is
        # tested in regression/test_group_binning.jl
        function zrows_of(w)
            qz = isnothing(w) ? quantile(d.z, [0, 1/3, 2/3, 1]) :
                                Margins._weighted_quantiles(d.z, w, [0, 1/3, 2/3, 1])
            Dict("T$i" => findall(z -> (i == 3 ? qz[i] <= z <= qz[i+1] : qz[i] <= z < qz[i+1]), d.z)
                 for i in 1:3)
        end
        zrows = zrows_of(nothing)
        cm_z = CtxMats(om, _with(d, :z, 0.5))
        cm_g = CtxMats(om, _with(d, :g, "b"))

        # (name, Margins kwargs, ctx column, rows-of(ctxvalue), mats-of(ctxvalue), vars)
        paths = [
            ("population", (;), nothing, _ -> allrows, _ -> cm_base, [:x, :z, :g, :b]),
            ("groups=h", (; groups=:h), "h", c -> hrows[c], _ -> cm_base, [:x, :z, :g, :b]),
            ("groups=(z,3)", (; groups=(:z, 3)), "z", c -> zrows[c], _ -> cm_base, [:x, :g, :b]),
            ("scenarios z=0.5", (; scenarios=(z=0.5,)), "z", _ -> allrows, _ -> cm_z, [:x, :g, :b]),
            ("scenarios g=b", (; scenarios=(g="b",)), "g", _ -> allrows, _ -> cm_g, [:x, :z, :b]),
        ]

        for scale in (:response, :link), (wname, wcol) in (("none", nothing), ("w", :w), ("w0", :w0))
            w = isnothing(wcol) ? nothing : d[!, wcol]
            zrows = zrows_of(w)
            wkw = isnothing(wcol) ? (;) : (; weights=wcol)
            @testset "$linkname scale=$scale weights=$wname" begin
                for (pname, pkw, ctxname, rows_of, cm_of, pvars) in paths
                    for measure in MEASURES, backend in (:ad, :fd)
                        vars = _vars_for(measure, pvars)
                        label = "$linkname/$scale/$wname/$pname/$measure/$backend"
                        res = DataFrame(population_margins(m, d; type=:effects, vars, scale,
                                                           measure, backend, pkw..., wkw...))
                        _check_population_rows(label, om, res, rows_of, cm_of, w, scale,
                                               measure, vars; ctxname)
                        _count!("effects calls")
                    end
                    # predictions (no backend/measure dimension)
                    res = DataFrame(population_margins(m, d; type=:predictions, scale,
                                                       pkw..., wkw...))
                    @test nrow(res) == (ctxname === nothing ? 1 : length(unique(string.(_ctxcol(res, ctxname)))))
                    for (k, r) in enumerate(eachrow(res))
                        cval = ctxname === nothing ? nothing : string(_ctxcol(res, ctxname)[k])
                        est, se = _oracle_prediction(om, cm_of(cval), rows_of(cval), w, scale)
                        _check("$linkname/$scale/$wname/$pname/predictions ctx=$cval", r, est, se)
                    end
                    _count!("prediction calls")
                end
            end
        end

        # Profiles (no weights argument)
        gfreq = Dict(l => count(==(l), string.(d.g)) / N for l in GLEVELS)
        zbar, xbar, bbar = mean(d.z), mean(d.x), mean(d.b)
        mg = means_grid(d)
        cg = cartesian_grid(x=[1.0, 2.0], g=["a", "c"])
        for scale in (:response, :link)
            @testset "$linkname profile scale=$scale" begin
                for (gname, grid) in (("means_grid", mg), ("cartesian_grid", cg))
                    point(r) = gname == "means_grid" ?
                        Dict{Symbol, Any}(:x => xbar, :z => zbar, :b => bbar, :g => gfreq) :
                        Dict{Symbol, Any}(:x => Float64(r.x), :z => zbar, :b => bbar, :g => string(r.g))
                    for measure in MEASURES, backend in (:ad, :fd)
                        vars = _vars_for(measure, [:x, :z, :g, :b])
                        label = "$linkname/$scale/profile:$gname/$measure/$backend"
                        res = DataFrame(profile_margins(m, d, grid; type=:effects, vars, scale,
                                                        measure, backend))
                        # every continuous var and the Bool appear once per profile
                        nprof = nrow(grid)
                        for v in filter(!=(:g), vars)
                            @test count(==(string(v)), res.variable) == nprof
                        end
                        for r in eachrow(res)
                            est, se = _oracle_profile(om, point(r), Symbol(r.variable), r.contrast,
                                                      scale, measure)
                            _check("$label x=$(r.x)", r, est, se;
                                   known_broken=_known_broken(; label, var=Symbol(r.variable),
                                                              measure, scale))
                        end
                        _count!("effects calls")
                    end
                    res = DataFrame(profile_margins(m, d, grid; type=:predictions, scale))
                    @test nrow(res) == nrow(grid)
                    for r in eachrow(res)
                        X = _profile_X(om, point(r))
                        est = b -> only(_pred(om, X, b, scale))
                        _check("$linkname/$scale/profile:$gname/predictions", r, est(om.β), _se(om, est))
                    end
                    _count!("prediction calls")
                end
            end
        end
    end
    @info "feature-matrix oracle" counts=join(["$k=$v" for (k, v) in sort(collect(COUNTS))], ", ") max_rel_disagreement=join(["$k=$(round(v; sigdigits=2))" for (k, v) in sort(collect(MAXREL))], ", ") seconds=round(time() - t0; digits=1)
end

end # module
