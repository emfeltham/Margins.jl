# test/support/golden_cases.jl
#
# Shared definition of the golden-snapshot cases for the R6/R7 refactor (Margins 3.0):
# a fixed, deterministic set of Margins calls and a schema-robust way of turning each result
# into (key, estimate, se) rows. Included by
#   test/support/generate_golden_references.jl   (writes test/support/golden/margins_golden.csv)
#   test/validation/test_golden_equivalence.jl   (reruns the calls and compares)
#
# Data are built WITHOUT an RNG (Weyl sequences), so the snapshot does not depend on the
# RNG streams of a particular Julia version.
#
# Row keys are built from the variable, the contrast and the context values only — never
# from column positions or from `t_stat` — and context column names are normalized
# (`over_g` and `at_g` both become `g`), so the keys survive the output-schema changes
# planned for 3.0 (`t_stat` → `statistic` + `df`, `over_` group prefixes).

module MarginsGolden

using DataFrames, CategoricalArrays, GLM, StatsModels, Statistics
using CovarianceMatrices, MixedModels, FixedEffectModels
using Margins
using FormulaCompiler: CategoricalMixture

const GLEVELS = ["a", "b", "c"]
const MEASURES = (:effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx)
const CONTINUOUS = (:x, :z)
const LINKS = ["identity", "logit", "probit", "cloglog", "log", "inverse", "negbin"]

_frac(t) = t - floor(t)

"""Inverse-CDF Poisson draw from a uniform `u`."""
function _poisson_inv(u, λ)
    k = 0; p = exp(-λ); c = p
    while u > c && k < 1000
        k += 1; p *= λ / k; c += p
    end
    return k
end

"""Deterministic data for one link (same structure as the feature-matrix oracle)."""
function matrix_data(linkname; n=400)
    i = 1:n
    x = 0.5 .+ 2 .* _frac.(i .* 0.6180339887498949 .+ 0.1)
    z = 2 .* _frac.(i .* 0.7548776662466927 .+ 0.3) .- 1
    g = categorical(GLEVELS[1 .+ mod.(i .* 7 .+ div.(i, 5), 3)]; levels=GLEVELS)
    b = _frac.(i .* 0.5698402909980532) .< 0.4
    h = categorical(["p", "q", "r"][1 .+ mod.(i .* 11 .+ div.(i, 3), 3)]; levels=["p", "q", "r"])
    w = 0.3 .+ _frac.(i .* 0.4142135623730951)
    w0 = ifelse.(_frac.(i .* 0.3247179572447460) .< 0.15, 0.0, w)
    u = _frac.(i .* i .* 0.1234567891234 .+ i .* 0.271828182845)
    gb = Float64.(g .== "b"); gc = Float64.(g .== "c")
    s(c0, k) = c0 .+ k .* (0.3 .* x .+ 0.2 .* z .+ 0.15 .* x .* z .+ 0.25 .* gb .- 0.15 .* x .* gc .+ 0.3 .* b)
    y = if linkname == "identity"
        s(5.0, 1.0) .+ (u .- 0.5)
    elseif linkname == "logit"
        Float64.(u .< 1 ./ (1 .+ exp.(-s(2.0, 1.0))))
    elseif linkname == "probit"
        Float64.(u .< GLM.linkinv.(Ref(ProbitLink()), s(1.3, 0.5)))
    elseif linkname == "cloglog"
        Float64.(u .< 1 .- exp.(-exp.(s(-1.6, 0.5))))
    elseif linkname == "log"
        Float64.(_poisson_inv.(u, exp.(s(1.3, 0.5))))
    elseif linkname == "inverse"
        (1 ./ s(0.25, 0.1)) .* (0.5 .+ u)
    elseif linkname == "negbin"
        Float64.(_poisson_inv.(u, exp.(s(1.3, 0.5)) .* (0.5 .+ _frac.(i .* 0.913))))
    end
    return DataFrame(y=y, x=x, z=z, g=g, b=b, h=h, w=w, w0=w0)
end

const F = @formula(y ~ x * z + x * g + b)

function fit_matrix_model(linkname, d; kw...)
    linkname == "identity" && return lm(F, d; kw...)
    linkname == "logit"    && return glm(F, d, Binomial(), LogitLink(); kw...)
    linkname == "probit"   && return glm(F, d, Binomial(), ProbitLink(); kw...)
    linkname == "cloglog"  && return glm(F, d, Binomial(), CloglogLink(); kw...)
    linkname == "log"      && return glm(F, d, Poisson(), LogLink(); kw...)
    linkname == "inverse"  && return glm(F, d, Gamma(), InverseLink(); kw...)
    linkname == "negbin"   && return glm(F, d, NegativeBinomial(2.0), NegativeBinomialLink(2.0); kw...)
    error("unknown link $linkname")
end

"""Mid-size synthetic model for the six benchmark-style operations."""
function bench_data(; n=5000)
    i = 1:n
    rel_levels = ["family", "friend", "neighbor", "other"]
    rel_g = ["Catholic", "Protestant", "None"]
    d = DataFrame(
        age_h = 20 .+ 50 .* _frac.(i .* 0.6180339887498949),
        dist_inv = 1 ./ (1 .+ 5 .* _frac.(i .* 0.7548776662466927)),
        wealth_p = _frac.(i .* 0.5698402909980532) .* 4,
        wealth_h = _frac.(i .* 0.4142135623730951 .+ 0.2) .* 4,
        socio = _frac.(i .* 0.3247179572447460) .< 0.45,
        same_building = _frac.(i .* 0.2360679774997897) .< 0.2,
        relation = categorical(rel_levels[1 .+ mod.(i .* 5 .+ div.(i, 7), 4)]; levels=rel_levels),
        religion = categorical(rel_g[1 .+ mod.(i .* 13 .+ div.(i, 11), 3)]; levels=rel_g),
    )
    u = _frac.(i .* i .* 0.1234567891234 .+ i .* 0.271828182845)
    η = -0.5 .+ 0.02 .* (d.age_h .- 45) .+ 0.8 .* d.dist_inv .+ 0.3 .* d.wealth_p .- 0.2 .* d.wealth_h .+
        0.4 .* d.socio .+ 0.5 .* d.socio .* d.dist_inv .+ 0.3 .* (d.relation .== "family") .-
        0.2 .* d.same_building .+ 0.1 .* (d.religion .== "None") .+ 0.01 .* d.age_h .* (d.relation .== "friend")
    d.y = u .< 1 ./ (1 .+ exp.(-η))
    return d
end

const FB = @formula(y ~ age_h * relation + dist_inv * socio + wealth_p + wealth_h + same_building + religion)

fit_bench_model(d) = glm(FB, d, Binomial(), LogitLink())

# ---------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------

_vars_for(measure, candidates) = measure === :effect ? candidates : filter(in(CONTINUOUS), candidates)

"""
Vector of `(case_id, operation, thunk)`; each thunk returns a Margins result. Models are
fitted once, here. Repeated calls on the same objects must give the same results, so the
cases do not clear FormulaCompiler's compiled-formula cache between calls.
"""
function cases()
    out = Tuple{String, String, Function}[]
    paths = [
        ("population", (;), [:x, :z, :g, :b]),
        ("groups=h", (; groups=:h), [:x, :z, :g, :b]),
        ("groups=(z,3)", (; groups=(:z, 3)), [:x, :g, :b]),
        ("scenarios z=0.5", (; scenarios=(z=0.5,)), [:x, :g, :b]),
        ("scenarios g=b", (; scenarios=(g="b",)), [:x, :z, :b]),
    ]
    for linkname in LINKS
        d = matrix_data(linkname)
        m = fit_matrix_model(linkname, d)
        for scale in (:response, :link), wcol in (nothing, :w, :w0)
            wname = isnothing(wcol) ? "none" : string(wcol)
            wkw = isnothing(wcol) ? (;) : (; weights=wcol)
            for (pname, pkw, pvars) in paths
                for measure in MEASURES, backend in (:ad, :fd)
                    vars = _vars_for(measure, pvars)
                    push!(out, ("$linkname/$scale/$wname/$pname/$measure/$backend", "population_effects",
                                () -> population_margins(m, d; type=:effects, vars, scale, measure,
                                                         backend, pkw..., wkw...)))
                end
                push!(out, ("$linkname/$scale/$wname/$pname/predictions", "population_predictions",
                            () -> population_margins(m, d; type=:predictions, scale, pkw..., wkw...)))
            end
        end
        for scale in (:response, :link)
            for (gname, gridf) in (("means_grid", () -> means_grid(d)),
                                   ("cartesian_grid", () -> cartesian_grid(x=[1.0, 2.0], g=["a", "c"])))
                for measure in MEASURES, backend in (:ad, :fd)
                    vars = _vars_for(measure, [:x, :z, :g, :b])
                    push!(out, ("$linkname/$scale/profile:$gname/$measure/$backend", "profile_effects",
                                () -> profile_margins(m, d, gridf(); type=:effects, vars, scale,
                                                      measure, backend)))
                end
                push!(out, ("$linkname/$scale/profile:$gname/predictions", "profile_predictions",
                            () -> profile_margins(m, d, gridf(); type=:predictions, scale)))
            end
        end
    end

    # The six benchmark operations (cf. ~/.julia/dev/_wt/review/verify_bench.jl), each on
    # both backends where a backend applies.
    db = bench_data()
    mb = fit_bench_model(db)
    cg = () -> cartesian_grid(socio=[false, true], dist_inv=[1.0, 1/6], relation=["family"],
                              religion=["Catholic"], same_building=[false])
    push!(out, ("bench/APM", "APM", () -> profile_margins(mb, db, cg(); type=:predictions)))
    push!(out, ("bench/AAP", "AAP", () -> population_margins(mb, db; type=:predictions)))
    for backend in (:ad, :fd)
        push!(out, ("bench/MEM/$backend", "MEM", () -> profile_margins(mb, db, cg(); type=:effects, backend)))
        push!(out, ("bench/AME (all)/$backend", "AME (all)",
                    () -> population_margins(mb, db; type=:effects, backend)))
        push!(out, ("bench/AME (age_h)/$backend", "AME (age_h)",
                    () -> population_margins(mb, db; type=:effects, vars=[:age_h], backend)))
        push!(out, ("bench/AME (scenario)/$backend", "AME (scenario)",
                    () -> population_margins(mb, db; type=:effects, vars=[:wealth_p, :wealth_h],
                                             scenarios=(dist_inv=[1/6],), backend)))
    end
    _coverage_cases!(out)
    _model_interface_cases!(out)
    _second_difference_cases!(out)
    return out
end

const FM = @formula(y ~ x * z + x * g + b + (1 | h))
const FFE = @formula(y ~ x * z + x * g + b + fe(h))
const FFE_S = @formula(y ~ x * z + x * g + b + fe(hs))
const ALLV = [:x, :z, :g, :b]

# Paths the 3.0 release changes that the matrix above does not reach (added 2026-10-19,
# before cache removal and the request/model-class refactors): models fitted with
# estimation weights (the `weights=:model` default), CovarianceMatrices estimators, more
# group/scenario forms, and mixed models with and without weights. Offset and
# FixedEffectModels cases are in `_model_interface_cases!` (added in step 5, with the fixes).
function _coverage_cases!(out)
    push_pop!(id, m, d; kw...) = begin
        # a variable set by a scenario or used by groups cannot also be a focal variable
        vars = ALLV
        haskey(kw, :scenarios) && (vars = setdiff(vars, keys(kw[:scenarios])))
        haskey(kw, :groups) && (vars = setdiff(vars, Margins._extract_group_variables(kw[:groups])))
        push!(out, ("$id/effect/ad", "population_effects",
                    () -> population_margins(m, d; type=:effects, vars, kw...)))
        push!(out, ("$id/predictions", "population_predictions",
                    () -> population_margins(m, d; type=:predictions, kw...)))
    end
    push_prof!(id, m, d; kw...) = begin
        push!(out, ("$id/profile:means_grid/effect/ad", "profile_effects",
                    () -> profile_margins(m, d, means_grid(d); type=:effects, vars=ALLV, kw...)))
        push!(out, ("$id/profile:means_grid/predictions", "profile_predictions",
                    () -> profile_margins(m, d, means_grid(d); type=:predictions, kw...)))
    end

    # Estimation weights: the default averages with them; :none is the 2.x behaviour
    for linkname in ("identity", "logit", "log")
        d = matrix_data(linkname)
        m = fit_matrix_model(linkname, d; wts=d.w)
        for (wname, wkw) in (("model", (;)), ("none", (; weights=:none)))
            for (pname, pkw) in (("population", (;)), ("groups=h", (; groups=:h)),
                                 ("groups=(z,3)", (; groups=(:z, 3))),
                                 ("scenarios z=0.5", (; scenarios=(z=0.5,))))
                push_pop!("wts:$linkname/$wname/$pname", m, d; pkw..., wkw...)
            end
        end
        push_prof!("wts:$linkname", m, d)
    end

    # CovarianceMatrices estimators: heteroskedasticity-robust and clustered
    for linkname in ("identity", "logit")
        d = matrix_data(linkname)
        m = fit_matrix_model(linkname, d)
        for (vname, V) in (("HC1", HC1()), ("CR1(h)", CR1(d.h)))
            for (pname, pkw) in (("population", (;)), ("groups=h", (; groups=:h)),
                                 ("scenarios z=0.5", (; scenarios=(z=0.5,))))
                push_pop!("vcov:$linkname/$vname/$pname", m, d; vcov=V, pkw...)
            end
            push_prof!("vcov:$linkname/$vname", m, d; vcov=V)
        end
    end

    # More group and scenario forms, unweighted and weighted
    d = matrix_data("logit")
    m = fit_matrix_model("logit", d)
    for (wname, wkw) in (("none", (;)), ("w", (; weights=:w)))
        for (pname, pkw) in (("groups=h+scenarios z=0.5", (; groups=:h, scenarios=(z=0.5,))),
                             ("groups=h=>(x,2)", (; groups=:h => (:x, 2))),
                             ("groups=(x,[1,2])", (; groups=(:x, [1.0, 2.0]))),
                             ("groups=[h,(x,2)]", (; groups=[:h, (:x, 2)])),
                             ("scenarios z×b", (; scenarios=(z=[0.0, 0.5], b=[false, true]))))
            push_pop!("ctx:logit/$wname/$pname", m, d; pkw..., wkw...)
        end
    end

    # Mixed models (fixed effects of FM), with and without estimation weights
    for (mname, fitf) in (
            ("lmm", (d; kw...) -> fit(MixedModel, FM, d; progress=false, kw...)),
            ("glmm-logit", (d; kw...) -> fit(MixedModel, FM, d, Bernoulli(); progress=false, kw...)))
        d = matrix_data(mname == "lmm" ? "identity" : "logit")
        for (wname, fkw) in (("unweighted", (;)), ("wts", (; wts=d.w)))
            m = fitf(d; fkw...)
            push_pop!("mixed:$mname/$wname/population", m, d)
            push_pop!("mixed:$mname/$wname/groups=g", m, d; groups=:g)
            push_prof!("mixed:$mname/$wname", m, d)
        end
    end
    return out
end

# Offsets and FixedEffectModels (added in step 5 of the 3.0 plan, after the fixes: offsets now
# enter η everywhere, and FixedEffectModels goes through the shared request path with its
# absorbed fixed effects as a per-row offset and its `esample`).
function _model_interface_cases!(out)
    push_pop!(id, m, d; vars=ALLV, kw...) = begin
        haskey(kw, :scenarios) && (vars = setdiff(vars, keys(kw[:scenarios])))
        haskey(kw, :groups) && (vars = setdiff(vars, Margins._extract_group_variables(kw[:groups])))
        push!(out, ("$id/effect/ad", "population_effects",
                    () -> population_margins(m, d; type=:effects, vars, kw...)))
        push!(out, ("$id/predictions", "population_predictions",
                    () -> population_margins(m, d; type=:predictions, kw...)))
    end

    # Poisson with an exposure offset (oracle: AAP = mean(predict(m)) = mean(y))
    d = matrix_data("log")
    i = 1:nrow(d)
    d.e = 0.5 .+ 2 .* _frac.(i .* 0.3819660112501051)
    off = log.(d.e)
    m = glm(F, d, Poisson(), LogLink(); offset=off)
    for scale in (:response, :link)
        for (pname, pkw) in (("population", (;)), ("groups=h", (; groups=:h)),
                             ("scenarios z=0.5", (; scenarios=(z=0.5,))), ("w", (; weights=:w)))
            push_pop!("offset:poisson/$scale/$pname", m, d; scale, pkw...)
        end
        for measure in (:elasticity, :semielasticity_eydx)
            push!(out, ("offset:poisson/$scale/population/$measure/ad", "population_effects",
                        () -> population_margins(m, d; type=:effects, vars=[:x, :z], scale, measure)))
        end
        push!(out, ("offset:poisson/$scale/profile:means_grid/effect/ad", "profile_effects",
                    () -> profile_margins(m, d, means_grid(d); type=:effects, vars=ALLV, scale, offset=log(2.0))))
        push!(out, ("offset:poisson/$scale/profile:cartesian_grid/predictions", "profile_predictions",
                    () -> profile_margins(m, d, cartesian_grid(x=[1.0, 2.0], g=["a", "c"]);
                                          type=:predictions, scale, offset=[0.0, 0.1, 0.2, 0.3])))
    end

    # GLMM (Poisson) with an offset
    gm = fit(MixedModel, FM, d, Poisson(); offset=off, progress=false)
    push_pop!("offset:glmm-poisson/population", gm, d)
    push_pop!("offset:glmm-poisson/groups=g", gm, d; groups=:g)
    push!(out, ("offset:glmm-poisson/profile:means_grid/predictions", "profile_predictions",
                () -> profile_margins(gm, d, means_grid(d); type=:predictions, offset=log(2.0))))

    # FixedEffectModels: the identity-link data with h absorbed; clustered on h
    d = matrix_data("identity")
    fm = reg(d, FFE, Vcov.cluster(:h); save=:fe)
    for (pname, pkw) in (("population", (; weights=:none)), ("groups=g", (; groups=:g, weights=:none)),
                         ("groups=(z,3)", (; groups=(:z, 3), weights=:none)),
                         ("scenarios z=0.5", (; scenarios=(z=0.5,), weights=:none)), ("w", (; weights=:w)))
        push_pop!("fem/$pname", fm, d; vars=[:x, :z, :g, :b], pkw...)
    end
    push!(out, ("fem/profile:means_grid/effect/ad", "profile_effects",
                () -> profile_margins(fm, d, means_grid(d); type=:effects, vars=[:x, :z, :g, :b])))
    push!(out, ("fem/profile:cartesian_grid/predictions", "profile_predictions",
                () -> profile_margins(fm, d, cartesian_grid(x=[1.0, 2.0], h=["p", "r"]); type=:predictions)))

    # An esample-dropping fit: a singleton level of the absorbed fixed effect
    ds = copy(d)
    ds.hs = string.(ds.h); ds.hs[1] = "singleton"
    fs = reg(ds, FFE_S; save=:fe)
    sum(fs.esample) == nrow(ds) - 1 || error("golden case: expected one singleton dropped")
    push_pop!("fem-esample/population", fs, ds; vars=[:x, :z, :g, :b], weights=:none)
    push_pop!("fem-esample/groups=g", fs, ds; vars=[:x, :z, :b], groups=:g, weights=:none)
    return out
end

# Second differences with `groups=` and a categorical variable that appears only inside an
# interaction (added in step 8 of the 3.0 plan, after checking them against the result-level route,
# a stratum-subset oracle and hand computations from `modelcols`; test_second_differences_groups.jl).
# A second-difference table is reported in the margins row format (`_sd_as_rows`): estimate =
# second_diff, keyed by the levels compared, the evaluation point and the strata/profile columns.
function _second_difference_cases!(out)
    d = matrix_data("logit")
    m = fit_matrix_model("logit", d)
    sd!(id, f) = push!(out, ("sd:logit/$id", "second_differences", () -> _sd_as_rows(f())))
    sd!("b/groups=h", () -> second_differences(m, d, [:x, :z, :g], :b; groups=:h))
    sd!("b/groups=h/profile g", () -> second_differences(m, d, [:x, :z], :b; groups=:h, profile=(g=["a", "c"],)))
    sd!("b/groups=[h,(z,2)]", () -> second_differences(m, d, [:x, :g], :b; groups=[:h, (:z, 2)]))
    sd!("b/groups=h=>g/w", () -> second_differences(m, d, [:x, :z], :b; groups=:h => :g, weights=:w))
    sd!("z/groups=h/observed", () -> second_differences(m, d, [:x, :g], :z; groups=:h))
    sd!("z/groups=h/observed/fd", () -> second_differences(m, d, [:x, :g], :z; groups=:h, backend=:fd))
    sd!("z/groups=[h,g]/mean", () -> second_differences(m, d, :x, :z; groups=[:h, :g], at=:mean))
    sd!("z/groups=h/median/w", () -> second_differences(m, d, :x, :z; groups=:h, at=:median, weights=:w))
    sd!("z/groups=(x,2)/at", () -> second_differences(m, d, [:x, :b], :z; groups=(:x, 2), at=[-0.5, 0.5]))
    sd!("z/groups=h/levels", () -> second_differences(m, d, :x, :z; groups=:h, levels=[-0.5, 0.0, 0.5]))

    # g only inside interactions (FullDummyCoding): AME of g, and g as focal variable and modifier
    mi = glm(@formula(y ~ x * z + g & x + g & z & x + b), d, Binomial(), LogitLink())
    for scale in (:response, :link)
        push!(out, ("interaction-only g/$scale/population/effect/ad", "population_effects",
                    () -> population_margins(mi, d; type=:effects, vars=[:g, :x, :b], scale)))
        push!(out, ("interaction-only g/$scale/groups=h/effect/ad", "population_effects",
                    () -> population_margins(mi, d; type=:effects, vars=[:g], groups=:h, scale)))
    end
    push!(out, ("interaction-only g/profile:means_grid/effect/ad", "profile_effects",
                () -> profile_margins(mi, d, means_grid(d); type=:effects, vars=[:g, :x])))
    push!(out, ("sd:interaction-only g/g × b", "second_differences",
                () -> _sd_as_rows(second_differences(mi, d, :g, :b))))
    push!(out, ("sd:interaction-only g/x × g", "second_differences",
                () -> _sd_as_rows(second_differences(mi, d, :x, :g; warn=false))))
    return out
end

# A second-difference table in the margins row format `rows` reads
function _sd_as_rows(sd::DataFrame)
    keycols = [c for c in names(sd) if startswith(c, "over_") ||
               !(c in ("variable", "contrast", "modifier", "modifier_type", "ame_at_level1", "ame_at_level2",
                       "second_diff", "se", "statistic", "df", "p_value", "ci_lower", "ci_upper",
                       "significant", "delta_used"))]
    out = select(sd, keycols)
    out.variable = string.(sd.variable)
    out.contrast = sd.contrast
    out.estimate = sd.second_diff
    out.se = sd.se
    out.df = sd.df
    return out
end

# ---------------------------------------------------------------------------------------
# Schema-robust row keys
# ---------------------------------------------------------------------------------------

# Columns that are statistics, not identifiers (current and planned 3.0 names)
const STAT_COLS = Set(["type", "variable", "contrast", "estimate", "se", "t_stat", "statistic",
                       "df", "p_value", "ci_lower", "ci_upper", "lower", "upper", "n"])

_normname(c) = replace(replace(String(c), r"^over_" => ""), r"^at_" => "")

# Numbers in keys are rounded to 10 significant digits: profile values such as means_grid
# means can differ in the last ulp between environments (e.g. Statistics versions), and
# a key only has to identify the row; the values themselves are compared at 1e-12.
_fmt(v::Bool) = string(v)
_fmt(v::Real) = string(round(Float64(v); sigdigits=10))
_fmt(v::AbstractString) = String(v)
_fmt(v::CategoricalValue) = string(unwrap(v))
_fmt(v::CategoricalMixture) = "mix(" * join(["$(_fmt(l)):$(_fmt(w))" for (l, w) in
                                             sort(collect(zip(v.levels, v.weights)); by=p -> string(p[1]))], ",") * ")"
_fmt(v) = string(v)

_normcontrast(c) = c in ("dy/dx", "derivative") ? "d" : strip(String(c))

"""DataFrame(case_id, operation, key, estimate, se, df) for one result (`df`: the inference
degrees of freedom, `Inf` for z; recorded since step 5 of the 3.0 plan, when df became a
property of the model class)."""
function rows(case_id, operation, result)
    df = result isa DataFrame ? result : DataFrame(result)
    ctx = sort([c for c in names(df) if !(c in STAT_COLS)]; by=_normname)
    keys = map(1:nrow(df)) do k
        parts = ["$(_normname(c))=$(_fmt(df[k, c]))" for c in ctx]
        if hasproperty(df, :variable)
            push!(parts, "var=$(df.variable[k])", "contrast=$(_normcontrast(df.contrast[k]))")
        end
        isempty(parts) ? "all" : join(parts, "|")
    end
    allunique(keys) || error("duplicate keys in case $case_id: $(keys[findall(k -> count(==(k), keys) > 1, keys)])")
    return DataFrame(case_id=fill(case_id, nrow(df)), operation=fill(operation, nrow(df)),
                     key=keys, estimate=Float64.(df.estimate), se=Float64.(df.se),
                     df=Float64.(df.df))
end

"""Run every case and return the stacked rows."""
function run_all()
    out = DataFrame(case_id=String[], operation=String[], key=String[], estimate=Float64[], se=Float64[], df=Float64[])
    for (cid, op, f) in cases()
        append!(out, rows(cid, op, f()))
    end
    return out
end

"""
What the snapshot's bits depend on besides Margins/FormulaCompiler: the versions that
change the fitted β and the data handling, and the bounds-checking mode. `Pkg.test` runs
with `--check-bounds=yes`, which disables `@inbounds`/SIMD reductions and so changes
summation order (e.g. the means in `means_grid`) at the ulp level.
"""
function fingerprint()
    vers = join(["$(nameof(M))=$(pkgversion(M))" for M in (GLM, StatsModels, CategoricalArrays,
                                                         MixedModels, CovarianceMatrices, FixedEffectModels)], ";")
    return vers * ";check_bounds=" * string(Base.JLOptions().check_bounds)
end

end # module
