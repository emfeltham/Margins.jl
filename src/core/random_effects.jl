# core/random_effects.jl
# The `random_effects` option of mixed models (Margins 3.1, phase 1 of
# notes/RE_OPTIONS_PLAN.md in the paper repository).
#
# A mixed model's linear predictor is η = x'β + z'b. FormulaCompiler compiles only x'β
# (`fixed_effects_form` drops the random-effects terms), so by default every random effect is
# 0: the estimates describe a typical unit (a rater and a village whose random effects are 0).
# `random_effects = :conditional` adds each row's predicted random effects (the conditional
# modes) to η, z'b̂: through a wrapper model whose formula carries z'b as extra coefficients
# (β, b̂) with the joint covariance (MixedModels extension), or, above its size limit, as a per-row
# η offset (the path FixedEffectModels' absorbed effects and GLM offsets use). The estimates then
# describe the units in the data.
# `random_effects = :marginal` averages the inverse link over the random effects: zᵢ'b is normal,
# N(0, τ²) (with random intercepts only, τ² is the sum of their variances, however many factors
# are crossed or nested), so ŷᵢ = E[g⁻¹(ηᵢ + τZ)] is a one-dimensional Gauss–Hermite integral.
# It is done in the scale (`AveragedResponseScale`), so every kernel gets population-averaged
# quantities and their gradients without change.
#
# Model classes opt in with methods of the generics below (MixedModels: `ext/MixedModelsExt.jl`).
#
# | generic | meaning | default |
# |:--|:--|:--|
# | `_supports_random_effects(model)` | does the model have random effects? | `false` |
# | `_re_factors(model)` | the grouping factors, as `fnames` names them | `Symbol[]` |
# | `_re_grouping_vars(model, factors)` | the data columns that define those factors | `Symbol[]` |
# | `_re_slope_vars(model, factors)` | the variables in their random-effects model matrix Z | `Symbol[]` |
# | `_re_offset(model, table, spec)` | z'b̂ at each row of `table`, and the factors it used | error |
# | `_re_intercept_variance(model, factors)` | Σ of the random-intercept variances (`:marginal`) | error |

"""
    REAveraging

The quadrature of `random_effects = :marginal` (internal): E[f(η + u)], u ~ N(0, τ²), is
Σₖ wₖ f(η + τzₖ), with (zₖ, wₖ) the Gauss–Hermite rule for N(0, 1). `τ` is the SD of u (with
random slopes, where it varies by row, the largest it can take in the request, which the rule is
checked at; each row's is computed from its model row, see `RowSpreadOffset`), `nodes` the
rule's size before nodes of negligible weight are dropped.
"""
struct REAveraging
    τ::Float64
    nodes::Int
    z::Vector{Float64}
    w::Vector{Float64}
end

"""
    RowSpreadOffset(shift, n, pos, Σ, τmax)

The η offset of `random_effects = :marginal` with random slopes (internal): `shift` is the
model's own offset (`nothing` or one value per row, `n` rows). The random-effects model columns
of the integrated factors are columns of the compiled formula (coefficient 0, variance 0; the
request's model is the mixed model's wrapper), so u = zᵢ'b has SD τᵢ = √(zᵢ'Σzᵢ) with zᵢ read from
the evaluated model row: `pos[a]` is the column of element a of z (0 for a constant 1, a random
intercept) and Σ the block-diagonal covariance of b over the integrated factors. τ therefore
follows every counterfactual row (scenarios, contrasts, grid rows), and ∂τᵢ/∂w = zᵢ'Σ(∂zᵢ/∂w)/τᵢ
comes from the derivative evaluator's Jacobian rows of those columns. `linear_predictor` returns a
`SpreadEta`, which the averaged scale integrates with that row's τ.

`τmax` is the largest τ the quadrature rule was checked at (`Inf` for an explicit `re_nodes`): a
row above it (a counterfactual value not known when the rule was chosen) is an error, never a
silently less accurate average.
"""
struct RowSpreadOffset{O<:Union{Nothing,Vector{Float64}}}
    shift::O
    n::Int
    pos::Vector{Int}
    Σ::Matrix{Float64}
    τmax::Float64
end
Base.length(o::RowSpreadOffset) = o.n
Base.getindex(o::RowSpreadOffset, rows::AbstractVector{<:Integer}) =
    RowSpreadOffset(isnothing(o.shift) ? nothing : o.shift[rows], length(rows), o.pos, o.Σ, o.τmax)

@inline _zval(p::Int, x) = p == 0 ? 1.0 : @inbounds x[p]

# τᵢ = √(zᵢ'Σzᵢ) of the model row x
@inline function _row_tau(o::RowSpreadOffset, x::AbstractVector{Float64})
    acc = 0.0
    @inbounds for a in eachindex(o.pos)
        za = _zval(o.pos[a], x)
        for b in eachindex(o.pos)
            acc += za * o.Σ[a, b] * _zval(o.pos[b], x)
        end
    end
    τ = sqrt(max(acc, 0.0))
    τ > o.τmax * (1 + 1e-8) + 1e-12 && _throw_tau_above_rule(τ, o.τmax)
    return τ
end

# ∂τᵢ/∂(variable j) = zᵢ'Σ(∂zᵢ/∂x_j)/τᵢ, with ∂zᵢ/∂x_j the Jacobian rows J[pos, j] (0 where τᵢ = 0)
@inline function _row_dtau(o::RowSpreadOffset, x::AbstractVector{Float64}, J::AbstractMatrix{Float64},
                           j::Int, τ::Float64)
    τ > 0 || return 0.0
    acc = 0.0
    @inbounds for b in eachindex(o.pos)
        pb = o.pos[b]
        pb == 0 && continue
        dz = J[pb, j]
        iszero(dz) && continue
        s = 0.0
        for a in eachindex(o.pos)
            s += _zval(o.pos[a], x) * o.Σ[a, b]
        end
        acc += s * dz
    end
    return acc / τ
end

@noinline _throw_tau_above_rule(τ, τmax) = throw(MarginsError(
    "random_effects = :marginal: a row evaluated at a counterfactual value has SD of z'b = " *
    "$(round(τ; sigdigits=4)), above the largest ($(round(τmax; sigdigits=4))) the automatic " *
    "quadrature rule was checked at. Pass re_nodes = <an integer> (e.g. 256) to fix the rule."))

# η with its row's random-effects SD (see `RowSpreadOffset`)
struct SpreadEta
    η::Float64
    τ::Float64
end

"""
    SlopedOffset(base, vars, doff)

The η offset of a `:conditional` request for the effect of a variable with a random slope on the
offset path (internal; `re_se = :fixed` above the wrapper's limit). `base` is the offset itself
(`nothing` or a vector, z'b̂ included). For each variable `vars[k]` and row i, `doff[i, k]` =
∂oᵢ/∂vars[k] = ∂(zᵢ'b̂)/∂w, each group's slope; the continuous accumulator adds d₁·doff to ∂ŷ/∂w
and the matching term to its gradient.
"""
struct SlopedOffset{O}
    base::O
    vars::Vector{Symbol}
    doff::Matrix{Float64}
end
Base.length(o::SlopedOffset) = size(o.doff, 1)
Base.getindex(o::SlopedOffset, rows::AbstractVector{<:Integer}) =
    SlopedOffset(isnothing(o.base) ? nothing : o.base[rows], o.vars, o.doff[rows, :])

# Every form of η offset a request or engine can carry
const AnyOffset = Union{Nothing, Vector{Float64}, RowSpreadOffset, SlopedOffset}

"""
    RandomEffectsSpec

The normalized `random_effects` options of a request on a mixed model (internal), `nothing` on
the request for the default `random_effects = :zero`.

| field | meaning |
|:--|:--|
| `mode` | `:conditional`, `:marginal`, or `:hybrid` (a NamedTuple `random_effects` with both: some factors conditioned on, others averaged over) |
| `new_levels` | `re_new`: `:zero` (a level absent from the fit gets random effects 0) or `:error` (`:conditional` only) |
| `factors` | `:conditional` / `:hybrid`: the factors conditioned on; `:marginal`: those averaged over (`re_terms`, default all). The others are 0 (or, `:hybrid`, in `integrated`) |
| `integrated` | `:hybrid`: the factors averaged over; empty otherwise |
| `used` | the factors that entered: `:conditional` population requests and `:marginal`, all of `factors`; `:conditional` profile requests, those whose grouping columns the reference grid sets (the others are 0) |
| `average` | `:marginal`: the quadrature over u = z'b (`REAveraging`), or `nothing` when it changes nothing (identity link, or τ = 0) or is not yet known (set by `_finalize_marginal`) |
| `nodes` | the `re_nodes` keyword |
| `se` | `:conditional`: `:joint` (standard errors include the prediction error of b̂) or `:fixed` (β only); `:none` otherwise |
| `wrapped` | the request's model is the mixed model's wrapper: `:conditional` with `:joint`, or with `:fixed` when the conditioned factors have at most the wrapper's limit of random-effect coefficients (else the offset path); `:marginal` with random slopes and a nonlinear link (z's columns in the formula, see `RowSpreadOffset`) |
"""
struct RandomEffectsSpec
    mode::Symbol
    new_levels::Symbol
    factors::Vector{Symbol}
    integrated::Vector{Symbol}
    used::Vector{Symbol}
    average::Union{Nothing,REAveraging}
    nodes::Union{Symbol,Int}
    se::Symbol
    wrapped::Bool
end

_with_used(spec::RandomEffectsSpec, used::Vector{Symbol}) =
    RandomEffectsSpec(spec.mode, spec.new_levels, spec.factors, spec.integrated, used, spec.average,
                      spec.nodes, spec.se, spec.wrapped)
_with_average(spec::RandomEffectsSpec, a) =
    RandomEffectsSpec(spec.mode, spec.new_levels, spec.factors, spec.integrated, spec.used, a,
                      spec.nodes, spec.se, spec.wrapped)

# The factors a spec conditions on (their b̂ enter η) and those it averages over
_conditioned(spec::RandomEffectsSpec) = spec.mode === :marginal ? Symbol[] : spec.factors
_integrated(spec::RandomEffectsSpec) =
    spec.mode === :marginal ? spec.factors : spec.mode === :hybrid ? spec.integrated : Symbol[]
_conditions(spec) = !isnothing(spec) && spec.mode in (:conditional, :hybrid)
_integrates(spec) = !isnothing(spec) && spec.mode in (:marginal, :hybrid)

# The request's model is the mixed model's wrapper (`REWrapperModel`, MixedModels extension), whose
# formula evaluates z (`:conditional`: z'b̂ as coefficients; `:marginal`: z's columns for τ), so
# random-slope variables need nothing special
_is_wrapped(spec) = !isnothing(spec) && spec.wrapped

const _RANDOM_EFFECTS_MODES = (:zero, :conditional, :marginal)

_supports_random_effects(model) = false
_supports_random_effects(model::StatsModels.TableRegressionModel) = _supports_random_effects(model.model)
_re_factors(model) = Symbol[]
_re_grouping_vars(model, factors) = Symbol[]
_re_slope_vars(model, factors) = Symbol[]
_re_offset(model, table::NamedTuple, spec::RandomEffectsSpec) =
    error("random_effects = :$(spec.mode) is not implemented for $(typeof(model))")
_re_intercept_variance(model, factors) =
    error("random_effects = :marginal is not implemented for $(typeof(model))")
_re_has_slopes(model, factors) = false
_re_spread_sq(model, table::NamedTuple, factors) =
    error("random_effects = :marginal with random slopes is not implemented for $(typeof(model))")

"""
    _validate_random_effects(model, random_effects, re_new, re_terms, re_nodes) -> Union{Nothing, RandomEffectsSpec}

The `random_effects`, `re_new`, `re_terms` and `re_nodes` keywords: `nothing` for
`random_effects = :zero` (every random effect 0, the behaviour of every model class), else the
options of a conditional or marginal request. A keyword that would change nothing is rejected
rather than silently ignored: `re_terms` without `:conditional` or `:marginal`, `re_new` without
`:conditional`, `re_nodes` without `:marginal`.
"""
function _validate_random_effects(model, random_effects, re_new, re_terms, re_nodes=:auto, re_se=:auto)
    # The NamedTuple form, one mode per grouping factor (unlisted factors: :zero): a single mode
    # other than :zero is that mode with `re_terms`; :conditional and :marginal together is :hybrid
    cond_nt = Symbol[]; int_nt = Symbol[]
    if random_effects isa NamedTuple
        isnothing(re_terms) || throw(ArgumentError(
            "re_terms cannot be combined with random_effects given per factor (a NamedTuple): its " *
            "keys already name the factors"))
        _supports_random_effects(model) || throw(ArgumentError(
            "random_effects given per grouping factor applies only to models with random effects " *
            "(MixedModels.jl); a $(nameof(typeof(model))) has none"))
        all_f = _re_factors(model)
        bad_keys = setdiff(collect(Symbol, keys(random_effects)), all_f)
        isempty(bad_keys) || throw(ArgumentError(
            "random_effects: $(join(repr.(bad_keys), ", ")) $(length(bad_keys) == 1 ? "is not a grouping factor" : "are not grouping factors") " *
            "of the model; its grouping factors are $(join(repr.(all_f), ", "))"))
        for (f, mode) in pairs(random_effects)
            (mode isa Symbol && mode in _RANDOM_EFFECTS_MODES) || throw(ArgumentError(
                "random_effects.$f must be :zero, :conditional or :marginal, got $(repr(mode))"))
            mode === :conditional && push!(cond_nt, f)
            mode === :marginal && push!(int_nt, f)
        end
        random_effects = isempty(cond_nt) && isempty(int_nt) ? :zero :
                         isempty(int_nt) ? :conditional : isempty(cond_nt) ? :marginal : :hybrid
        random_effects === :conditional && (re_terms = cond_nt)
        random_effects === :marginal && (re_terms = int_nt)
        random_effects === :hybrid && (re_terms = cond_nt)
    elseif !(random_effects isa Symbol && random_effects in _RANDOM_EFFECTS_MODES)
        throw(ArgumentError(
            "random_effects must be :zero, :conditional, :marginal or a NamedTuple of those per " *
            "grouping factor (e.g. (village = :conditional, rater = :marginal)), got $(repr(random_effects))"))
    end
    conditions = random_effects in (:conditional, :hybrid)
    integrates = random_effects in (:marginal, :hybrid)
    (re_new isa Symbol && re_new in (:zero, :error)) || throw(ArgumentError(
        "re_new must be :zero or :error, got $(repr(re_new))"))
    (re_nodes === :auto || (re_nodes isa Integer && re_nodes >= 2)) || throw(ArgumentError(
        "re_nodes must be :auto or an integer ≥ 2 (Gauss–Hermite nodes), got $(repr(re_nodes))"))
    conditions || re_new === :zero || throw(ArgumentError(
        "re_new applies only with random_effects = :conditional"))
    random_effects === :zero && !isnothing(re_terms) && throw(ArgumentError(
        "re_terms applies only with random_effects = :conditional or :marginal"))
    integrates || re_nodes === :auto || throw(ArgumentError(
        "re_nodes applies only with random_effects = :marginal"))
    re_se in (:auto, :joint, :fixed) || throw(ArgumentError(
        "re_se must be :joint or :fixed, got $(repr(re_se))"))
    conditions || re_se === :auto || throw(ArgumentError(
        "re_se applies only with random_effects = :conditional (no predicted random effects enter " *
        ":zero or :marginal estimates)"))
    random_effects === :zero && return nothing
    _supports_random_effects(model) || throw(ArgumentError(
        "random_effects = $(repr(random_effects)) applies only to models with random effects " *
        "(MixedModels.jl); a $(nameof(typeof(model))) has none"))
    all_factors = _re_factors(model)
    bad_nt = setdiff([cond_nt; int_nt], all_factors)
    isempty(bad_nt) || throw(ArgumentError(
        "random_effects: $(join(repr.(bad_nt), ", ")) $(length(bad_nt) == 1 ? "is not a grouping factor" : "are not grouping factors") " *
        "of the model; its grouping factors are $(join(repr.(all_factors), ", "))"))
    factors = if isnothing(re_terms)
        copy(all_factors)
    else
        req_factors = re_terms isa Symbol ? [re_terms] :
            (re_terms isa Union{Tuple,AbstractVector} && all(t -> t isa Symbol, re_terms)) ? collect(Symbol, re_terms) :
            throw(ArgumentError("re_terms must be a Symbol or a collection of Symbols, got $(repr(re_terms))"))
        isempty(req_factors) && throw(ArgumentError(
            "re_terms is empty; use random_effects = :zero to set every random effect to 0"))
        bad = setdiff(req_factors, all_factors)
        isempty(bad) || throw(ArgumentError(
            "re_terms: $(join(repr.(bad), ", ")) $(length(bad) == 1 ? "is not a grouping factor" : "are not grouping factors") " *
            "of the model; its grouping factors are $(join(repr.(all_factors), ", "))"))
        unique(req_factors)
    end
    integrated = random_effects === :hybrid ? unique(int_nt) : Symbol[]
    se = conditions ? (re_se === :fixed ? :fixed : :joint) : :none
    wrap_fixed = se === :fixed && _re_wrap_fixed(model, factors)
    if random_effects === :hybrid && se === :fixed && !wrap_fixed
        throw(ArgumentError(
            "random_effects with both :conditional and :marginal factors needs the wrapper model, whose " *
            "limit of 2000 random-effect coefficients the conditioned factors exceed"))
    end
    wrapped = se === :joint || wrap_fixed ||
              (random_effects === :marginal && _re_has_slopes(model, factors) && !(_link(model) isa GLM.IdentityLink))
    return RandomEffectsSpec(random_effects, re_new, factors, integrated, copy(factors), nothing,
                             re_nodes isa Integer ? Int(re_nodes) : re_nodes, se, wrapped)
end

# ---- random_effects = :marginal: Gauss–Hermite averaging over u = z'b ~ N(0, τ²) -------------

"""
    _gauss_hermite_normal(K) -> (z, w)

The K-point Gauss–Hermite rule for the standard normal (probabilists' Hermite polynomials):
Σₖ wₖ f(zₖ) ≈ E[f(Z)], exact for polynomials of degree ≤ 2K − 1. Golub–Welsch: the nodes are
the eigenvalues of the Jacobi matrix (zero diagonal, off-diagonal √k), the weights the squared
first components of its eigenvectors.
"""
function _gauss_hermite_normal(K::Int)
    J = LinearAlgebra.SymTridiagonal(zeros(K), sqrt.(Float64.(1:K-1)))
    F = LinearAlgebra.eigen(J)
    w = abs2.(F.vectors[1, :])
    return (F.values, w ./ sum(w))
end

# Drop nodes whose weight is negligible next to the largest (the tails of a large rule); the
# convergence check of `:auto` is made on the pruned rule, so pruning cannot hide an error
function _pruned(z, w; rel=1e-17)
    keep = w .>= rel * maximum(w)
    return (z[keep], w[keep])
end

const _RE_AUTO_NODES = (16, 32, 64, 128, 256, 512, 1024)
const _RE_AUTO_GRID = collect(range(-20.0, 20.0; length=401))
const _RE_RTOL = 1e-10
const _RE_ATOL = 1e-13

# The averaged (μ, dμ/dη, d²μ/dη²) at each η of the grid, for rule (τz, w)
function _averaged_on_grid(link, τ, z, w)
    s = AveragedResponseScale(link, z, w, τ)
    return [yhat2(s, η) for η in _RE_AUTO_GRID]
end

_close(a::NTuple{3,Float64}, b::NTuple{3,Float64}) =
    all(abs(a[i] - b[i]) <= _RE_RTOL * abs(b[i]) + _RE_ATOL for i in 1:3)

"""
    _re_rule(link, τ, re_nodes) -> REAveraging

The quadrature for u ~ N(0, τ²). An integer `re_nodes` is that Gauss–Hermite rule, unpruned.
`:auto` doubles the rule from 16 nodes until the averaged ŷ and its first two η-derivatives agree
with the next rule to a relative 1e-10 (absolute 1e-13) at every η in [−20, 20]. The integrand is
a function of η alone, so this check covers every row of every call, whatever the data; it costs
a few milliseconds. With random slopes τ varies by row and the rule is checked at the largest
(a larger τ needs more nodes). No convergence by 1024 nodes is an error (it would need τ far beyond any
fitted model's).
"""
function _re_rule(link, τ::Float64, re_nodes)
    if re_nodes isa Integer
        (z, w) = _gauss_hermite_normal(Int(re_nodes))
        return REAveraging(τ, Int(re_nodes), z, w)
    end
    (z, w) = _pruned(_gauss_hermite_normal(first(_RE_AUTO_NODES))...)
    prev = _averaged_on_grid(link, τ, z, w)
    for K in _RE_AUTO_NODES[2:end]
        (z2, w2) = _pruned(_gauss_hermite_normal(K)...)
        cur = _averaged_on_grid(link, τ, z2, w2)
        if all(_close.(prev, cur))
            return REAveraging(τ, K ÷ 2, z, w)
        end
        (z, w, prev) = (z2, w2, cur)
    end
    throw(MarginsError(
        "random_effects = :marginal: the Gauss–Hermite average over the random effects (SD $τ) " *
        "did not converge with up to $(last(_RE_AUTO_NODES)) nodes. Pass re_nodes=<n> to use a fixed rule."))
end

"""
    _finalize_marginal(model, spec, table, settings, shift) -> (spec, offset)

The averaging of `random_effects = :marginal` (or the averaged factors of `:hybrid`) over
`_integrated(spec)` (the others are 0 or conditioned on), at the rows
of `table` (the data, or the completed reference grid); `shift` is the request's η offset so far.
With random intercepts only, u = z'b has one SD τ for every row and the offset is `shift`. With
random slopes (the request's model is then the wrapper) τᵢ is computed from each evaluated model row (`RowSpreadOffset`),
and the rule is chosen for the largest τ over the rows of `table` under each counterfactual
setting in `settings` (`_marginal_settings`). `spec.average` is `nothing` when the average cannot
change any estimate: an identity link (E[η + u] = η, so `:marginal` = `:zero`) or τ = 0 at every row.
"""
function _finalize_marginal(model, spec::RandomEffectsSpec, table::NamedTuple, settings, shift)
    _link(model) isa GLM.IdentityLink && return (spec, shift)
    ints = _integrated(spec)
    if _re_has_slopes(model, ints)
        τmax = _marginal_tau_max(model, spec, table, settings)
        iszero(τmax) && return (spec, shift)
        spec = _with_average(spec, _re_rule(_link(model), τmax, spec.nodes))
        guard = spec.nodes isa Int ? Inf : τmax
        return (spec, _re_row_spread(model, shift, length(first(table)), guard))
    end
    τ² = _re_intercept_variance(model, ints)
    iszero(τ²) && return (spec, shift)
    return (_with_average(spec, _re_rule(_link(model), sqrt(τ²), spec.nodes)), shift)
end

# The largest SD of z'b over the rows of `table` under each setting
function _marginal_tau_max(model, spec::RandomEffectsSpec, table::NamedTuple, settings)
    τ² = 0.0
    for s in settings
        # (a grid column holding mixtures takes plain levels too: widen it before setting)
        base = isempty(s) ? table : _widen_mixture_columns(table, keys(s))
        t = isempty(s) ? table : _apply_counterfactuals(base, s)
        τ² = max(τ², maximum(_re_spread_sq(model, t, _integrated(spec)); init=0.0))
    end
    return sqrt(τ²)
end

function _widen_mixture_columns(table::NamedTuple, vars)
    ks = Tuple(v for v in vars if haskey(table, v) && eltype(table[v]) <: CategoricalMixture)
    isempty(ks) && return table
    return merge(table, NamedTuple{ks}(Tuple(Vector{Any}(table[v]) for v in ks)))
end

"""
    _marginal_with_scenarios(req, specs) -> (spec, offset)

A population request's `random_effects` spec and offset when its scenarios are replaced by
`specs` after it was built (the model-level second differences set their evaluation points, or
shift the modifier, this way): with random slopes under `:marginal` and an automatic rule, the
rule is chosen again if the new settings take the SD of z'b above the one it was checked at.
"""
function _marginal_with_scenarios(model, spec, offset, specs, vars, table::NamedTuple)
    (offset isa RowSpreadOffset && isfinite(offset.τmax)) || return (spec, offset)
    τmax = _marginal_tau_max(model, spec, table, _marginal_settings(model, spec, specs, vars, table))
    τmax <= offset.τmax && return (spec, offset)
    spec = _with_average(spec, _re_rule(_link(model), τmax, spec.nodes))
    return (spec, RowSpreadOffset(offset.shift, offset.n, offset.pos, offset.Σ, τmax))
end

"""
    _marginal_settings(model, spec, scenario_specs, vars, table) -> Vector{Dict}

The counterfactual settings of the random-slope variables a `:marginal` request evaluates, known
when it is built: each scenario (its slope variables), crossed with every level of each
categorical focal variable with a random slope (its contrasts evaluate the rows at each level).
The empty setting is the rows as they are. The levels come from `level_table` (the data; for a
profile request `table` is the completed grid). Scenarios set later go through
`_marginal_with_scenarios`; `RowSpreadOffset`'s `τmax` guards anything else.
"""
function _marginal_settings(model, spec::RandomEffectsSpec, scenario_specs, vars, table::NamedTuple,
                            level_table::NamedTuple=table)
    slopes = _re_slope_vars(model, _integrated(spec))
    S = Dict{Symbol,Any}[Dict{Symbol,Any}()]
    for s in scenario_specs
        d = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(s) if Symbol(k) in slopes)
        isempty(d) || push!(S, d)
    end
    # the levels a categorical contrast evaluates: those of the data (`level_table`), not of the
    # grid rows, which may hold one level or a mixture
    for v in (isnothing(vars) ? Symbol[] : intersect(vars, slopes))
        col = haskey(level_table, v) ? level_table[v] : table[v]
        _is_continuous_column(col) && continue
        lv = col isa CategoricalArray ? levels(col) : unique(col)
        S = [merge(s, Dict{Symbol,Any}(v => l)) for s in S for l in lv]
    end
    return S
end

"""
    _validate_random_effects_request(model, spec, type, vars, vars_explicit, changed, grid, data_nt)

The cases not supported yet, rejected before any computation:

- the effect of a categorical variable with a random slope (`t` in `(1 + t | rater)`): its
  contrast changes z, which the offset is computed from. Continuous random-slope variables are
  supported (their slope terms are in a `SlopedOffset`);
- a variable with a random slope set counterfactually in a population request (`scenarios`, or a
  second-differences modifier): z'b̂ and the SD of z'b are computed from the observed values. A
  profile grid may set such a variable (z is evaluated at the grid rows).
"""
function _validate_random_effects_request(model, spec::RandomEffectsSpec, type::Symbol, vars,
                                          vars_explicit::Bool, changed::Vector{Symbol}, grid,
                                          data_nt::NamedTuple)
    # (`:marginal` with random slopes: τ depends on the slope variables, held at the row's values)
    spec.mode === :zero && return nothing
    spec.mode === :marginal && isnothing(spec.average) && return nothing   # = :zero (identity link)
    _is_wrapped(spec) && return nothing   # z is evaluated by the compiled formula: nothing to restrict
    slopes = _re_slope_vars(model, spec.factors)
    isempty(slopes) && return nothing
    if type === :effects && !isnothing(vars)
        bad = [v for v in intersect(vars, slopes) if !_is_continuous_column(data_nt[v])]
        if !isempty(bad)
            names_str = join((":$v" for v in bad), ", ")
            msg = "With random_effects = :$(spec.mode), the contrasts of $names_str (categorical, " *
                  "with a random slope) are not supported yet: a contrast changes the random-effects " *
                  "model matrix. Use random_effects = :zero (the effect for a group whose random " *
                  "effects are 0), or re_terms with only the grouping factors without that slope."
            if !vars_explicit
                rest = setdiff(vars, bad)
                msg *= isempty(rest) ? "" :
                       "\nOr pass the other variables explicitly: vars = [" * join((":$v" for v in rest), ", ") * "]"
            end
            throw(MarginsError(msg))
        end
    end
    set_by_grid = isnothing(grid) ? Symbol[] : Symbol.(names(grid))
    bad = intersect(setdiff(changed, set_by_grid), slopes)
    isempty(bad) || throw(MarginsError(
        "With random_effects = :$(spec.mode), $(join((":$v" for v in bad), ", ")) cannot be set " *
        "counterfactually: $(length(bad) == 1 ? "it has" : "they have") a random slope, and the random-effects " *
        "contribution is computed from the observed values. Use a reference grid " *
        "(profile_margins), random_effects = :zero, or re_terms without that slope."))
    return nothing
end

"""
    _profile_re_table(model, completed, grid, spec) -> (table, used)

The columns `_re_offset` reads at the reference-grid rows: the completed grid's (the variables of
Z), plus the grouping columns the user's grid sets. A factor whose grouping columns the grid does
not set gets random effects 0 at every row (a typical rater), and is left out of `used`.
"""
function _profile_re_table(model, completed::DataFrame, grid::DataFrame, spec::RandomEffectsSpec)
    used = Symbol[f for f in spec.factors
                  if all(v -> string(v) in names(grid), _re_grouping_vars(model, [f]))]
    tbl = Tables.columntable(completed)
    extra = Symbol[]
    for v in _re_grouping_vars(model, used)
        haskey(tbl, v) || push!(extra, v)
    end
    isempty(extra) || (tbl = merge(tbl, NamedTuple{Tuple(extra)}(Tuple(grid[!, string(v)] for v in extra))))
    return (tbl, used)
end

# A population request under `:conditional` needs the grouping columns of the conditioned factors at
# every row (without them the random effects would be 0, not the predicted ones)
function _check_re_grouping_present(model, data_nt::NamedTuple, factors)
    for f in factors
        absent = [v for v in _re_grouping_vars(model, [f]) if !haskey(data_nt, v)]
        isempty(absent) || throw(ArgumentError(
            "random_effects = :conditional needs $(join(repr.(absent), ", ")) at every row (the random " *
            "effects of $(repr(f)) depend on $(length(absent) == 1 ? "it" : "them")); it is not a column of the data"))
    end
    return nothing
end

# The grouping columns a reference grid may set under `random_effects = :conditional` (they are
# not predictors of the fixed-effects formula, so the grid check would reject them otherwise)
_re_grid_vars(model, ::Nothing) = Symbol[]
_re_grid_vars(model, spec::RandomEffectsSpec) =
    _conditions(spec) ? [_re_grouping_vars(model, spec.factors); _hidden_columns(model)] : Symbol[]

# The wrapper model of a `spec.wrapped` request, and the data with its hidden columns (MixedModels
# extension)
_re_wrapper_model(model, cols, spec, vcov) =
    error("random_effects = :$(spec.mode) through a wrapper model is not implemented for $(typeof(model))")
# The `RowSpreadOffset` of a `:marginal` wrapper
_re_row_spread(model, shift, n, τmax) =
    error("random_effects = :marginal with random slopes is not implemented for $(typeof(model))")
_augment_grid(model, grid, new_levels::Symbol) = grid
# A lightweight formula carrier for slope-only variables on the scalable offset path.
_re_offset_model(model, cols, spec, vcov) = (model, cols)
# Whether `re_se = :fixed` can use the wrapper (its random-effect coefficients within the limit)
_re_wrap_fixed(model, factors) = false
_hidden_columns(model) = Symbol[]
_joint_used(model, grid) = Symbol[]

"""
    _set_random_effects_metadata!(metadata, model, spec)

Record the random-effects treatment on a result: `:random_effects` (`:zero`, `:conditional`,
`:marginal` or `:hybrid`) and `:re_terms`, the grouping factors whose random effects entered η,
predicted or averaged over (empty for `:zero`); `:re_averaged` (`:hybrid`), `:re_se` (when b̂
enters), `:re_sd` and `:re_nodes` (when averaging). For a model with random effects, also `:re_factors`, all its grouping factors (the
printed result names those set to 0).
"""
function _set_random_effects_metadata!(metadata::AbstractDict, model, spec)
    _supports_random_effects(model) || return metadata
    metadata[:re_factors] = _re_factors(model)
    isnothing(spec) && return metadata
    metadata[:random_effects] = spec.mode
    metadata[:re_terms] = unique!([spec.used; spec.integrated])
    spec.mode === :hybrid && (metadata[:re_averaged] = copy(spec.integrated))
    _conditions(spec) && (metadata[:re_se] = spec.se)
    if _integrates(spec)
        a = spec.average
        metadata[:re_sd] = isnothing(a) ? 0.0 : a.τ
        metadata[:re_nodes] = isnothing(a) ? 0 : a.nodes
    end
    return metadata
end

# The engine's quadrature for a request (`nothing` unless `random_effects = :marginal` changes
# something)
_re_average(::Nothing) = nothing
_re_average(spec::RandomEffectsSpec) = spec.average

# A column that has a derivative (continuous), as opposed to contrasts
_is_continuous_column(col) = col isa AbstractVector{<:Real} && !(eltype(col) <: Bool)

"""
    _re_active_slope_vars(model, spec) -> Vector{Symbol}

The random-slope variables whose slope terms enter the request's effects through a
`SlopedOffset`: on the `:conditional` offset path, those of the factors used (a profile grid may
leave a factor at 0). None on a wrapper (z is in the compiled formula) or under `:marginal`
(random slopes are on the wrapper; with random intercepts only, nothing depends on a variable).
"""
_re_active_slope_vars(model, spec::RandomEffectsSpec) =
    _is_wrapped(spec) ? Symbol[] :
    spec.mode === :conditional ? _re_slope_vars(model, spec.used) :
    Symbol[]

_re_slope_derivatives(model, table, spec, vars) =
    error("random-slope effects are not implemented for $(typeof(model))")

