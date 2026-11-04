# core/random_effects.jl
# The `random_effects` option of mixed models (Margins 3.1, phase 1 of
# notes/RE_OPTIONS_PLAN.md in the paper repository).
#
# A mixed model's linear predictor is η = x'β + z'b. FormulaCompiler compiles only x'β
# (`fixed_effects_form` drops the random-effects terms), so by default every random effect is
# 0: the estimates describe a typical unit (a rater and a village whose random effects are 0).
# `random_effects = :conditional` adds each row's predicted random effects (the conditional
# modes, `ranef`) as a per-row η offset, z'b̂, through the offset path FixedEffectModels' absorbed
# effects and GLM offsets use. The estimates then describe the units in the data.
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

"""
    RandomEffectsSpec

The normalized `random_effects` options of a request on a mixed model (internal), `nothing` on
the request for the default `random_effects = :zero`.

| field | meaning |
|:--|:--|
| `mode` | `:conditional` (the only mode besides `:zero` so far) |
| `new_levels` | `re_new`: `:zero` (a level absent from the fit gets random effects 0) or `:error` |
| `factors` | the grouping factors whose predicted random effects enter η (`re_terms`, default all) |
| `used` | the factors that entered: all of `factors` for population requests; for profile requests, those whose grouping columns the reference grid sets (the others are 0) |
"""
struct RandomEffectsSpec
    mode::Symbol
    new_levels::Symbol
    factors::Vector{Symbol}
    used::Vector{Symbol}
end

_with_used(spec::RandomEffectsSpec, used::Vector{Symbol}) =
    RandomEffectsSpec(spec.mode, spec.new_levels, spec.factors, used)

const _RANDOM_EFFECTS_MODES = (:zero, :conditional, :marginal)

_supports_random_effects(model) = false
_supports_random_effects(model::StatsModels.TableRegressionModel) = _supports_random_effects(model.model)
_re_factors(model) = Symbol[]
_re_grouping_vars(model, factors) = Symbol[]
_re_slope_vars(model, factors) = Symbol[]
_re_offset(model, table::NamedTuple, spec::RandomEffectsSpec) =
    error("random_effects = :$(spec.mode) is not implemented for $(typeof(model))")

"""
    _validate_random_effects(model, random_effects, re_new, re_terms) -> Union{Nothing, RandomEffectsSpec}

The `random_effects`, `re_new` and `re_terms` keywords: `nothing` for `random_effects = :zero`
(every random effect 0, the behaviour of every model class), else the options of a
conditional request. `re_new` and `re_terms` are rejected unless `random_effects =
:conditional`, so that a keyword that would change nothing is not silently ignored.
"""
function _validate_random_effects(model, random_effects, re_new, re_terms)
    (random_effects isa Symbol && random_effects in _RANDOM_EFFECTS_MODES) || throw(ArgumentError(
        "random_effects must be :zero, :conditional or :marginal, got $(repr(random_effects))"))
    (re_new isa Symbol && re_new in (:zero, :error)) || throw(ArgumentError(
        "re_new must be :zero or :error, got $(repr(re_new))"))
    if random_effects !== :conditional
        (re_new === :zero && isnothing(re_terms)) || throw(ArgumentError(
            "re_new and re_terms apply only with random_effects = :conditional"))
    end
    random_effects === :zero && return nothing
    _supports_random_effects(model) || throw(ArgumentError(
        "random_effects = :$random_effects applies only to models with random effects " *
        "(MixedModels.jl); a $(nameof(typeof(model))) has none"))
    random_effects === :marginal && throw(MarginsError(
        "random_effects = :marginal (population-averaged estimates, which average over the " *
        "random-effects distribution) is not implemented yet. Use random_effects = :zero (a " *
        "unit whose random effects are 0, identical to :marginal under an identity link) or " *
        ":conditional (the units in the data)."))
    all_factors = _re_factors(model)
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
    return RandomEffectsSpec(:conditional, re_new, factors, copy(factors))
end

"""
    _validate_random_effects_request(model, spec, type, vars, vars_explicit, changed, grid)

The cases phase 1 does not support, rejected before any computation:

- the effect of a variable with a random slope in a conditioned factor (`x` in
  `(1 + x | rater)`): it includes each rater's slope, which an offset cannot carry;
- a variable with a random slope set counterfactually in a population request (`scenarios`, or a
  second-differences modifier): the offset z'b̂ is computed from the observed values. A profile
  grid may set such a variable (z is evaluated at the grid rows).
"""
function _validate_random_effects_request(model, spec::RandomEffectsSpec, type::Symbol, vars,
                                          vars_explicit::Bool, changed::Vector{Symbol}, grid)
    slopes = _re_slope_vars(model, spec.factors)
    isempty(slopes) && return nothing
    if type === :effects && !isnothing(vars)
        bad = intersect(vars, slopes)
        if !isempty(bad)
            names_str = join((":$v" for v in bad), ", ")
            msg = "With random_effects = :conditional, the effect of $names_str includes each " *
                  "group's random slope for $(length(bad) == 1 ? "it" : "them"), which is not " *
                  "supported yet. Use random_effects = :zero (the effect for a group whose random " *
                  "effects are 0), or re_terms to condition only on grouping factors without that slope."
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
        "With random_effects = :conditional, $(join((":$v" for v in bad), ", ")) cannot be set " *
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

# The grouping columns a reference grid may set under `random_effects = :conditional` (they are
# not predictors of the fixed-effects formula, so the grid check would reject them otherwise)
_re_grid_vars(model, ::Nothing) = Symbol[]
_re_grid_vars(model, spec::RandomEffectsSpec) = _re_grouping_vars(model, spec.factors)

"""
    _set_random_effects_metadata!(metadata, model, spec)

Record the random-effects treatment on a result: `:random_effects` (`:zero` or `:conditional`)
and `:re_terms`, the grouping factors whose predicted random effects entered η (empty for
`:zero`). For a model with random effects, also `:re_factors`, all its grouping factors (the
printed result names those set to 0).
"""
function _set_random_effects_metadata!(metadata::AbstractDict, model, spec)
    _supports_random_effects(model) || return metadata
    metadata[:re_factors] = _re_factors(model)
    isnothing(spec) && return metadata
    metadata[:random_effects] = spec.mode
    metadata[:re_terms] = copy(spec.used)
    return metadata
end
