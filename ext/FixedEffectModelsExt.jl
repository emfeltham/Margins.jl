"""
    FixedEffectModelsExt

Package extension supplying the FixedEffectModels.jl methods of Margins' model-class interface
(`src/core/model_interface.jl`). `population_margins`, `profile_margins` and
`second_differences` are the shared ones: scenarios, groups, weights, inference and every
validator apply as for any other model. What is specific to a `FixedEffectModel`:

- **Offset:** the absorbed fixed effects enter η as a per-row offset, Σₖ feₖ (their estimates,
  which need `reg(...; save=:fe)`), so predictions are right at every row, within `groups` and
  under `scenarios` (each row keeps its own fixed effects). At reference-grid rows the offset is
  the `offset=` keyword, an `offset` grid column, or the estimates of the fixed-effect levels the
  grid sets.
- **Estimation rows:** only the rows in `esample` (rows with missing values, non-positive
  weights or singleton fixed-effect groups are dropped by `reg`). The data must be the table
  given to `reg`, with one row per entry of `esample`; it is subset here.
- **Inference df:** `dof_residual(model)`, which FixedEffectModels adjusts for clustering, as
  its own `coeftable` uses.
- **Weights:** `reg` does not store its weights, so `weights=:model` cannot read them; it
  averages without weights and warns once per session.
- **Link:** identity; IV models use the structural (second-stage) coefficients.

# Blocked (with informative errors)
- Predictions (or effects that depend on the level of ŷ) without `save=:fe`
- Predictions with a fixed effect interacted with a continuous variable (`fe(g)&x`)
- Counterfactual scenarios on fixed-effect variables for predictions
- Marginal effects of absorbed fixed-effect variables
- Marginal effects of a variable with absorbed group-specific slopes (`x` in `fe(g)&x`): its
  derivative includes the slopes, which are not in `coef()` or `vcov()`
"""
module FixedEffectModelsExt

using Margins
using Margins: validate_model_methods, MarginsError, MarginsRequest, UnknownWeights,
               _formula_predictor_names
import FixedEffectModels
using FixedEffectModels: FixedEffectModel
using GLM
using GLM: IdentityLink
import FormulaCompiler
import StatsModels
using StatsModels: formula
using DataFrames: DataFrame, nrow, ncol, names
using Tables
using GLM: dof_residual

# ---------------------------------------------------------------------------
# Link, response, inference df, weights
# ---------------------------------------------------------------------------
# FixedEffectModels are linear (identity link)
function Margins._link(model::FixedEffectModel)
    _warn_if_iv(model)
    return IdentityLink()
end

Margins._response_name(model::FixedEffectModel) = Symbol(model.responsename)

# The residual df FixedEffectModels uses for its own t tests (`coeftable`, `confint`): already
# adjusted for clustering (min number of clusters − 1 with a cluster vcov), and net of the
# absorbed fixed effects otherwise. Decision D1: a property of the model, whatever `vcov` is passed.
Margins._inference_df(model::FixedEffectModel) = Margins._finite_df(dof_residual(model))

# Regressors dropped as collinear (with each other or with the fixed effects): FixedEffectModels
# keeps no pivot, but reports a dropped coefficient as exactly 0 with a NaN variance in its stored
# `vcov`, which is the rule here (whatever `vcov=` the call passes)
Margins._dropped_coefficients(model::FixedEffectModel) =
    [i for i in eachindex(model.coef) if iszero(model.coef[i]) && isnan(model.vcov[i, i])]

# A FixedEffectModel keeps neither the `weights=` column name nor the weight vector
# (FixedEffectModels 1.13), so whether `reg` was weighted cannot be known: `weights=:model`
# averages without weights and warns once per session (decision D2).
Margins._model_weights(::FixedEffectModel) = UnknownWeights(
    "a FixedEffectModels fit does not store the weights it was estimated with " *
    "(reg(...; weights=:w)), so they cannot be read.")

# ---------------------------------------------------------------------------
# Estimation rows: `esample`
# ---------------------------------------------------------------------------
# The data must be the table given to `reg` (one row per entry of `esample`); it is subset here.
# Any other row count is an error: a table that merely has `nobs` rows (e.g. one subset by hand)
# cannot be told apart from a different table of that length, which would misalign silently.
function Margins._estimation_rows(model::FixedEffectModel, n::Int)
    es = model.esample
    n == length(es) && return all(es) ? nothing : findall(es)
    throw(ArgumentError(
        "The data has $n rows, but the model was fit to a table with $(length(es)) rows " *
        "($(model.nobs) of them in its estimation sample). Pass the table given to reg(), " *
        "unsubset: Margins selects the estimation rows from it (model.esample)."))
end

# ---------------------------------------------------------------------------
# Offset: the absorbed fixed effects, Σₖ feₖ, at each estimation row
# ---------------------------------------------------------------------------
Margins._has_offset(model::FixedEffectModel) = !isempty(model.fekeys)

# The columns of `model.fe` holding fixed-effect estimates (the others are the fekeys)
_fe_value_columns(model::FixedEffectModel) = [c for c in names(model.fe) if !(Symbol(c) in model.fekeys)]

function Margins._offset(model::FixedEffectModel, data_nt::NamedTuple)
    _validate_fe_offset_available(model)
    n = length(first(data_nt))
    n == model.nobs || throw(ArgumentError(
        "The data has $n rows but the model's estimation sample has $(model.nobs); the fixed " *
        "effects cannot be matched to rows. Pass the table given to reg()."))
    rows = findall(model.esample)
    off = zeros(Float64, n)
    for c in _fe_value_columns(model)
        vals = model.fe[!, c]
        for (i, r) in enumerate(rows)
            v = vals[r]
            ismissing(v) && throw(ArgumentError(
                "The fixed-effect estimate $c is missing at estimation row $r"))
            off[i] += Float64(v)
        end
    end
    return off
end

# At reference-grid rows: the estimates of the fixed-effect levels the grid sets (FixedEffectModels'
# `predict` convention). Consulted only when there is neither `offset=` nor an `offset` grid column.
function Margins._grid_offset(model::FixedEffectModel, grid::DataFrame)
    _validate_fe_offset_available(model)
    given = [k for k in model.fekeys if string(k) in names(grid)]
    if length(given) != length(model.fekeys)
        missing_keys = setdiff(model.fekeys, given)
        throw(ArgumentError(
            "The prediction at a reference-grid row includes the absorbed fixed effects, so " *
            "profile_margins needs their value at each row: add the fixed-effect variable(s) " *
            "$(join(string.(":", missing_keys), ", ")) to the grid (their estimated fixed effects are " *
            "looked up), or pass offset=<number or one value per grid row> (the sum of the fixed " *
            "effects), or add an `offset` column to the grid. (Before Margins 3.0 the average fixed " *
            "effect was used silently.)"))
    end
    out = zeros(Float64, nrow(grid))
    for (k, c) in zip(model.fekeys, _fe_value_columns(model))
        lookup = _build_fe_lookup(model.fe, k, Symbol(c))
        for (i, level) in enumerate(grid[!, string(k)])
            key = _fe_key(level)
            haskey(lookup, key) || throw(ArgumentError(
                "Reference grid: :$k = $(repr(key)) has no estimated fixed effect in the model " *
                "(not a level of the estimation sample)."))
            out[i] += lookup[key]
        end
    end
    return out
end

_fe_key(v) = Margins.CategoricalArrays.unwrap(v)

"""
    _build_fe_lookup(fe_df, key_col, value_col) -> Dict

Fixed-effect level → estimate, over the estimation rows of `fe_df`.
"""
function _build_fe_lookup(fe_df::DataFrame, key_col::Symbol, value_col::Symbol)
    keys_data = fe_df[!, key_col]
    vals_data = fe_df[!, value_col]
    lookup = Dict{Any, Float64}()
    for i in eachindex(keys_data)
        v = vals_data[i]
        ismissing(v) && continue
        k = _fe_key(keys_data[i])
        haskey(lookup, k) || (lookup[k] = Float64(v))
    end
    return lookup
end

# ---------------------------------------------------------------------------
# fixed_effects_form (FormulaCompiler's public model-class hook): return clean formula
# without fe() terms. FixedEffectModel.formula_schema already has fe() stripped and schema
# applied
# ---------------------------------------------------------------------------
function FormulaCompiler.fixed_effects_form(model::FixedEffectModel)
    return formula(model)  # StatsModels.formula returns formula_schema (clean, no fe())
end

# ---------------------------------------------------------------------------
# validate_model_methods: FixedEffectModel already passes via StatsAPI dispatch,
# but add an explicit method to skip the generic try/catch path
# ---------------------------------------------------------------------------
function Margins.validate_model_methods(model::FixedEffectModel)
    _warn_if_iv(model)
    return nothing
end

# ---------------------------------------------------------------------------
# _predictor_names: the absorbed fixed effects are predictors too. They are not in the
# formula Margins evaluates (fe() terms are stripped), but a reference grid may set them for
# prediction (the FE estimate of that level is looked up), and a scenario on one is rejected
# below for predictions rather than as "not a predictor".
# ---------------------------------------------------------------------------
Margins._predictor_names(model::FixedEffectModel) =
    sort!(unique!(vcat(_formula_predictor_names(model), _get_fe_variables(model))))

# ---------------------------------------------------------------------------
# _validate_focal_vars / _validate_request: the FE-specific constraints, in the shared validation of
# Margins._normalize_request (so a FixedEffectModel also gets the measure filter, the
# vars/groups and scenarios/groups checks and the Σ handling of every other model)
# ---------------------------------------------------------------------------
function Margins._validate_request(model::FixedEffectModel, req::MarginsRequest)
    # Block scenarios on FE variables for predictions (`save=:fe` is checked where the
    # fixed-effect offset is read, `_offset` / `_grid_offset`)
    if req.type === :predictions
        _validate_no_fe_scenarios(model, req.scenarios)
    end
    # Block effects of a variable with absorbed slopes, named or auto-detected
    if req.type === :effects && !isnothing(req.vars)
        _validate_no_fe_slope_vars(model, req.vars, req.vars_explicit)
    end
    return nothing
end

# Block FE variables in vars (for effects), before the data checks on those columns
function Margins._validate_focal_vars(model::FixedEffectModel, vars)
    fe_vars = _get_fe_variables(model)
    for v in (vars isa Symbol ? [vars] : vars)
        if v in fe_vars
            throw(MarginsError(
                "Variable :$v is an absorbed fixed effect in this model.\n" *
                "Marginal effects for absorbed FE variables are not meaningful — " *
                "their coefficients are not estimated in coef().\n" *
                "Remove :$v from vars, or use a model without fe($v)."
            ))
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

# Track whether the IV info message has been shown to avoid spamming
const _IV_WARNING_SHOWN = Ref(false)

"""
    _warn_if_iv(model::FixedEffectModel)

Log a one-time informational note when working with an IV model.
IV models use structural (second-stage) coefficients, which is standard
practice in Stata and R marginaleffects.
"""
function _warn_if_iv(model::FixedEffectModel)
    if !isnan(model.F_kp) && !_IV_WARNING_SHOWN[]
        @info "IV model detected: margins computed using structural (second-stage) coefficients " *
              "and covariance matrix, consistent with Stata and R marginaleffects. " *
              "Interpret effects as structural marginal effects."
        _IV_WARNING_SHOWN[] = true
    end
end

"""
    _validate_fe_offset_available(model::FixedEffectModel)

Error if the fixed-effect estimates are not saved (model.fe is empty), or if a fixed effect is
interacted with a continuous variable (`fe(g)&x`: its contribution is a slope, not an offset).
"""
function _validate_fe_offset_available(model::FixedEffectModel)
    if isdefined(FixedEffectModels, :has_cont_fe_interaction) &&
       FixedEffectModels.has_cont_fe_interaction(model.formula)
        throw(MarginsError(
            "The model interacts a fixed effect with a continuous variable (fe(g)&x), so the " *
            "absorbed fixed effects are not an additive offset; predictions (and effects that " *
            "depend on the level of ŷ) are not supported for it."))
    end
    if nrow(model.fe) == 0
        throw(MarginsError(
            "Predictions require absorbed fixed-effect estimates, but none were saved. (So do " *
            "effects that depend on the level of ŷ: elasticities, :semielasticity_eydx.)\n\n" *
            "Re-fit the model with `save=:fe` to save FE estimates:\n" *
            "  model = reg(data, @formula(y ~ x1 + x2 + fe(group)); save=:fe)\n\n" *
            "Then call population_margins(model, data; type=:predictions)."
        ))
    end
end

"""
    _validate_no_fe_scenarios(model, scenarios)

Block counterfactual scenarios on FE variables for predictions.
Setting everyone to a specific FE group conflates causal effects with unobserved heterogeneity.
"""
function _validate_no_fe_scenarios(model::FixedEffectModel, scenarios)
    isnothing(scenarios) && return
    fe_vars = _get_fe_variables(model)

    # Handle both NamedTuple and Tuple-of-Pairs
    scenario_vars = if scenarios isa NamedTuple
        collect(keys(scenarios))
    else
        [p.first for p in scenarios if p isa Pair]
    end

    for v in scenario_vars
        if v in fe_vars
            throw(MarginsError(
                "Counterfactual scenarios on absorbed FE variable :$v are not meaningful.\n" *
                "Setting everyone to a specific $v level conflates causal effects with\n" *
                "unobserved heterogeneity captured by the fixed effect.\n" *
                "Remove :$v from scenarios."
            ))
        end
    end
end

"""
    _fe_slope_variables(model::FixedEffectModel) -> Vector{Symbol}

Return the variables that a fixed effect is interacted with (`x` in `fe(g)&x` or `fe(g)*x`).
"""
function _fe_slope_variables(model::FixedEffectModel)
    out = Symbol[]
    for t in FixedEffectModels.eachterm(model.formula.rhs)
        FixedEffectModels.is_cont_fe_int(t) || continue
        for s in t.terms
            FixedEffectModels.has_fe(s) || append!(out, StatsModels.termvars(s))
        end
    end
    return unique!(out)
end

"""
    _validate_no_fe_slope_vars(model, vars, vars_explicit)

Block marginal effects of a variable with absorbed group-specific slopes. With `fe(g)&x`,
∂ŷ/∂x includes the slope of each row's group, which FixedEffectModels absorbs: it is in neither
`coef()` nor `vcov()`, so an effect computed from the coefficients would leave it out.
"""
function _validate_no_fe_slope_vars(model::FixedEffectModel, vars, vars_explicit::Bool)
    slope_vars = _fe_slope_variables(model)
    bad = intersect(vars, slope_vars)
    isempty(bad) && return
    names_str = join((":$v" for v in bad), ", ")
    msg = "The model absorbs group-specific slopes in $names_str (a fixed effect interacted " *
          "with $(length(bad) == 1 ? "it" : "them"), fe(g)&x). The marginal effect of " *
          "$(length(bad) == 1 ? "this variable" : "these variables") includes those slopes, " *
          "which are not in coef() or vcov(), so it is not supported."
    if !vars_explicit
        rest = setdiff(vars, bad)
        msg *= isempty(rest) ? " No other variable has an effect to compute." :
               "\nPass the other variables explicitly: vars = [" * join((":$v" for v in rest), ", ") * "]"
    end
    throw(MarginsError(msg))
end

"""
    _get_fe_variables(model::FixedEffectModel) -> Vector{Symbol}

Return the symbols of absorbed fixed-effect variables.
"""
function _get_fe_variables(model::FixedEffectModel)
    return model.fekeys
end

end # module
