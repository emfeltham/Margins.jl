"""
    FixedEffectModelsExt

Package extension enabling Margins.jl support for FixedEffectModels.jl.

Supports marginal effects (AME, MEM) and predictions (AAP, APM) for
fixed-effect models, including IV models. Predictions require `save=:fe`
at model fit time.

# Supported
- `population_margins(model, data; type=:effects, ...)` — AME
- `profile_margins(model, data, grid; type=:effects, ...)` — MEM
- `population_margins(model, data; type=:predictions, ...)` — AAP (requires save=:fe)
- `profile_margins(model, data, grid; type=:predictions, ...)` — APM (requires save=:fe)
- IV models — supported with informational note about structural coefficients

# Blocked (with informative errors)
- Predictions without `save=:fe` — absorbed FE estimates needed
- Counterfactual scenarios on FE variables for predictions
"""
module FixedEffectModelsExt

using Margins
using Margins: _auto_link, validate_model_methods, MarginsError,
               PredictionsResult, EffectsResult, MarginsRequest,
               _normalize_request, _population_result, _profile_result, _formula_predictor_names
using FixedEffectModels: FixedEffectModel
using GLM
using GLM: IdentityLink
import FormulaCompiler
using StatsModels: formula
using DataFrames: DataFrame, nrow, ncol, names
using Tables
using Statistics: mean

# ---------------------------------------------------------------------------
# _auto_link: FixedEffectModels always use identity link (linear models only)
# ---------------------------------------------------------------------------
function Margins._auto_link(model::FixedEffectModel)
    _warn_if_iv(model)
    return IdentityLink()
end

# ---------------------------------------------------------------------------
# _model_weights: a FixedEffectModel keeps neither the `weights=` column name nor the weight
# vector (FixedEffectModels 1.13), so `weights=:model` cannot recover them and resolves to no
# weights; a weighted `reg` needs `weights=:w` passed to Margins explicitly. Same as the generic
# fallback, spelled out so the reason is on record.
# ---------------------------------------------------------------------------
Margins._model_weights(::FixedEffectModel) = nothing

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
    # For predictions: require save=:fe
    if req.type === :predictions
        _validate_fe_available(model)
    end

    # Block scenarios on FE variables for predictions
    if req.type === :predictions
        _validate_no_fe_scenarios(model, req.scenarios)
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
# population_margins: override for FixedEffectModel to add FE offsets. The request is
# normalized as for every model; only the offsets are FE-specific. (Step 5 of the 3.0 plan
# replaces these overrides with a per-row offset hook.)
# ---------------------------------------------------------------------------
function Margins.population_margins(
    model::FixedEffectModel, data;
    type::Symbol=:effects, vars=nothing, scale::Symbol=:response,
    backend::Symbol=:ad, scenarios=nothing, groups=nothing, measure::Symbol=:effect,
    contrasts::Symbol=:baseline,
    alpha::Real=0.05, vcov=GLM.vcov, weights=:model, ci_alpha=nothing
)
    Margins._reject_ci_alpha(ci_alpha)
    # Shared validation, data preparation, weights (`:model` resolves to none here: see
    # _model_weights above) and the one evaluation of `vcov` for this call
    req = _normalize_request(model, data; type, vars, scale, backend, scenarios, groups,
                             measure, contrasts, alpha, vcov, weights)
    (result, _) = _population_result(req)

    # Add FE offset to prediction estimates (SEs unchanged — FE offset is constant w.r.t. β)
    if type === :predictions
        return _adjust_population_result_with_fe(result, model, data, req.weights)
    end
    return result
end

# ---------------------------------------------------------------------------
# profile_margins: override for FixedEffectModel to add FE offsets
# ---------------------------------------------------------------------------
function Margins.profile_margins(
    model::FixedEffectModel, data, reference_grid::DataFrame;
    type::Symbol=:effects, vars=nothing, scale::Symbol=:response,
    backend::Symbol=:ad, measure::Symbol=:effect,
    contrasts::Symbol=:baseline,
    alpha::Real=0.05, vcov=GLM.vcov, interval::Symbol=:confidence, ci_alpha=nothing,
)
    Margins._reject_ci_alpha(ci_alpha)
    # Shared validation of the options and the reference grid; one evaluation of `vcov`
    req = _normalize_request(model, data; type, vars, scale, backend, reference_grid, measure,
                             contrasts, alpha, vcov, weights=:none, interval)
    result = _profile_result(req)

    # Add FE offset for predictions
    if type === :predictions
        return _adjust_profile_result_with_fe(result, model, data, req.reference_grid)
    end

    return result
end

# ---------------------------------------------------------------------------
# FE offset computation
# ---------------------------------------------------------------------------

"""
    _compute_population_fe_offset(model, data, weights) -> Float64

Compute the (weighted) average FE offset across the sample.
For identity link: ŷ = Xβ + Σ_k fe_k, so the offset is mean(Σ_k fe_k).
"""
function _compute_population_fe_offset(model::FixedEffectModel, data, weights)
    fe_df = model.fe
    fekeys = model.fekeys

    # Compute total FE offset per observation
    n = nrow(fe_df)
    total_offset = zeros(n)
    for k in fekeys
        fe_col = Symbol("fe_", k)
        vals = fe_df[!, fe_col]
        for i in 1:n
            v = vals[i]
            total_offset[i] += ismissing(v) ? 0.0 : Float64(v)
        end
    end

    # (Weighted) average
    if isnothing(weights)
        return mean(total_offset)
    else
        # Use same weight vector (aligned to esample if needed)
        w = weights
        return sum(w .* total_offset) / sum(w)
    end
end

"""
    _compute_profile_fe_offsets(model, data, reference_grid) -> Vector{Float64}

Compute FE offsets for each profile row.

For FE variables present in the reference grid: look up FE estimates for that level.
For FE variables NOT in the reference grid: use population-average FE offset for that variable.
"""
function _compute_profile_fe_offsets(model::FixedEffectModel, data, reference_grid::DataFrame)
    fe_df = model.fe
    fekeys = model.fekeys
    n_profiles = nrow(reference_grid)
    offsets = zeros(n_profiles)

    for k in fekeys
        fe_col = Symbol("fe_", k)
        fe_vals = fe_df[!, fe_col]

        if string(k) in names(reference_grid)
            # FE variable is in reference grid — look up FE estimate per row
            grid_levels = reference_grid[!, string(k)]
            lookup = _build_fe_lookup(fe_df, k, fe_col)

            for i in 1:n_profiles
                level = grid_levels[i]
                offset = get(lookup, level, 0.0)
                offsets[i] += offset
            end
        else
            # FE variable NOT in reference grid — use population average for this FE
            avg = mean(skipmissing(fe_vals))
            for i in 1:n_profiles
                offsets[i] += avg
            end
        end
    end

    return offsets
end

"""
    _build_fe_lookup(fe_df, key_col, value_col) -> Dict

Build a lookup table from FE level → FE estimate.
Handles both string and numeric FE variables.
"""
function _build_fe_lookup(fe_df::DataFrame, key_col::Symbol, value_col::Symbol)
    keys_data = fe_df[!, key_col]
    vals_data = fe_df[!, value_col]
    lookup = Dict{Any, Float64}()
    for i in eachindex(keys_data)
        k = keys_data[i]
        v = vals_data[i]
        if !ismissing(v) && !haskey(lookup, k)
            lookup[k] = Float64(v)
        end
    end
    return lookup
end

"""
    _adjust_population_result_with_fe(result, model, data, weights) -> result

Adjust a population PredictionsResult by adding FE offsets.
"""
function _adjust_population_result_with_fe(result::PredictionsResult, model::FixedEffectModel, data, weights)
    fe_offset = _compute_population_fe_offset(model, data, weights)
    adjusted_estimates = result.estimates .+ fe_offset
    return PredictionsResult(adjusted_estimates, result.standard_errors,
                            result.profile_values, result.group_values, result.gradients, result.metadata)
end

# Non-prediction results pass through unchanged
_adjust_population_result_with_fe(result, model, data, weights) = result

"""
    _adjust_profile_result_with_fe(result, model, data, reference_grid) -> result

Adjust a profile PredictionsResult by adding FE offsets per profile row.
"""
function _adjust_profile_result_with_fe(result::PredictionsResult, model::FixedEffectModel, data, reference_grid::DataFrame)
    offsets = _compute_profile_fe_offsets(model, data, reference_grid)
    adjusted_estimates = result.estimates .+ offsets
    return PredictionsResult(adjusted_estimates, result.standard_errors,
                            result.profile_values, result.group_values, result.gradients, result.metadata)
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
    _validate_fe_available(model::FixedEffectModel)

Error if FE estimates are not saved (model.fe is empty).
"""
function _validate_fe_available(model::FixedEffectModel)
    if nrow(model.fe) == 0
        throw(MarginsError(
            "Predictions require absorbed fixed-effect estimates, but none were saved.\n\n" *
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
    _get_fe_variables(model::FixedEffectModel) -> Vector{Symbol}

Return the symbols of absorbed fixed-effect variables.
"""
function _get_fe_variables(model::FixedEffectModel)
    return model.fekeys
end

end # module
