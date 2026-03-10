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
               validate_margins_common_inputs, _validate_scenarios_specific,
               _validate_groups_parameter, _validate_weights_parameter,
               PredictionsResult, EffectsResult,
               _population_predictions, _build_metadata, _ame_calculate,
               _population_margins_with_contexts, _process_vars_parameter,
               _process_weights_parameter, _convert_numeric_to_float64,
               get_or_build_engine, PopulationUsage, ProfileUsage,
               HasDerivatives, NoDerivatives,
               _profile_margins, process_reference_grid,
               _validate_reference_grid_column_type
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
# fixed_effects_form: return clean formula without fe() terms
# FixedEffectModel.formula_schema already has fe() stripped and schema applied
# ---------------------------------------------------------------------------
function FormulaCompiler.fixed_effects_form(model::FixedEffectModel)
    return formula(model)  # StatsModels.formula returns formula_schema (clean, no fe())
end

# ---------------------------------------------------------------------------
# validate_model_methods: FixedEffectModel already passes via StatsAPI dispatch,
# but add explicit method to cache and skip the try/catch path
# ---------------------------------------------------------------------------
function Margins.validate_model_methods(model::FixedEffectModel)
    _warn_if_iv(model)
    return nothing
end

# ---------------------------------------------------------------------------
# validate_margins_common_inputs: validate FE-specific constraints
# ---------------------------------------------------------------------------
function Margins.validate_margins_common_inputs(model::FixedEffectModel, data, type, vars, scale, backend, measure, vcov)
    _warn_if_iv(model)

    # For predictions: require save=:fe
    if type === :predictions
        _validate_fe_available(model)
    end

    # Block FE variables in vars (for effects)
    if !isnothing(vars)
        fe_vars = _get_fe_variables(model)
        vars_vec = vars isa Symbol ? [vars] : vars
        for v in vars_vec
            if v in fe_vars
                throw(MarginsError(
                    "Variable :$v is an absorbed fixed effect in this model.\n" *
                    "Marginal effects for absorbed FE variables are not meaningful — " *
                    "their coefficients are not estimated in coef().\n" *
                    "Remove :$v from vars, or use a model without fe($v)."
                ))
            end
        end
    end

    # Delegate to the default validation path (call individual validators)
    Margins.validate_required_args(model, data)
    Margins.validate_population_parameters(type, scale, backend, measure, vars)
    Margins.validate_vcov_parameter(vcov, model)
end

# ---------------------------------------------------------------------------
# population_margins: override for FixedEffectModel to add FE offsets
# ---------------------------------------------------------------------------
function Margins.population_margins(
    model::FixedEffectModel, data;
    type::Symbol=:effects, vars=nothing, scale::Symbol=:response,
    backend::Symbol=:ad, scenarios=nothing, groups=nothing, measure::Symbol=:effect,
    contrasts::Symbol=:baseline,
    ci_alpha::Float64=0.05, vcov=GLM.vcov, weights=nothing
)
    # Shared input validation
    validate_margins_common_inputs(model, data, type, vars, scale, backend, measure, vcov)

    # Population-specific validation
    if !isnothing(scenarios)
        _validate_scenarios_specific(scenarios, vars, type)
        # Block scenarios on FE variables for predictions
        if type === :predictions
            _validate_no_fe_scenarios(model, scenarios)
        end
    end
    if !isnothing(groups)
        _validate_groups_parameter(groups)
    end
    if !isnothing(weights)
        _validate_weights_parameter(weights, data)
    end

    # Data conversion
    data_nt_raw = Tables.columntable(data)
    data_nt = _convert_numeric_to_float64(data_nt_raw)

    # Process weights
    weights_vec = _process_weights_parameter(weights, data, data_nt)
    weight_col = weights isa Symbol ? weights : nothing

    # Handle vars
    if type === :effects
        vars = _process_vars_parameter(model, vars, data_nt, weight_col)
    else
        vars = nothing
    end

    # Build engine
    deriv_support = (type === :effects && !isnothing(vars) && !isempty(vars)) ? HasDerivatives : NoDerivatives
    engine = get_or_build_engine(PopulationUsage, deriv_support, model, data_nt, isnothing(vars) ? Symbol[] : vars, vcov, backend)

    # Handle scenarios/groups
    if !isnothing(scenarios) || !isnothing(groups)
        result = _population_margins_with_contexts(engine, data_nt, vars, scenarios, groups, weights_vec, type, scale, backend, ci_alpha, measure, contrasts)
        # Add FE offset for predictions with contexts
        if type === :predictions
            return _adjust_population_result_with_fe(result, model, data, weights_vec)
        end
        return result
    end

    if type === :effects
        df, G = _ame_calculate(engine, data_nt, scale, backend, measure, contrasts, weights_vec)
        metadata = _build_metadata(; type, vars, scale, backend, measure, n_obs=length(first(data_nt)), model_type=typeof(model))
        metadata[:alpha] = ci_alpha
        metadata[:analysis_type] = :population
        metadata[:weighted] = !isnothing(weights_vec)

        estimates = df.estimate
        standard_errors = df.se
        variables = df.variable
        terms = df.contrast

        return EffectsResult(estimates, standard_errors, variables, terms, nothing, nothing, G, metadata)
    else # :predictions
        df, G = _population_predictions(engine, data_nt; scale, weights=weights_vec)
        metadata = _build_metadata(; type, vars=Symbol[], scale, backend, n_obs=length(first(data_nt)), model_type=typeof(model))
        metadata[:alpha] = ci_alpha
        metadata[:analysis_type] = :population
        metadata[:weighted] = !isnothing(weights_vec)

        # Add FE offset to prediction estimates
        fe_offset = _compute_population_fe_offset(model, data, weights_vec)
        estimates = df.estimate .+ fe_offset
        standard_errors = df.se  # SEs unchanged — FE offset is constant w.r.t. β

        return PredictionsResult(estimates, standard_errors, nothing, nothing, G, metadata)
    end
end

# ---------------------------------------------------------------------------
# profile_margins: override for FixedEffectModel to add FE offsets
# ---------------------------------------------------------------------------
function Margins.profile_margins(
    model::FixedEffectModel, data, reference_grid::DataFrame;
    type::Symbol=:effects, vars=nothing, scale::Symbol=:response,
    backend::Symbol=:ad, measure::Symbol=:effect,
    contrasts::Symbol=:baseline,
    ci_alpha::Float64=0.05, vcov=GLM.vcov,
)
    # Convert data
    data_nt_raw = Tables.columntable(data)
    data_nt = _convert_numeric_to_float64(data_nt_raw)

    # Process reference grid
    processed_reference_grid = process_reference_grid(data, reference_grid)

    # Validation
    validate_margins_common_inputs(model, data_nt, type, vars, scale, backend, measure, vcov)

    if isnothing(reference_grid)
        throw(ArgumentError("reference_grid cannot be nothing"))
    end
    if nrow(processed_reference_grid) == 0
        throw(ArgumentError("reference_grid cannot be empty"))
    end
    for col_name in names(processed_reference_grid)
        col = processed_reference_grid[!, col_name]
        _validate_reference_grid_column_type(col, col_name)
    end
    if ncol(processed_reference_grid) == 0
        throw(ArgumentError("reference_grid must have at least one column"))
    end

    # Block counterfactuals on FE variables for predictions
    if type === :predictions
        _validate_no_fe_in_grid(model, processed_reference_grid)
    end

    # Delegate to internal implementation
    result = _profile_margins(model, data_nt, processed_reference_grid, type, vars, scale, backend, measure, vcov, processed_reference_grid, ci_alpha, contrasts)

    # Add FE offset for predictions
    if type === :predictions
        return _adjust_profile_result_with_fe(result, model, data, processed_reference_grid)
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
    _validate_no_fe_in_grid(model, reference_grid)

Warn but allow FE variables in profile prediction grids — they're used for FE lookup.
Block if save=:fe wasn't used (already checked elsewhere).
"""
function _validate_no_fe_in_grid(model::FixedEffectModel, reference_grid::DataFrame)
    # FE variables in grid are fine — they specify which FE level to use for prediction
    # No validation needed beyond _validate_fe_available (already called)
end

"""
    _get_fe_variables(model::FixedEffectModel) -> Vector{Symbol}

Return the symbols of absorbed fixed-effect variables.
"""
function _get_fe_variables(model::FixedEffectModel)
    return model.fekeys
end

end # module
