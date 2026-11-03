# at_point.jl
# Local derivative computations for second differences at specific points

using Statistics: mean, median, std

# ============================================================================
# Constants for Finite Difference Numerical Stability
# ============================================================================

# Automatic delta computation: default factor of standard deviation
# Rationale: 1% of SD provides good bias-variance balance for most smooth functions
# - Smaller values increase numerical error from cancellation
# - Larger values increase approximation bias from non-linearity
const DEFAULT_DELTA_FACTOR = 0.01

# Relative delta factor for large evaluation points (|eval_pt| > 1)
# Ensures δ ≥ |eval_pt| × 1e-8 to maintain numerical distinctness of (eval_pt ± δ)
# Stays well above machine epsilon while scaling proportionally with magnitude
const RELATIVE_DELTA_FACTOR = 1e-8

# Minimum delta for small evaluation points (|eval_pt| ≤ 1)
# Prevents δ → 0 when eval_pt ≈ 0, which would cause division issues in slope calculation
# Points are checked for distinctness with `pt_minus < pt_plus`, not `≈`
const SMALL_VALUE_MIN_DELTA = 1e-9

# ============================================================================
# Helper Functions for _second_differences_local
# ============================================================================

"""
    _adjust_delta_for_eval_point(eval_pt::Real, δ_raw::Float64)::Float64

Adjust the finite difference step size based on the evaluation point magnitude
to avoid floating-point precision issues.

For large values (|eval_pt| > 1), uses relative delta (≥1e-8 relative error).
For small values, uses absolute delta with minimum 1e-9.

# Arguments
- `eval_pt`: Evaluation point for the modifier
- `δ_raw`: Raw delta value (before adjustment)

# Returns
- Adjusted delta value ensuring numerical stability
"""
function _adjust_delta_for_eval_point(eval_pt::Real, δ_raw::Float64)::Float64
    # Rationale: Floating-point arithmetic loses precision when δ is too small relative
    # to eval_pt. For eval_pt + δ ≈ eval_pt when δ < ε|eval_pt| where ε ≈ 1e-16 (machine epsilon).
    # We use RELATIVE_DELTA_FACTOR to stay well above machine epsilon, ensuring (eval_pt ± δ) are numerically distinct.
    if abs(eval_pt) > 1.0
        # For large values, use relative delta (at least RELATIVE_DELTA_FACTOR relative error)
        # This ensures δ scales proportionally with the magnitude of eval_pt
        max(δ_raw, abs(eval_pt) * RELATIVE_DELTA_FACTOR)
    else
        # For small values near zero, use absolute delta with minimum SMALL_VALUE_MIN_DELTA
        # Prevents degenerate case where δ → 0 when eval_pt ≈ 0
        max(δ_raw, SMALL_VALUE_MIN_DELTA)
    end
end

"""
    _build_finite_diff_scenarios(
        modifier::Symbol,
        eval_pt::Real,
        δ::Float64,
        profile_point::NamedTuple
    )::NamedTuple

Build scenarios for symmetric finite difference computation around an evaluation point.

Creates modifier scenarios at [eval_pt - δ, eval_pt + δ] and merges with fixed
profile variables. Validates that the two points are numerically distinct.

# Arguments
- `modifier`: Name of the modifier variable
- `eval_pt`: Center point for finite difference
- `δ`: Step size (already adjusted for numerical stability)
- `profile_point`: Fixed profile variables to hold constant

# Returns
- `NamedTuple` with all scenario variables (modifier + profile)

# Throws
- `ErrorException` if pt_minus ≈ pt_plus (numerical precision issue)
"""
function _build_finite_diff_scenarios(
    modifier::Symbol,
    eval_pt::Real,
    δ::Float64,
    profile_point::NamedTuple
)::NamedTuple
    # Build scenarios for symmetric (centered) finite difference: [eval_pt - δ, eval_pt + δ]
    # Rationale: Symmetric FD has O(δ²) truncation error vs O(δ) for forward/backward FD.
    # For smooth functions: f'(x) ≈ [f(x+δ) - f(x-δ)]/(2δ) + O(δ²)
    # This provides better approximation for same δ, or allows larger δ for same accuracy.
    pt_minus = eval_pt - δ
    pt_plus = eval_pt + δ

    # Validate that perturbations produced numerically distinct points
    # If this fails, floating-point precision is insufficient for the requested δ
    if !(pt_minus < pt_plus)
        error("Cannot compute finite difference at eval_point=$eval_pt with δ=$δ. " *
              "Points ($pt_minus, $pt_plus) are numerically identical. " *
              "Solutions: (1) Specify a larger delta (try delta=1e-6 or larger), " *
              "(2) Use a different evaluation point via `at` parameter, or " *
              "(3) If using inverse-transformed variables, specify explicit delta based on transformed scale.")
    end

    # Create modifier scenarios with both perturbed points
    # These will be passed to population_margins to compute AME at each point
    modifier_scenarios = NamedTuple{(modifier,)}(([pt_minus, pt_plus],))

    # Merge with fixed profile variables from current combination
    # Profile vars remain constant during the finite difference (only modifier varies)
    return merge(modifier_scenarios, profile_point)
end

"""
    _process_variable_contrasts(
        ames::EffectsResult,
        vars_vec::Vector{Symbol},
        modifier::Symbol,
        vcov_matrix::AbstractMatrix{<:Real},
        eval_pt::Real,
        δ::Float64,
        profile_point::NamedTuple,
        alpha::Real
    )::Vector{NamedTuple}

Process all variables and their contrasts to extract second difference results.

For each variable in `vars_vec`, finds all contrasts and computes the second difference
(derivative of AME with respect to modifier) using pairwise differences.

# Arguments
- `ames`: AME results from population_margins (at modifier ± δ)
- `vars_vec`: Vector of focal variables to process
- `modifier`: Modifier variable name
- `vcov_matrix`: Parameter covariance matrix
- `eval_pt`: Evaluation point for the modifier
- `δ`: Step size used in finite difference
- `profile_point`: Fixed profile variables (added to results)

# Returns
- Vector of NamedTuples, one per variable × contrast combination
"""
function _process_variable_contrasts(
    ames::EffectsResult,
    vars_vec::Vector{Symbol},
    modifier::Symbol,
    vcov_matrix::AbstractMatrix{<:Real},
    eval_pt::Real,
    δ::Float64,
    profile_point::NamedTuple,
    alpha::Real=0.05,
    contrast::Union{Nothing,AbstractString}=nothing
)::Vector{NamedTuple}
    results = NamedTuple[]
    idx = _effect_index(ames)

    for var in vars_vec
        # Identify all contrasts for this variable
        # Continuous variables have single "dy/dx" contrast
        # Categorical variables have multiple contrasts (e.g., "level2 - level1")
        _require_variable(idx, var)
        unique_contrasts = _variable_contrasts(idx, var)
        if !isnothing(contrast)
            label = _contrast_label(contrast)
            label in unique_contrasts || error("Variable :$var has no contrast '$contrast'. " *
                "Contrasts available: $(join(unique_contrasts, ", "))")
            unique_contrasts = [label]
        end

        # For each contrast, compute second difference (interaction effect)
        # This measures how the focal variable's effect changes with the modifier
        for contrast in unique_contrasts
            # Use _second_differences_pairwise to compute slope from the two AME points
            # modifier_type=:continuous ensures proper scaling by modifier distance
            second_diff_result = _second_differences_pairwise(
                ames, var, modifier, vcov_matrix;
                contrast = contrast,
                modifier_type = :continuous,
                alpha = alpha,
                include_gradients = true
            )

            if nrow(second_diff_result) == 0
                # This should never happen with valid inputs - indicates a bug or data corruption
                error("_second_differences_pairwise() returned empty result for variable :$var " *
                      "(contrast: '$contrast') at modifier :$modifier = $eval_pt. " *
                      "This indicates an internal error. Please report this issue with a reproducible example.")
            end

            # The unified second-difference row: second_diff is the slope
            # (AME[modifier + δ] - AME[modifier - δ]) / (2δ), i.e. ∂AME/∂modifier at eval_pt,
            # with modifier_type = :continuous and modifier_level1/2 = eval_pt ∓ δ
            sd_row = copy(second_diff_result[1, :])  # DataFrameRow → NamedTuple

            # Append the evaluation point and step to the shared schema
            base_result = merge(sd_row, (eval_point = eval_pt, delta_used = δ))

            # Merge profile variables into result for stratification tracking
            # This allows users to see which profile configuration each derivative corresponds to
            result_with_profile = merge(base_result, profile_point)
            push!(results, result_with_profile)
        end
    end

    return results
end

# ============================================================================
# at = :observed: the average derivative over the observed modifier values
# ============================================================================

# The request with the modifier column shifted by `shift` for every row (each row keeps its own
# other values); every other column of the prepared data is shared.
function _shifted_modifier_request(req::MarginsRequest, modifier::Symbol, shift::Real)
    col = _column(req.data_nt, modifier)
    col isa CategoricalArray && throw(ArgumentError("Modifier :$modifier is categorical in the fitted " *
        "model, so a local derivative is not defined. Pass modifier_type=:categorical to compare its levels."))
    shifted = _narrow_missing_eltype([ismissing(v) ? missing : Float64(v) + shift for v in col])
    return _with_data(req, merge(req.data_nt, NamedTuple{(modifier,)}((shifted,))))
end

# AME at modifier + δ and modifier − δ with the shift applied to every row, which is the average over
# observations of the cross-derivative ∂²ŷᵢ/∂x∂m at each row's own (xᵢ, mᵢ, …). The two results are
# stacked into one EffectsResult with a synthetic scenario column for the modifier (the shifts −δ and
# +δ) so the pairing, scaling by 2δ, delta method, df and structural-zero logic are the ones used for
# every other derivative; eval_point and the levels are NaN (no single evaluation point).
function _observed_derivative_rows(req::MarginsRequest, vars_vec, modifier, modifier_vals,
                                   profile_point, δ_raw; alpha, contrast)
    δ = _adjust_delta_for_eval_point(maximum(abs, modifier_vals), δ_raw)
    point_req = _with_scenarios(req, isempty(profile_point) ? nothing : profile_point)
    fit(shift) = _run(_shifted_modifier_request(point_req, modifier, shift))
    lo, hi = fit(-δ), fit(δ)
    n_lo, n_hi = length(lo.estimates), length(hi.estimates)
    n_lo == n_hi && lo.variables == hi.variables && lo.terms == hi.terms ||
        error("Internal error: the shifted-modifier results do not have the same rows.")
    pv = isnothing(hi.profile_values) ? NamedTuple() : hi.profile_values
    cells = merge(NamedTuple{(modifier,)}((vcat(fill(-δ, n_lo), fill(δ, n_hi)),)),
                  NamedTuple{keys(pv)}(Tuple(vcat(pv[k], pv[k]) for k in keys(pv))))
    md = copy(hi.metadata)
    md[:scenarios_vars] = collect(keys(cells))
    md[:groups_vars] = Symbol[]
    stacked = EffectsResult(
        vcat(lo.estimates, hi.estimates), vcat(lo.standard_errors, hi.standard_errors),
        vcat(lo.variables, hi.variables), vcat(lo.terms, hi.terms), cells, nothing,
        vcat(lo.gradients, hi.gradients), md,
    )
    _align_second_difference_df!(stacked, req.df)
    rows = _process_variable_contrasts(stacked, vars_vec, modifier, req.Σ, NaN, δ, profile_point, alpha, contrast)
    return [merge(r, (modifier_level1 = NaN, modifier_level2 = NaN)) for r in rows]
end

# ============================================================================
# Main Function
# ============================================================================

# The variables a local derivative holds fixed: `profile`, plus the other variables of an `at`
# NamedTuple (values given in `at` take precedence if conflict)
function _local_profile(at, profile::NamedTuple, modifier::Symbol)
    at isa NamedTuple || return profile
    other_vars = NamedTuple{Tuple(k for k in keys(at) if k != modifier)}(
        Tuple(at[k] for k in keys(at) if k != modifier)
    )
    return merge(profile, other_vars)
end

# Checks specific to a local derivative: a numeric modifier, and an `at` this function can read
function _check_local_derivative_inputs(cols::NamedTuple, modifier::Symbol, at)
    modifier_vals = _column(cols, modifier)
    if !all(x -> x isa Number || ismissing(x), modifier_vals)
        modifier_type = typeof(first(skipmissing(modifier_vals)))
        error("Modifier :$modifier must be numeric for a local derivative (`at`). " *
              "Found type: $modifier_type. " *
              "Continuous modifiers require numeric values (Int, Float64, etc.).")
    end
    if at isa NamedTuple
        if !haskey(at, modifier)
            provided_keys = join(string.(keys(at)), ", ")
            error("When `at` is a NamedTuple, it must include the modifier :$modifier. " *
                  "Provided keys: ($provided_keys). " *
                  "Example: at=($modifier=50, other_var=100)")
        end
    elseif !(at in (:mean, :median, :observed) || at isa Real || at isa AbstractVector)
        error("Invalid `at` parameter: must be :mean, :median, :observed, a Real, a Vector, or a NamedTuple")
    end
    return nothing
end

# The one request of a local derivative. The scenario variables (validated here, once) are the modifier,
# set to eval_point ± δ, and the profile variables; `at=:observed` shifts the modifier column instead, so
# it is not a scenario variable there.
function _local_derivative_request(model, cols::NamedTuple, vars_vec::Vector{Symbol}, modifier::Symbol, at,
                                   profile::NamedTuple; scale, backend, measure, contrasts, vcov, weights, alpha)
    held = collect(Symbol, keys(_local_profile(at, profile, modifier)))
    scenario_vars = at === :observed ? held : Symbol[modifier; held]
    return _normalize_request(model, cols; type=:effects, vars=vars_vec, scale, backend, measure,
                              contrasts, alpha, vcov, weights, scenario_vars,
                              changed_vars=Symbol[modifier; held])
end

# Engine for a local derivative ∂AME/∂modifier (symmetric finite difference of the AME at
# `at ± δ`, with the `profile` variables held fixed). Public entry point:
# `second_differences(model, data, variables, modifier; at=…)`, whose docstring describes the
# keywords (`at`, `profile`, `delta`, `scale`, `contrasts`, `alpha`, `p_adjust`, `include_gradients`).
# Here `vcov` is required and may be a matrix, a function of the model, or a CovarianceMatrices estimator.
function _second_differences_local(
    model::RegressionModel,
    data,
    variables::Union{Symbol,Vector{Symbol}},
    modifier::Symbol,
    vcov; # An AbstractMatrix, a Function returning one, or an estimator
    at::Union{Symbol,Real,AbstractVector,NamedTuple}=:observed,
    profile::NamedTuple=NamedTuple(),
    delta::Union{Symbol,Real}=:auto,
    scale::Symbol=:response,
    measure::Symbol=:effect,
    contrasts::Symbol=:baseline,
    alpha::Real=0.05,
    p_adjust::Symbol=:none,
    include_gradients::Bool=false,
    weights=:model,
    contrast::Union{Nothing,AbstractString}=nothing,
    backend::Symbol=:ad,
)
    _check_p_adjust(p_adjust)
    0 < alpha < 1 || error("alpha must be in (0, 1), got $alpha")
    cols = Tables.columntable(data)

    # 1. Validate inputs
    if !hasproperty(cols, modifier)
        error("Modifier variable :$modifier not found in data. " *
              "Available columns: $(_column_list(cols))")
    end

    modifier in keys(profile) && error("`profile` contains the modifier :$modifier; the modifier is " *
        "varied by the calculation (use `at` to choose where). Remove it from `profile`.")

    if any(x -> x isa Bool, skipmissing(_column(cols, modifier)))
        error("Modifier :$modifier is Bool, so a local derivative is not defined. Use `levels` or " *
              "leave modifier_type=:auto to compare false and true.")
    end
    _check_local_derivative_inputs(cols, modifier, at)

    # 2. Normalize variables to Vector{Symbol}; profile and focal variables must exist in data
    vars_vec = variables isa Symbol ? [variables] : variables
    _check_focal_and_profile_vars(cols, vars_vec, keys(profile))

    # 3. One request: the shared validators (`contrasts`, `weights`, ...) and Σ
    req = _local_derivative_request(model, cols, vars_vec, modifier, at, profile;
                                    scale, backend, measure, contrasts, vcov, weights, alpha)
    return _local_derivatives(req, cols, vars_vec, modifier; at, profile, delta, alpha, p_adjust,
                              include_gradients, contrast)
end

# The local derivatives of a validated request (`_local_derivative_request`): one AME computation per
# (profile point, evaluation point ± δ) from the request, with no keyword parsing or validation in
# between. `cols` is the data as given (the evaluation points and δ come from its modifier column).
function _local_derivatives(req::MarginsRequest, cols::NamedTuple, vars_vec::Vector{Symbol}, modifier::Symbol;
                            at, profile::NamedTuple, delta, alpha::Real, p_adjust::Symbol,
                            include_gradients::Bool, contrast)
    # 3. Extract modifier values from data
    modifier_vals = _column(cols, modifier)
    modifier_vals_clean = collect(skipmissing(modifier_vals))
    # `:mean` / `:median` use the weights too, so the point matches the weighted population
    wvec = req.weights
    keep = [!ismissing(v) for v in modifier_vals]
    wclean = isnothing(wvec) ? nothing : wvec[keep]

    # 4. Parse `at` parameter to determine evaluation points and complete profile specification
    # The `at` parameter supports multiple formats for flexibility:
    # - :mean/:median - symbolic locations based on data
    # - Real value - single specific evaluation point
    # - Vector - multiple evaluation points
    # - NamedTuple - complete profile specification (modifier + other variables)
    full_profile = _local_profile(at, profile, modifier)
    eval_points = if at isa NamedTuple
        # NamedTuple case: user provided complete profile including modifier value
        # Example: at=(age=40, income=50000, region="north")
        # The modifier value says where to evaluate the derivative; the other variables are
        # held fixed (merged into `full_profile` by _local_profile)
        modifier_val = at[modifier]
        modifier_val isa AbstractVector ? Float64.(modifier_val) : [Float64(modifier_val)]
    elseif at === :mean
        # Evaluate derivative at mean(modifier), convenient default
        [isnothing(wclean) ? mean(modifier_vals_clean) : _weighted_mean(modifier_vals_clean, wclean)]
    elseif at === :observed
        # Average derivative over the observed values: the modifier column is shifted, not set (see below)
        [NaN]
    elseif at === :median
        # Evaluate derivative at median(modifier), robust to outliers
        [isnothing(wclean) ? median(modifier_vals_clean) : _weighted_median(modifier_vals_clean, wclean)]
    elseif at isa Real
        # Evaluate at specific modifier value
        [Float64(at)]
    else # AbstractVector (checked by _check_local_derivative_inputs)
        # Evaluate at multiple modifier values (creates one result row per value)
        Float64.(at)
    end

    # 5. Compute delta (step size for finite difference)
    # Rationale: Delta controls the bias-variance tradeoff in FD approximation
    # - Too large: high bias (poor approximation of local derivative)
    # - Too small: high variance (numerical instability, cancellation error)
    # Auto delta = DEFAULT_DELTA_FACTOR × SD is empirically good for most cases
    δ_raw = if delta === :auto
        # Auto delta: DEFAULT_DELTA_FACTOR (1%) of standard deviation of modifier
        # This scales appropriately with the natural variation in the modifier
        sd_modifier = std(modifier_vals_clean)
        DEFAULT_DELTA_FACTOR * sd_modifier
    elseif delta isa Real
        Float64(delta)
    else
        error("Invalid `delta` parameter: must be :auto or a Real number")
    end

    if δ_raw <= 0
        error("Delta must be positive, got δ = $δ_raw. " *
              "Valid values: positive numbers (e.g., delta=0.01, delta=1.0, delta=100.0) or delta=:auto. " *
              "For variables with small variance, try delta=1e-6 to delta=0.1 range.")
    end

    # The minimum step (absolute near zero, relative for large |eval_pt|) is enforced per
    # evaluation point in _adjust_delta_for_eval_point

    # 6. Expand profile to Cartesian product if it contains vectors
    # Rationale: When profile=(age=[30,60], region=["north","south"]), we need to compute
    # derivatives at all 4 combinations: (30,north), (30,south), (60,north), (60,south)
    # This enables stratified analysis: "does the interaction vary across subgroups?"
    profile_combinations = _expand_profile_to_cartesian(full_profile)

    # 7. Covariance matrix: the request's, the one evaluation of `vcov` for the call
    vcov_matrix = req.Σ

    # 8. Main computation loop: iterate over all profile combinations and evaluation points
    # Overall algorithm:
    #   For each (profile_point, eval_point) combination:
    #     1. Compute AME at modifier values [eval_pt - δ, eval_pt + δ]
    #     2. Use symmetric FD to estimate ∂AME/∂modifier at eval_pt
    #     3. Apply delta method for standard errors
    # Result: derivatives stratified by profile and evaluated at specified points
    results = []

    for profile_point in profile_combinations
        for eval_pt in eval_points
            if at === :observed
                append!(results, _observed_derivative_rows(
                    req, vars_vec, modifier, modifier_vals_clean, profile_point, δ_raw; alpha, contrast,
                ))
                continue
            end
            # Adjust delta for numerical stability at this specific evaluation point
            # Accounts for magnitude of eval_pt to prevent floating-point issues
            δ = _adjust_delta_for_eval_point(eval_pt, δ_raw)

            # Build finite difference scenarios: modifier at [eval_pt ± δ] with fixed profile
            # This creates a counterfactual: "what if everyone had modifier = eval_pt ± δ?"
            full_scenarios = _build_finite_diff_scenarios(modifier, eval_pt, δ, profile_point)

            # Compute population average marginal effects at both perturbed modifier values
            # This is the computationally expensive step: evaluates AME across entire dataset
            # (one computation from the request, with the scenarios set to this point)
            ames = _run(_with_scenarios(req, full_scenarios))

            # t vs z from the model and the vcov as given (same rule as `_inference_df`)
            _align_second_difference_df!(ames, req.df)

            # Extract derivatives for all variables and contrasts using symmetric FD formula
            # slope = (AME[modifier + δ] - AME[modifier - δ]) / (2δ)
            point_results = _process_variable_contrasts(
                ames, vars_vec, modifier, vcov_matrix, eval_pt, δ, profile_point, alpha, contrast
            )

            # Accumulate results from this (profile_point, eval_point) combination
            append!(results, point_results)
        end
    end

    return _finalize_frame!(DataFrame(results), p_adjust, alpha, include_gradients; weights_source=req.weights_source)
end
