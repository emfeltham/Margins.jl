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
# Helper functions for the local derivative (_local_derivatives)
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
- `ArgumentError` if pt_minus ≈ pt_plus (numerical precision issue)
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
        throw(ArgumentError("Cannot compute finite difference at eval_point=$eval_pt with δ=$δ. " *
              "Points ($pt_minus, $pt_plus) are numerically identical. " *
              "Solutions: (1) Specify a larger delta (try delta=1e-6 or larger), " *
              "(2) Use a different evaluation point via `at` parameter, or " *
              "(3) If using inverse-transformed variables, specify explicit delta based on transformed scale."))
    end

    # Create modifier scenarios with both perturbed points
    # These will be passed to population_margins to compute AME at each point
    modifier_scenarios = NamedTuple{(modifier,)}(([pt_minus, pt_plus],))

    # Merge with fixed profile variables from current combination
    # Profile vars remain constant during the finite difference (only modifier varies)
    return merge(modifier_scenarios, profile_point)
end

"""
    _process_variable_contrasts(ames, vars_vec, modifier, eval_pt, δ, profile_point,
                                alpha=0.05, contrast=nothing) -> Vector{NamedTuple}

The local-derivative rows of one AME computation at modifier = eval_pt ∓ δ (or shifted by ∓δ):
for every `groups=` stratum of `ames` (`_result_strata`; one stratum without groups), every focal
variable and every contrast, the slope (AME[+δ] − AME[−δ]) / (2δ) with its delta-method SE
(`_second_differences_pairwise` with `modifier_type=:continuous`; the standard errors use
`ames.inference.vcov`). Each row gets `eval_point`, `delta_used`, the stratum's `over_<var>`
values and the `profile` point.
"""
function _process_variable_contrasts(
    ames::EffectsResult,
    vars_vec::Vector{Symbol},
    modifier::Symbol,
    eval_pt::Real,
    δ::Float64,
    profile_point::NamedTuple,
    alpha::Real=0.05,
    contrast::Union{Nothing,AbstractString}=nothing
)::Vector{NamedTuple}
    results = NamedTuple[]
    for (stratum, over) in _result_strata(ames, modifier)
        idx = _effect_index(stratum)
        for var in vars_vec
            # Every contrast of the variable ("dy/dx" for a continuous one), or the one asked for
            _require_variable(idx, var)
            unique_contrasts = _variable_contrasts(idx, var)
            if !isnothing(contrast)
                label = _contrast_label(contrast)
                label in unique_contrasts || throw(ArgumentError("Variable :$var has no contrast " *
                    "'$contrast'. Contrasts available: $(join(unique_contrasts, ", "))"))
                unique_contrasts = [label]
            end

            for c in unique_contrasts
                # The slope between the two AME points (modifier_type=:continuous divides by 2δ)
                sd = _second_differences_pairwise(stratum, var, modifier; contrast = c,
                                                  modifier_type = :continuous, alpha, include_gradients = true)
                nrow(sd) == 1 || error("Internal error: expected one second difference for :$var " *
                    "(contrast '$c') at :$modifier = $eval_pt, got $(nrow(sd)). Please report this issue.")
                # The unified second-difference row (∂AME/∂modifier at eval_pt, modifier_level1/2 =
                # eval_pt ∓ δ), then the evaluation point and step, the stratum and the profile point
                row = merge(copy(sd[1, :]), (eval_point = eval_pt, delta_used = δ))
                push!(results, merge(row, over, profile_point))
            end
        end
    end
    return results
end

# ============================================================================
# at = :observed: the average derivative over the observed modifier values
# ============================================================================

# AME at modifier + δ and modifier − δ with the shift applied to every row, which is the average over
# observations of the cross-derivative ∂²ŷᵢ/∂x∂m at each row's own (xᵢ, mᵢ, …). It is one computation
# with two contexts, the counterfactuals `modifier => Shift(-δ)` and `Shift(+δ)` (with the `profile`
# point set as well), whose context column holds the shifts −δ and +δ, so the pairing, scaling by 2δ,
# delta method, df and structural-zero logic are the ones used for every other derivative; eval_point
# and the levels are NaN (no single evaluation point).
function _observed_derivative_rows(req::MarginsRequest, vars_vec, modifier, modifier_vals,
                                   profile_point, δ_raw; alpha, contrast)
    δ = _adjust_delta_for_eval_point(maximum(abs, modifier_vals), δ_raw)
    _column(req.data_nt, modifier) isa CategoricalArray && throw(ArgumentError("Modifier :$modifier is " *
        "categorical in the fitted model, so a local derivative is not defined. Pass " *
        "modifier_type=:categorical to compare its levels."))
    shifts = merge(NamedTuple{(modifier,)}(([Shift(-δ), Shift(δ)],)), profile_point)
    ames = _run(_with_scenarios(req, shifts))
    rows = _process_variable_contrasts(ames, vars_vec, modifier, NaN, δ, profile_point, alpha, contrast)
    return [merge(r, (modifier_level1 = NaN, modifier_level2 = NaN)) for r in rows]
end

# ============================================================================
# The local derivative: request and computation
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
        throw(ArgumentError("Modifier :$modifier must be numeric for a local derivative (`at`). " *
              "Found type: $modifier_type. " *
              "Continuous modifiers require numeric values (Int, Float64, etc.)."))
    end
    if at isa NamedTuple
        if !haskey(at, modifier)
            provided_keys = join(string.(keys(at)), ", ")
            throw(ArgumentError("When `at` is a NamedTuple, it must include the modifier :$modifier. " *
                  "Provided keys: ($provided_keys). " *
                  "Example: at=($modifier=50, other_var=100)"))
        end
    elseif !(at in (:mean, :median, :observed) || at isa Real || at isa AbstractVector)
        throw(ArgumentError("Invalid `at` parameter: must be :mean, :median, :observed, a Real, a " *
                            "Vector, or a NamedTuple"))
    end
    return nothing
end

# The one request of a local derivative. The scenario variables (validated here, once) are the modifier,
# set to eval_point ± δ, and the profile variables. `at=:observed` shifts the modifier (`Shift(±δ)`)
# rather than setting it, so it is not validated as a scenario variable there (it is still a changed
# variable, which the derived-column detector checks).
function _local_derivative_request(model, cols::NamedTuple, vars_vec::Vector{Symbol}, modifier::Symbol, at,
                                   profile::NamedTuple; scale, backend, measure, contrasts, vcov, weights, alpha,
                                   df=nothing, groups=nothing, warn::Bool=true)
    held = collect(Symbol, keys(_local_profile(at, profile, modifier)))
    scenario_vars = at === :observed ? held : Symbol[modifier; held]
    return _normalize_request(model, cols; type=:effects, vars=vars_vec, scale, backend, measure,
                              contrasts, alpha, vcov, weights, df, groups, scenario_vars,
                              changed_vars=Symbol[modifier; held], warn)
end

# Where `at=:mean` / `:median` put the derivative: the (weighted) mean / median of the modifier values
# `vals` (weights `w`, or `nothing`)
_local_eval_point(at::Symbol, vals, w) = at === :mean ?
    (isnothing(w) ? mean(vals) : _weighted_mean(vals, w)) :
    (isnothing(w) ? median(vals) : _weighted_median(vals, w))

# The evaluation points of `at` other than :observed, for the modifier values `vals` (non-missing, with
# weights `w` or `nothing`)
function _local_eval_points(at, modifier::Symbol, vals, w)::Vector{Float64}
    if at isa NamedTuple
        # The modifier entry says where; the other variables are held fixed (`_local_profile`)
        v = at[modifier]
        return v isa AbstractVector ? Float64.(v) : [Float64(v)]
    elseif at === :mean || at === :median
        return [_local_eval_point(at, vals, w)]
    elseif at isa Real
        return [Float64(at)]
    else  # AbstractVector (checked by _check_local_derivative_inputs)
        return Float64.(at)
    end
end

# The finite-difference step before its per-point adjustment: 1% of the modifier's SD (`:auto`) or the
# given value. One step for the whole call, also with `groups=` (a numerical setting, not part of the
# estimand).
function _local_delta(delta, vals)::Float64
    δ_raw = if delta === :auto
        DEFAULT_DELTA_FACTOR * std(vals)
    elseif delta isa Real
        Float64(delta)
    else
        throw(ArgumentError("Invalid `delta` parameter: must be :auto or a Real number"))
    end
    δ_raw > 0 || throw(ArgumentError("Delta must be positive, got δ = $δ_raw. " *
        "Valid values: positive numbers (e.g., delta=0.01, delta=1.0, delta=100.0) or delta=:auto. " *
        "For variables with small variance, try delta=1e-6 to delta=0.1 range."))
    return δ_raw
end

"""
    _local_derivatives(req, cols, vars_vec, modifier; at, profile, delta, p_adjust,
                       include_gradients, contrast) -> DataFrame

The local derivatives ∂AME/∂modifier of a validated request (`_local_derivative_request`): one AME
computation per (profile point, evaluation point) from the request, at modifier = point ± δ (or every
row's modifier shifted by ±δ for `at=:observed`), with no keyword parsing or validation in between.
`cols` is the data restricted to the estimation rows (the evaluation points and δ come from its
modifier column). The significance level is the request's.

With `groups=` (in the request) each AME computation covers every stratum, and each stratum gets its
own rows (`_process_variable_contrasts`). `at=:mean` / `:median` then evaluate at each stratum's own
(weighted) mean / median, so a stratum's row is the one the same call would give on that stratum's data
alone: one computation per stratum (`_with_group_specs`). δ is the call's (from all rows).
"""
function _local_derivatives(req::MarginsRequest, cols::NamedTuple, vars_vec::Vector{Symbol}, modifier::Symbol;
                            at, profile::NamedTuple, delta, p_adjust::Symbol,
                            include_gradients::Bool, contrast)
    alpha = req.alpha
    # Modifier values (non-missing) and their weights: `:mean` / `:median` use the weights too, so the
    # point matches the weighted population
    modifier_vals = _column(cols, modifier)
    keep = findall(!ismissing, modifier_vals)
    vals = modifier_vals[keep]
    w = isnothing(req.weights) ? nothing : req.weights[keep]
    δ_raw = _local_delta(delta, vals)

    # (request, evaluation points) pairs: the whole request, or with a stratum-specific point one pair
    # per non-empty stratum
    plan = if at === :observed
        [(req, [NaN])]
    elseif (at === :mean || at === :median) && !isnothing(req.groups)
        pairs_ = Tuple{MarginsRequest,Vector{Float64}}[]
        for gspec in req.group_specs
            rows = intersect(_group_row_indices(gspec, req.data_nt), keep)
            isempty(rows) && continue
            wg = isnothing(req.weights) ? nothing : req.weights[rows]
            push!(pairs_, (_with_group_specs(req, [gspec]), [_local_eval_point(at, modifier_vals[rows], wg)]))
        end
        pairs_
    else
        [(req, _local_eval_points(at, modifier, vals, w))]
    end

    # Profile points: the Cartesian product of `profile` and the held variables of an `at` NamedTuple
    results = NamedTuple[]
    for profile_point in _expand_profile_to_cartesian(_local_profile(at, profile, modifier))
        for (r, eval_points) in plan, eval_pt in eval_points
            if at === :observed
                append!(results, _observed_derivative_rows(r, vars_vec, modifier, vals, profile_point,
                                                           δ_raw; alpha, contrast))
                continue
            end
            # The step for this point (relative for a large |eval_pt|), and the AME at eval_pt ∓ δ for
            # everyone (in every stratum), with the profile point set
            δ = _adjust_delta_for_eval_point(eval_pt, δ_raw)
            ames = _run(_with_scenarios(r, _build_finite_diff_scenarios(modifier, eval_pt, δ, profile_point)))
            append!(results, _process_variable_contrasts(ames, vars_vec, modifier, eval_pt, δ, profile_point,
                                                         alpha, contrast))
        end
    end

    return _finalize_frame!(DataFrame(results), p_adjust, alpha, include_gradients; weights_source=req.weights_source)
end
