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

# Minimum absolute delta to prevent catastrophic cancellation
# When δ is too small, (f(x+δ) - f(x-δ)) loses significant digits to floating-point error
# 1e-10 is well above machine epsilon (≈2e-16) while still allowing fine-grained derivatives
const MIN_ABSOLUTE_DELTA = 1e-10

# Relative delta factor for large evaluation points (|eval_pt| > 1)
# Ensures δ ≥ |eval_pt| × 1e-8 to maintain numerical distinctness of (eval_pt ± δ)
# Stays well above machine epsilon while scaling proportionally with magnitude
const RELATIVE_DELTA_FACTOR = 1e-8

# Minimum delta for small evaluation points (|eval_pt| ≤ 1)
# Prevents δ → 0 when eval_pt ≈ 0, which would cause division issues in slope calculation
# Slightly larger than MIN_ABSOLUTE_DELTA to ensure robustness near zero
const SMALL_VALUE_MIN_DELTA = 1e-9

# ============================================================================
# Helper Functions for second_differences_at
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
    if pt_minus ≈ pt_plus
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
        vcov_matrix::Matrix{Float64},
        eval_pt::Real,
        δ::Float64,
        profile_point::NamedTuple
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
    vcov_matrix::Matrix{Float64},
    eval_pt::Real,
    δ::Float64,
    profile_point::NamedTuple
)::Vector{NamedTuple}
    results = NamedTuple[]

    for var in vars_vec
        # Identify all contrasts for this variable
        # Continuous variables have single "dy/dx" contrast
        # Categorical variables have multiple contrasts (e.g., "level2 - level1")
        var_str = string(var)
        var_indices = findall(i -> ames.variables[i] == var_str, 1:length(ames.estimates))

        if isempty(var_indices)
            available_vars = unique(ames.variables)
            error("Variable :$var not found in AME results. " *
                  "This may indicate the variable was not included in the model or scenarios. " *
                  "Variables found in AME results: $(join(available_vars, ", "))")
        end

        # Get unique contrasts for this variable
        unique_contrasts = unique(ames.terms[var_indices])

        # For each contrast, compute second difference (interaction effect)
        # This measures how the focal variable's effect changes with the modifier
        for contrast in unique_contrasts
            # Use second_differences_pairwise to compute slope from the two AME points
            # modifier_type=:continuous ensures proper scaling by modifier distance
            second_diff_result = second_differences_pairwise(
                ames, var, modifier, vcov_matrix;
                contrast = contrast,
                modifier_type = :continuous
            )

            if nrow(second_diff_result) == 0
                # This should never happen with valid inputs - indicates a bug or data corruption
                error("second_differences_pairwise() returned empty result for variable :$var " *
                      "(contrast: '$contrast') at modifier :$modifier = $eval_pt. " *
                      "This indicates an internal error. Please report this issue with a reproducible example.")
            end

            # Extract slope and inference statistics
            # slope = (AME[modifier + δ] - AME[modifier - δ]) / (2δ)
            # This is the derivative ∂AME/∂modifier at eval_pt
            slope = second_diff_result.second_diff[1]
            se_slope = second_diff_result.se[1]
            gradient = second_diff_result.gradient[1]  # For custom variance calculations

            # Build result tuple with all relevant information
            base_result = (
                variable = var,
                contrast = contrast,
                modifier = modifier,
                eval_point = eval_pt,
                derivative = slope,
                se = se_slope,
                z_stat = slope / se_slope,
                p_value = second_diff_result.p_value[1],
                delta_used = δ,
                gradient = gradient
            )

            # Merge profile variables into result for stratification tracking
            # This allows users to see which profile configuration each derivative corresponds to
            result_with_profile = merge(base_result, profile_point)
            push!(results, result_with_profile)
        end
    end

    return results
end

# ============================================================================
# Main Function
# ============================================================================

"""
    second_differences_at(
        model, data, variables, modifier, vcov;
        at=:mean, profile=NamedTuple(), delta=:auto, scale=:response, contrasts=:baseline
    )

Compute local derivative of marginal effects with respect to a continuous modifier
at a specified profile (holding other variables fixed).

Uses finite differences around a specified point to estimate ∂AME/∂modifier at that point.
Supports:
- **Multiple focal variables**: Compute derivatives for several effects simultaneously
- **Profile specification**: Hold other variables constant while varying only the modifier
- **Flexible contrasts**: Choose baseline or pairwise contrasts for categorical variables

# Arguments
- `model::RegressionModel`: Fitted model
- `data`: Data frame used to fit the model
- `variables::Union{Symbol,Vector{Symbol}}`: Focal variable(s) to compute derivatives for
  - Single Symbol: One focal variable
  - Vector of Symbols: Multiple focal variables (all get same modifier)
- `modifier::Symbol`: Continuous modifier variable (the variable to differentiate with respect to)
- `vcov`: Parameter covariance matrix or function
  - `Matrix{Float64}`: Pre-computed covariance matrix
  - `Function`: Covariance function (e.g., `GLM.vcov`) - recommended for mixed models

# Keyword Arguments
- `at::Union{Symbol,Real,Vector,NamedTuple}`: Where to evaluate the derivative for the modifier
  - `:mean`: At mean(modifier) [default]
  - `:median`: At median(modifier)
  - Numeric value: At specified modifier value
  - Vector of values: Multiple evaluation points for modifier
  - NamedTuple: Explicit profile including modifier value (e.g., `(income=50000, age=40, region="north")`)
- `profile::NamedTuple`: Additional variables to hold fixed (NOT varied in finite difference)
  - Default: `NamedTuple()` (no additional variables fixed)
  - **Scalar values**: `(age=40, region="north")` - holds age and region constant while varying modifier
  - **Vector values**: `(socio4=[false, true])` - stratified estimation at each level (Cartesian product)
  - Combined with `at` to create full evaluation profile
- `delta::Union{Symbol,Real}`: Step size for finite difference on the modifier
  - `:auto`: 0.01 * sd(modifier) [default], automatically adjusted if too small
  - Numeric value: Explicit step size (recommended for inverse/transformed variables)
  - **Note**: For inverse or transformed modifiers with low variance, specify explicit delta
- `scale::Symbol`: Scale for predictions (:link or :response)
- `contrasts::Symbol`: Contrast coding for categorical focal variables
  - `:baseline`: Each level minus baseline (default) - produces K-1 contrasts for K levels
  - `:pairwise`: All pairwise comparisons - produces K(K-1)/2 contrasts for K levels
  - Only affects categorical variables; continuous variables unaffected

# Returns
DataFrame with columns:
- `variable`: Focal variable name
- `contrast`: Contrast description (for categorical variables, e.g., "Protestant - Catholic"; "dy/dx" for continuous)
- `modifier`: Modifier variable name
- `eval_point`: Point where derivative is evaluated (for modifier)
- `derivative`: ∂AME/∂modifier (per unit change in modifier)
- `se`: Standard error of derivative
- `z_stat`: Z-statistic
- `p_value`: P-value for H₀: derivative = 0
- `delta_used`: Actual delta used for FD
- `gradient`: Gradient vector (∂derivative/∂θ) for custom variance calculations
- Additional columns for each variable in `profile` (if specified)

**Note**: For categorical focal variables, one row is returned per contrast × eval_point × profile combination

# Examples
```julia
using Margins, GLM, DataFrames

# Local derivative at mean (simple case)
sd = second_differences_at(model, df, :education, :income, vcov(model))
# → Derivative of education effect w.r.t. income, evaluated at mean(income)

# At median with custom step size
sd = second_differences_at(
    model, df, :education, :income, vcov(model);
    at=:median, delta=1000
)

# At specific modifier value with other variables held constant
sd = second_differences_at(
    model, df, :education, :income, vcov(model);
    at=50000, # Evaluate at income=\$50k
    profile = (
        age=40, # Hold age=40 fixed
        region="north" # Hold region="north" fixed
    ) 
) 
# → "At income=\$50k (with age=40, region=north), how does education effect change per \$1k?"

# Multiple profiles: vary modifier, hold others constant
sd = second_differences_at(
    model, df, :education, :distance, vcov(model);
    at=[0.1, 0.5, 0.9],    # Evaluate at 3 distance values
    profile = (
        age=45, # Hold age constant
        income=60000 # Hold income constant
    )
)
# → 3 rows, one per distance evaluation point

# Full explicit profile (NamedTuple for `at`)
sd = second_differences_at(model, df, :education, :income, vcov(model);
                          at=(income=50000, age=40, region="north"))
# → Equivalent to: at=50000, profile=(age=40, region="north")

# Multiple focal variables (NEW: compare how different effects vary with modifier)
sd = second_differences_at(model, df, [:education, :experience, :gender], :age, vcov(model);
                          at=:mean)
# → 3 rows: derivatives of education, experience, and gender effects w.r.t. age
# → Compare which effects vary most with age

# Multiple variables with profile
sd = second_differences_at(model, df, [:x1, :x2, :x3], :income, vcov(model);
                          at=50000,
                          profile=(age=40, region="north"))
# → How do x1, x2, x3 effects vary with income at this specific profile?

# Multiple variables × multiple evaluation points
sd = second_differences_at(model, df, [:education, :experience], :age, vcov(model);
                          at=[30, 45, 60])
# → 6 rows: (education, experience) × (age=30, 45, 60)
# → Compare education vs experience effect variation across age ranges

# Vector-valued profile (stratified estimation)
sd = second_differences_at(model, df, :education, :income, vcov(model);
                          at=50000,
                          profile=(socio4=[false, true],       # Separate estimation for each socio4 level
                                  region="north"))
# → 2 rows: one for socio4=false, one for socio4=true
# → "At income=\$50k (region=north), how does education effect vary with socio4?"

# Multiple profile vectors (Cartesian product)
sd = second_differences_at(model, df, :education, :income, vcov(model);
                          at=[40000, 60000],
                          profile=(socio4=[false, true],
                                  gender=["male", "female"]))
# → 8 rows: 2 income levels × 2 socio4 levels × 2 gender levels
# → Stratified analysis across all combinations

# Pairwise contrasts for categorical variable (3 levels)
sd = second_differences_at(model, df, :religion, :age, vcov(model);
                          contrasts=:pairwise)
# → 3 rows with baseline contrasts=:baseline (default): Protestant - Catholic, None - Catholic
# → 3 rows with contrasts=:pairwise: Protestant - Catholic, None - Catholic, None - Protestant
# → Pairwise provides all K(K-1)/2 comparisons instead of just K-1 baseline comparisons

# Compare baseline vs pairwise contrasts
sd_baseline = second_differences_at(model, df, :religion, :age, vcov(model); contrasts=:baseline)
sd_pairwise = second_differences_at(model, df, :religion, :age, vcov(model); contrasts=:pairwise)
# → baseline: 2 rows (K-1 contrasts for K=3 levels)
# → pairwise: 3 rows (K(K-1)/2 = 3 contrasts for K=3 levels)
```

# Statistical Notes
- Uses two-point finite difference: (AME(at + δ) - AME(at - δ)) / (2δ)
- Delta method SE: SE(slope) = SE(AME₊ - AME₋) / (2δ)
- Symmetric FD provides better approximation than one-sided
- Choice of δ trades off bias vs variance
- Profile variables are held **constant** - only modifier is varied by ± δ

# See Also
- `second_differences()`: For discrete contrasts between pre-computed AME values
- `second_differences_pairwise()`: All pairwise modifier comparisons
"""
function second_differences_at(
    model::RegressionModel,
    data,
    variables::Union{Symbol,Vector{Symbol}},
    modifier::Symbol,
    vcov; # This can be Matrix{Float64} or Function
    at::Union{Symbol,Real,Vector,NamedTuple}=:mean,
    profile::NamedTuple=NamedTuple(),
    delta::Union{Symbol,Real}=:auto,
    scale::Symbol=:response,
    contrasts::Symbol=:baseline
)
    # 1. Validate inputs
    if !hasproperty(data, modifier)
        available_cols = join(string.(names(data)), ", ")
        error("Modifier variable :$modifier not found in data. " *
              "Available columns: $available_cols")
    end

    # Validate contrasts parameter
    if !(contrasts in (:baseline, :pairwise))
        error("Invalid contrasts parameter: $contrasts. " *
              "Must be :baseline or :pairwise.")
    end

    modifier_vals = data[!, modifier]
    if !all(x -> x isa Number || ismissing(x), modifier_vals)
        modifier_type = typeof(first(skipmissing(modifier_vals)))
        error("Modifier :$modifier must be numeric for second_differences_at(). " *
              "Found type: $modifier_type. " *
              "Continuous modifiers require numeric values (Int, Float64, etc.).")
    end

    # Validate profile variables exist
    for var in keys(profile)
        if !hasproperty(data, var)
            available_cols = join(string.(names(data)), ", ")
            error("Profile variable :$var not found in data. " *
                  "Available columns: $available_cols")
        end
    end

    # 2. Normalize variables to Vector{Symbol}
    vars_vec = variables isa Symbol ? [variables] : variables

    # Validate all variables exist in data
    for var in vars_vec
        if !hasproperty(data, var)
            available_cols = join(string.(names(data)), ", ")
            error("Focal variable :$var not found in data. " *
                  "Available columns: $available_cols")
        end
    end

    # 3. Extract modifier values from data
    modifier_vals_clean = collect(skipmissing(modifier_vals))

    # 4. Parse `at` parameter to determine evaluation points and complete profile specification
    # The `at` parameter supports multiple formats for flexibility:
    # - :mean/:median - symbolic locations based on data
    # - Real value - single specific evaluation point
    # - Vector - multiple evaluation points
    # - NamedTuple - complete profile specification (modifier + other variables)
    eval_points, full_profile = if at isa NamedTuple
        # NamedTuple case: user provided complete profile including modifier value
        # Example: at=(age=40, income=50000, region="north")
        # Split into modifier value (for evaluation) and profile variables (held fixed)
        if !haskey(at, modifier)
            provided_keys = join(string.(keys(at)), ", ")
            error("When `at` is a NamedTuple, it must include the modifier :$modifier. " *
                  "Provided keys: ($provided_keys). " *
                  "Example: at=($modifier=50, other_var=100)")
        end

        # Extract modifier value to determine where to evaluate derivative
        modifier_val = at[modifier]

        # Extract other variables (everything except modifier) as profile constraints
        other_vars = NamedTuple{Tuple(k for k in keys(at) if k != modifier)}(
            Tuple(at[k] for k in keys(at) if k != modifier)
        )

        # Merge with profile parameter (profile parameter takes precedence if conflict)
        merged_profile = merge(profile, other_vars)

        ([Float64(modifier_val)], merged_profile)
    elseif at === :mean
        # Evaluate derivative at mean(modifier), convenient default
        ([mean(modifier_vals_clean)], profile)
    elseif at === :median
        # Evaluate derivative at median(modifier), robust to outliers
        ([median(modifier_vals_clean)], profile)
    elseif at isa Real
        # Evaluate at specific modifier value
        ([Float64(at)], profile)
    elseif at isa Vector
        # Evaluate at multiple modifier values (creates one result row per value)
        (Float64.(at), profile)
    else
        error("Invalid `at` parameter: must be :mean, :median, a Real, a Vector, or a NamedTuple")
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

    # Enforce minimum delta to prevent catastrophic cancellation in floating-point arithmetic
    # When δ is too small, (f(x+δ) - f(x-δ)) suffers from subtractive cancellation
    # losing significant digits even though both function evaluations are accurate
    δ_raw = max(δ_raw, MIN_ABSOLUTE_DELTA)

    # 6. Expand profile to Cartesian product if it contains vectors
    # Rationale: When profile=(age=[30,60], region=["north","south"]), we need to compute
    # derivatives at all 4 combinations: (30,north), (30,south), (60,north), (60,south)
    # This enables stratified analysis: "does the interaction vary across subgroups?"
    profile_combinations = _expand_profile_to_cartesian(full_profile)

    # 7. Prepare vcov function and matrix once (avoid repeated wrapping in loops)
    vcov_func = if vcov isa Function
        vcov  # Already a function
    else
        # Wrap the matrix in a function for population_margins
        _model -> vcov
    end

    vcov_matrix = if vcov isa Function
        vcov(model)
    else
        vcov  # Already a matrix
    end

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
            # Adjust delta for numerical stability at this specific evaluation point
            # Accounts for magnitude of eval_pt to prevent floating-point issues
            δ = _adjust_delta_for_eval_point(eval_pt, δ_raw)

            # Build finite difference scenarios: modifier at [eval_pt ± δ] with fixed profile
            # This creates a counterfactual: "what if everyone had modifier = eval_pt ± δ?"
            full_scenarios = _build_finite_diff_scenarios(modifier, eval_pt, δ, profile_point)

            # Compute population average marginal effects at both perturbed modifier values
            # This is the computationally expensive step: evaluates AME across entire dataset
            # Pass contrasts parameter for categorical variables
            ames = population_margins(
                model, data;
                scenarios=full_scenarios,
                vars=vars_vec,
                type=:effects,
                scale=scale,
                vcov=vcov_func,
                contrasts=contrasts
            )

            # Extract derivatives for all variables and contrasts using symmetric FD formula
            # slope = (AME[modifier + δ] - AME[modifier - δ]) / (2δ)
            point_results = _process_variable_contrasts(
                ames, vars_vec, modifier, vcov_matrix, eval_pt, δ, profile_point
            )

            # Accumulate results from this (profile_point, eval_point) combination
            append!(results, point_results)
        end
    end

    return DataFrame(results)
end
