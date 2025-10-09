# at_point.jl
# Local derivative computations for second differences at specific points

using Statistics: mean, median, std

"""
    second_differences_at(
        model, data, variables, modifier, vcov;
        at=:mean, profile=NamedTuple(), delta=:auto, scale=:response
    )

Compute local derivative of marginal effects with respect to a continuous modifier
at a specified profile (holding other variables fixed).

Uses finite differences around a specified point to estimate ∂AME/∂modifier at that point.
Supports:
- **Multiple focal variables**: Compute derivatives for several effects simultaneously
- **Profile specification**: Hold other variables constant while varying only the modifier

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

# Returns
DataFrame with columns:
- `variable`: Focal variable name
- `contrast`: Contrast description (for categorical variables, e.g., "Protestant vs Catholic"; "derivative" for continuous)
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
    scale::Symbol=:response
)
    # 1. Validate inputs
    if !hasproperty(data, modifier)
        error("Modifier variable :$modifier not found in data")
    end

    modifier_vals = data[!, modifier]
    if !all(x -> x isa Number || ismissing(x), modifier_vals)
        error("Modifier :$modifier must be numeric for second_differences_at()")
    end

    # Validate profile variables exist
    for var in keys(profile)
        if !hasproperty(data, var)
            error("Profile variable :$var not found in data")
        end
    end

    # 2. Normalize variables to Vector{Symbol}
    vars_vec = variables isa Symbol ? [variables] : variables

    # Validate all variables exist in model
    for var in vars_vec
        if !hasproperty(data, var)
            error("Variable :$var not found in data")
        end
    end

    # 3. Extract modifier values from data
    modifier_vals_clean = collect(skipmissing(modifier_vals))

    # 4. Parse `at` parameter to get evaluation points and full profile
    eval_points, full_profile = if at isa NamedTuple
        # User provided full profile including modifier value
        if !haskey(at, modifier)
            error("When `at` is a NamedTuple, it must include the modifier :$modifier")
        end

        # Extract modifier value
        modifier_val = at[modifier]

        # Extract other variables (everything except modifier)
        other_vars = NamedTuple{Tuple(k for k in keys(at) if k != modifier)}(
            Tuple(at[k] for k in keys(at) if k != modifier)
        )

        # Merge with profile parameter
        merged_profile = merge(profile, other_vars)

        ([Float64(modifier_val)], merged_profile)
    elseif at === :mean
        ([mean(modifier_vals_clean)], profile)
    elseif at === :median
        ([median(modifier_vals_clean)], profile)
    elseif at isa Real
        ([Float64(at)], profile)
    elseif at isa Vector
        (Float64.(at), profile)
    else
        error("Invalid `at` parameter: must be :mean, :median, a Real, a Vector, or a NamedTuple")
    end

    # 5. Compute delta
    δ_raw = if delta === :auto
        # Auto delta: 1% of standard deviation
        sd_modifier = std(modifier_vals_clean)
        0.01 * sd_modifier
    elseif delta isa Real
        Float64(delta)
    else
        error("Invalid `delta` parameter: must be :auto or a Real number")
    end

    if δ_raw <= 0
        error("Delta must be positive, got δ = $δ_raw")
    end

    # Ensure delta is not too small for floating-point precision
    # Minimum absolute delta to prevent numerical issues
    min_abs_delta = 1e-10

    # Ensure the raw delta is at least this minimum
    δ_raw = max(δ_raw, min_abs_delta)

    # 6. Expand profile to Cartesian product if it contains vectors
    profile_combinations = _expand_profile_to_cartesian(full_profile)

    # 7. For each profile combination and evaluation point, compute symmetric FD for all variables
    results = []

    for profile_point in profile_combinations
        for eval_pt in eval_points
        # For each point, ensure delta is large enough relative to the point value
        # to avoid floating-point precision issues
        δ = if abs(eval_pt) > 1.0
            # For large values, use relative delta (at least 1e-8 relative error)
            max(δ_raw, abs(eval_pt) * 1e-8)
        else
            # For small values, use absolute delta with minimum
            max(δ_raw, 1e-9)
        end

        # Build scenarios for modifier: [eval_pt - δ, eval_pt + δ]
        pt_minus = eval_pt - δ
        pt_plus = eval_pt + δ

        # Final check: if points are still identical, error out
        if pt_minus ≈ pt_plus
            error("Cannot compute finite difference at eval_point=$eval_pt with δ=$δ. " *
                  "Points ($pt_minus, $pt_plus) are numerically identical. " *
                  "Specify a larger `delta` parameter.")
        end

        modifier_scenarios = NamedTuple{(modifier,)}(([pt_minus, pt_plus],))

        # Merge with fixed profile variables from current combination
        # Profile vars get their fixed value (not a vector)
        full_scenarios = merge(modifier_scenarios, profile_point)

        # Compute AME at both modifier points for ALL variables
        # population_margins expects vcov to be a function, so wrap matrix if needed
        vcov_func = if vcov isa Function
            vcov  # Already a function
        else
            # Wrap the matrix in a function
            _model -> vcov
        end

        ames = population_margins(
            model, data;
            scenarios=full_scenarios,
            vars=vars_vec,
            type=:effects,
            scale=scale,
            vcov=vcov_func
        )

        # Get the vcov matrix for gradient computations
        vcov_matrix = if vcov isa Function
            vcov(model)
        else
            vcov  # Already a matrix
        end

        # For each variable, compute second difference
        for var in vars_vec
            # Check if this is a categorical variable (has multiple contrasts)
            var_str = string(var)
            var_indices = findall(i -> ames.variables[i] == var_str, 1:length(ames.estimates))

            if isempty(var_indices)
                error("Variable :$var not found in AME results")
            end

            # Get unique contrasts for this variable
            unique_contrasts = unique(ames.terms[var_indices])

            # For each contrast, compute second difference
            for contrast in unique_contrasts
                sd_df = second_differences_pairwise(
                    ames, var, modifier, vcov_matrix;
                    contrast = contrast,
                    modifier_type = :continuous
                )

                if nrow(sd_df) == 0
                    # Skip if no valid contrasts (shouldn't happen, but defensive)
                    continue
                end

                # Extract the slope (already scaled by delta in second_differences_pairwise)
                slope = sd_df.second_diff[1]
                se_slope = sd_df.se[1]
                gradient = sd_df.gradient[1]

                # Build result tuple including profile variables
                base_result = (
                    variable = var,
                    contrast = contrast,
                    modifier = modifier,
                    eval_point = eval_pt,
                    derivative = slope,
                    se = se_slope,
                    z_stat = slope / se_slope,
                    p_value = sd_df.p_value[1],
                    delta_used = δ,
                    gradient = gradient
                )

                # Add profile variables to result
                result_with_profile = merge(base_result, profile_point)
                push!(results, result_with_profile)
            end
        end
        end
    end

    return DataFrame(results)
end
