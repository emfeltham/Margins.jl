# profile/core.jl
# Main profile_margins() function with reference grid approach

"""
    profile_margins(model, data, reference_grid; type=:effects, vars=nothing, scale=:response, backend=:ad, measure=:effect, contrasts=:baseline, ci_alpha=0.05, vcov=GLM.vcov) -> Union{EffectsResult, PredictionsResult}

Compute profile marginal effects or adjusted predictions at specific covariate combinations.

This function evaluates effects/predictions at representative points or user-specified scenarios,
implementing the "Profile" approach from the 2×2 framework (Population vs Profile × Effects vs Predictions).
It provides marginal effects at the mean (MEM), marginal effects at representative values (MER),
or adjusted predictions at specific profiles (APM/APR).

# Arguments
- `model`: Fitted statistical model supporting `coef()` and `vcov()` methods
- `data`: Data table (DataFrame, NamedTuple, or any Tables.jl-compatible format)
- `reference_grid`: DataFrame specifying covariate combinations for analysis
  - Use reference grid builders: `means_grid(data)`, `balanced_grid(data; vars...)`, `quantile_grid(data; vars...)`
  - Or use pure grid construction: `cartesian_grid(vars...)` (automatically completed with typical values)
  - Or provide custom DataFrame with desired covariate combinations
  - **Note**: Missing model variables are automatically completed with typical values internally

# Keyword Arguments
- `type::Symbol=:effects`: Analysis type
  - `:effects` - Marginal Effects at profiles (MEM/MER): derivatives/contrasts at specific points
  - `:predictions` - Adjusted Predictions at profiles (APM/APR): fitted values at specific points
- `vars=nothing`: Variables for effects analysis (Symbol, Vector{Symbol}, or :all_continuous)
  - Only required when `type=:effects`
  - Defaults to all explanatory variables from the model formula (not all data columns)
  - Only variables that appear in both the model specification and data are considered
  - Extra columns in data that aren't in the model are automatically ignored
- `scale::Symbol=:response`: Target scale for computation
  - `:response` - Response scale (default, applies inverse link function)  
  - `:link` - Linear predictor scale (link scale)
- `backend::Symbol=:ad`: Computational backend
  - `:ad` - Automatic differentiation (higher accuracy, small memory cost)
  - `:fd` - Finite differences (zero allocation, production-ready)
- `measure::Symbol=:effect`: Effect measure (only for `type=:effects`)
  - `:effect` - Marginal effects (default, current behavior)
  - `:elasticity` - Elasticities (percent change in y for percent change in x)
  - `:semielasticity_dyex` - Semielasticity d(y)/d(ln x) (change in y for percent change in x)
  - `:semielasticity_eydx` - Semielasticity d(ln y)/dx (percent change in y for unit change in x)
- `contrasts::Symbol=:baseline`: Contrast type for categorical variables
  - `:baseline` - Compare each level to reference level
  - `:pairwise` - All pairwise comparisons between levels
- `ci_alpha::Float64=0.05`: Type I error rate α for confidence intervals (confidence level = 1-α)
  - When specified, `ci_lower` and `ci_upper` columns are added to DataFrame output
- `vcov=GLM.vcov`: Covariance matrix function for standard errors
  - `GLM.vcov` - Model-based covariance matrix (default)
  - Custom function for robust/clustered standard errors

# Returns
`EffectsResult` or `PredictionsResult` containing:
- Results DataFrame with estimates, standard errors, t-statistics, p-values
- Profile columns (at_varname) showing covariate values for each estimate
- Parameter gradients matrix for delta-method standard errors
- Analysis metadata (options used, model info, etc.)

# Statistical Notes
- Standard errors computed via delta method using full model covariance matrix
- Categorical variables use baseline contrasts vs reference levels at each profile
- Profile approach enables interpretation at specific, meaningful covariate combinations
- More efficient than population approach when analyzing specific scenarios

# Examples
```julia
# Effects at sample means (MEM) - most common case
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:x1, :x2])
DataFrame(result)  # Convert to DataFrame with profile information

# Elasticities at sample means
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:x1], measure=:elasticity)

# Effects at specific scenarios (MER) using cartesian grid
result = profile_margins(model, data, cartesian_grid(x1=[0, 1], income=[25000, 50000]); 
                        type=:effects, vars=[:education])

# Semi-elasticities at specific profiles
result = profile_margins(model, data, cartesian_grid(x1=[-1, 0, 1]); 
                        type=:effects, vars=[:x2], measure=:semielasticity_dyex)

# Predictions at the mean (APM)
result = profile_margins(model, data, means_grid(data); type=:predictions)

# Balanced factorial designs
result = profile_margins(model, data, balanced_grid(data; education=:all, region=:all); type=:effects)

# Baseline contrasts for categorical variables (default)
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:education], contrasts=:baseline)

# Pairwise contrasts for categorical variables
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:education], contrasts=:pairwise)

# Multiple explicit profiles for complex analysis
reference_grid = DataFrame(
    x1=[0.0, 1.0], 
    x2=[1.0, 0.0], 
    region=["North", "South"]
)
result = profile_margins(model, data, reference_grid; type=:effects)

# High-performance with pre-built reference grid
reference_grid = DataFrame(x1=[0, 1, 2], x2=[10, 20, 30])
result = profile_margins(model, data, reference_grid; type=:predictions)

# Pure Cartesian product for systematic exploration
grid = cartesian_grid(age=[25, 35, 45], education=[12, 16])
result = profile_margins(model, data, grid; type=:effects, vars=[:income], backend=:ad)

# Data-driven range construction
grid = cartesian_grid(
    x1 = collect(range(extrema(data.x1)...; length = 5)),
    x2 = collect(range(extrema(data.x2)...; length = 5))
)
result = profile_margins(model, data, grid; type=:effects)

# Manual completion still available if needed
partial_grid = cartesian_grid(x1=[0, 1, 2])
complete_grid = complete_reference_grid(partial_grid, model, data)
result = profile_margins(model, data, complete_grid; type=:predictions)
```

# Automatic Reference Grid Completion
Missing model variables are automatically completed with typical values:
```julia
# Your data: region = 75% Urban, 25% Rural, treated = 60% true, 40% false
# Model: y ~ x1 + x2 + region + treated

# Specify only variables you care about
grid = cartesian_grid(x1=[0, 1, 2], x2=[10, 20])
result = profile_margins(model, data, grid; type=:effects)

# Internally completed with typical values:
# → x1, x2: your specified values
# → region: frequency-weighted (75% urban, 25% rural) 
# → treated: 0.6 (actual treatment rate)
# But output shows only x1, x2 in at_* columns!

# Effects "at means" for comparison
result = profile_margins(model, data, means_grid(data); type=:effects)
# → Shows all variables: at_x1, at_x2, at_region, at_treated
```

See also: [`population_margins`](@ref) for population-averaged effects and predictions.
"""

"""
    _extract_profile_values(reference_grid, result_length::Int) -> NamedTuple

Extract profile values from reference grid and expand to match the result length.
Each profile can generate multiple results (one per variable), so we need to repeat
each profile row for each variable it generates.
"""
function _extract_profile_values(reference_grid, result_length::Int)
    n_profiles = nrow(reference_grid)
    vars_per_profile = result_length ÷ n_profiles
    
    # Convert to named tuple with expanded values
    profile_dict = Dict{Symbol, Vector}()
    
    for col_name in names(reference_grid)
        col_data = reference_grid[!, col_name]
        # Repeat each profile value vars_per_profile times
        expanded_data = repeat(col_data, inner=vars_per_profile)
        profile_dict[Symbol(col_name)] = expanded_data
    end
    
    return NamedTuple(profile_dict)
end

"""
    _extract_and_filter_profile_values(reference_grid, result_length::Int, original_vars::Set{Symbol}) -> NamedTuple

Extract profile values from reference grid and filter to only show variables that were
originally specified by the user (hiding automatically added typical values).
Each profile can generate multiple results (one per variable), so we need to repeat
each profile row for each variable it generates.
"""
function _extract_and_filter_profile_values(reference_grid, result_length::Int, original_vars::Set{Symbol})
    n_profiles = nrow(reference_grid)
    vars_per_profile = result_length ÷ n_profiles
    
    # Convert to named tuple with expanded values, filtering to original variables only
    profile_dict = Dict{Symbol, Vector}()
    
    for col_name in names(reference_grid)
        # Only include variables that were in the original reference grid
        if Symbol(col_name) in original_vars
            col_data = reference_grid[!, col_name]
            # Repeat each profile value vars_per_profile times
            expanded_data = repeat(col_data, inner=vars_per_profile)
            profile_dict[Symbol(col_name)] = expanded_data
        end
    end
    
    return NamedTuple(profile_dict)
end

"""
    _profile_margins(model, data_nt, reference_grid, type, vars, scale, backend, measure, vcov, at_spec) -> Union{EffectsResult, PredictionsResult}

Internal implementation for both profile_margins methods.
This eliminates code duplication between the convenience method and DataFrame method.
"""
function _profile_margins(model, data_nt::NamedTuple, reference_grid::DataFrame, type::Symbol, vars, scale::Symbol, backend::Symbol, measure::Symbol, vcov, at_spec, ci_alpha::Float64, contrasts::Symbol)
    # Handle vars parameter with improved validation - use same helper as population_margins
    if type === :effects
        vars = _process_vars_parameter(model, vars, data_nt)
    else # type === :predictions
        vars = nothing  # Not needed for predictions
    end
    
    # Store original reference grid variable names for output filtering
    original_grid_vars = Set(Symbol.(names(reference_grid)))

    # Attempt to remove response variable from display if it appears in the provided grid.
    # Be defensive: formulas with transformed responses (e.g., log(y)) may not expose `.sym`.
    try
        response_var = Symbol(model.mf.f.lhs.sym)
        delete!(original_grid_vars, response_var)
    catch
        # If we cannot determine a simple response symbol, do nothing.
        # Reference grids almost never include the response variable anyway.
    end

    # Complete reference grid with typical values for missing model variables
    # Note: complete_reference_grid now filters to model variables internally
    completed_reference_grid = complete_reference_grid(reference_grid, model, data_nt)

    # Filter original_grid_vars to only model variables for proper output display
    # Get model variables from the completed grid to know what was actually kept
    model_vars_in_grid = Set(Symbol.(names(completed_reference_grid)))
    original_grid_vars = intersect(original_grid_vars, model_vars_in_grid)
    
    # Build zero-allocation engine with ProfileUsage optimization (including vcov function)
    # Determine derivative support based on type and vars
    deriv_support = (type === :effects && !isnothing(vars) && !isempty(vars)) ? HasDerivatives : NoDerivatives
    engine = get_or_build_engine(ProfileUsage, deriv_support, model, data_nt, isnothing(vars) ? Symbol[] : vars, vcov, backend)
    
    if type === :effects
        # Use the actual working function that already exists from the original utilities.jl
        # Pass original_grid_vars so we know which variables are scenario-defining
        df, G = _mem_continuous_and_categorical(engine, completed_reference_grid, original_grid_vars, scale, backend, measure, contrasts)
        
        # Convert symbol terms + profile info to descriptive strings for user display
        df = _convert_profile_terms_to_strings(df)
        
        metadata = _build_metadata(; type, vars, scale, backend, measure, n_obs=length(first(data_nt)), 
                                  model_type=typeof(model), at_spec=at_spec)
        
        # Add analysis_type for format auto-detection
        metadata[:analysis_type] = :profile
        metadata[:n_profiles] = nrow(completed_reference_grid)
        
        # Store confidence interval parameters in metadata
        metadata[:alpha] = ci_alpha
        
        # Extract raw components from DataFrame  
        estimates = df.estimate
        standard_errors = df.se
        variables = string.(df.variable)  # The "x" in dy/dx
        terms = string.(df.contrast)  # Convert Symbol to String
        
        # Extract profile values from reference grid - expand to match result length
        # Filter to show only original reference grid variables (hide automatic typical values)
        profile_values = _extract_and_filter_profile_values(completed_reference_grid, length(estimates), original_grid_vars)
        
        return EffectsResult(estimates, standard_errors, variables, terms, profile_values, nothing, G, metadata)
    else # :predictions  
        # Reference grid can contain CategoricalMixture objects directly - FormulaCompiler handles them
        df, G = _profile_predictions(engine, completed_reference_grid, scale)  # → APM/APR
        metadata = _build_metadata(; type, vars=Symbol[], scale, backend, n_obs=length(first(data_nt)), 
                                  model_type=typeof(model), at_spec=at_spec)
        
        # Add analysis_type for format auto-detection  
        metadata[:analysis_type] = :profile
        metadata[:n_profiles] = nrow(completed_reference_grid)
        
        # Store confidence interval parameters in metadata
        metadata[:alpha] = ci_alpha
        
        # Extract raw components from DataFrame
        estimates = df.estimate
        standard_errors = df.se
        
        # Extract profile values from reference grid - expand to match result length
        # Filter to show only original reference grid variables (hide automatic typical values)
        profile_values = _extract_and_filter_profile_values(completed_reference_grid, length(estimates), original_grid_vars)
        
        return PredictionsResult(estimates, standard_errors, profile_values, nothing, G, metadata)
    end
end

"""
    profile_margins(model, data, reference_grid; type=:effects, vars=nothing, scale=:response, backend=:ad, measure=:effect, contrasts=:baseline, ci_alpha=0.05, vcov=GLM.vcov) -> Union{EffectsResult, PredictionsResult}

Compute profile marginal effects or adjusted predictions at specific covariate combinations.

This function evaluates effects/predictions at user-specified scenarios using reference grids,
implementing the "Profile" approach from the 2×2 framework (Population vs Profile × Effects vs Predictions).

# Arguments
- `model`: Fitted statistical model supporting `coef()` and `vcov()` methods
- `data`: Data table (DataFrame, NamedTuple, or any Tables.jl-compatible format)
- `reference_grid`: DataFrame specifying covariate combinations for analysis
  - Use reference grid builders: `means_grid(data)`, `balanced_grid(data; vars...)`, `quantile_grid(data; vars...)`
  - Or use pure grid construction: `cartesian_grid(vars...)` (automatically completed with typical values)
  - Or provide custom DataFrame with desired covariate combinations
  - **Note**: Missing model variables are automatically completed with typical values internally

# Keyword Arguments
- `type::Symbol=:effects`: Analysis type
  - `:effects` - Marginal Effects at profiles: derivatives/contrasts at specific points
  - `:predictions` - Adjusted Predictions at profiles: fitted values at specific points
- `vars=nothing`: Variables for effects analysis (Symbol, Vector{Symbol}, or :all_continuous)
  - Defaults to all explanatory variables from the model formula (not all data columns)
  - Only variables that appear in both the model specification and data are considered
  - Extra columns in data that aren't in the model are automatically ignored
- `scale::Symbol=:response`: Target scale (:response or :link)
- `backend::Symbol=:ad`: Computational backend (:ad or :fd)
- `measure::Symbol=:effect`: Effect measure (:effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx)
- `contrasts::Symbol=:baseline`: Contrast type for categorical variables (:baseline or :pairwise)
- `ci_alpha::Float64=0.05`: Type I error rate α for confidence intervals (confidence level = 1-α)
- `vcov=GLM.vcov`: Covariance matrix function for standard errors

# Examples
```julia
# Effects at sample means (most common case)
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:x1, :x2])

# Balanced factorial designs
result = profile_margins(model, data, balanced_grid(data; education=:all, region=:all); type=:effects)

# Effects at specific scenarios using cartesian grid
result = profile_margins(model, data, cartesian_grid(x1=[0, 1], income=[25000, 50000]); 
                        type=:effects, vars=[:education])

# Predictions using quantile-based grid
result = profile_margins(model, data, quantile_grid(data; age=[0.25, 0.5, 0.75]); type=:predictions)

# Direct reference grid specification (maximum control)
reference_grid = DataFrame(age=[25, 35, 45], education=[12, 16, 20], region=["North", "South", "North"])
result = profile_margins(model, data, reference_grid; type=:effects, vars=[:income])

# Baseline contrasts for categorical variables (default behavior)
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:education], contrasts=:baseline)

# Pairwise contrasts for categorical variables
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:education], contrasts=:pairwise)

# Complex scenarios with categorical mixtures
using Margins: mix
reference_grid = DataFrame(x1=[0, 1, 2], categorical_var=[mix("A" => 0.7, "B" => 0.3), "A", "B"])
result = profile_margins(model, data, reference_grid; type=:predictions)
```
"""
function profile_margins(
    model, data, reference_grid::DataFrame;
    type::Symbol=:effects, vars=nothing, scale::Symbol=:response,
    backend::Symbol=:ad, measure::Symbol=:effect,
    contrasts::Symbol=:baseline,
    ci_alpha::Float64=0.05, vcov=GLM.vcov,
)
    # Convert data to NamedTuple for consistency
    data_nt_raw = Tables.columntable(data)

    # Convert all numeric columns to Float64 for type stability
    # This prevents heterogeneous NamedTuple types that cause allocations
    data_nt = _convert_numeric_to_float64(data_nt_raw)

    # Process reference grid to convert string categorical specifications to proper categorical values
    # This enables users to write cartesian_grid(education=["High School", "College"]) naturally
    processed_reference_grid = process_reference_grid(data, reference_grid)
    
    # Shared input validation for common parameters  
    validate_margins_common_inputs(model, data_nt, type, vars, scale, backend, measure, vcov)
    
    # Profile-specific validation for reference grid
    if isnothing(reference_grid)
        throw(ArgumentError("reference_grid cannot be nothing"))
    end
    
    # Reference grid specific validation (use processed grid for actual validation)
    if nrow(processed_reference_grid) == 0
        throw(ArgumentError("reference_grid cannot be empty"))
    end
    
    # Validate data types in reference grid - explicit error policy (use processed grid)
    for col_name in names(processed_reference_grid)
        col = processed_reference_grid[!, col_name]
        _validate_reference_grid_column_type(col, col_name)
    end
    
    if ncol(processed_reference_grid) == 0
        throw(ArgumentError("reference_grid must have at least one column"))
    end
    
    # Validate that reference grid variables exist in the data or model
    data_vars = Set(keys(data_nt))
    # Get model variables by compiling the formula and extracting variable references (using cache)
    try
        compiled = get_or_compile_formula(model, data_nt)
        model_vars = Set{Symbol}()
        for op in compiled.ops
            if op isa LoadOp
                Col = typeof(op).parameters[1]
                push!(model_vars, Col)
            elseif op isa ContrastOp
                Col = typeof(op).parameters[1]
                push!(model_vars, Col)
            end
        end
        
        # Check reference grid variables against both data and model variables
        for col_name in names(processed_reference_grid)
            col_symbol = Symbol(col_name)
            if !(col_symbol in data_vars) && !(col_symbol in model_vars)
                throw(ArgumentError("Reference grid variable '$col_name' not found in model data. Available variables: $(sort(collect(data_vars)))"))
            end
        end
    catch e
        # If we can't compile the formula (which might happen if the reference grid is invalid),
        # just check against data variables for basic validation
        for col_name in names(processed_reference_grid)
            col_symbol = Symbol(col_name)
            if !(col_symbol in data_vars)
                throw(ArgumentError("Reference grid variable '$col_name' not found in data. Available variables: $(sort(collect(data_vars)))"))
            end
        end
    end
    
    # Route to single implementation with processed reference grid
    return _profile_margins(model, data_nt, processed_reference_grid, type, vars, scale, backend, measure, vcov, processed_reference_grid, ci_alpha, contrasts)
end


"""
    _profile_predictions(engine, reference_grid, scale) -> (DataFrame, Matrix{Float64})

Compute adjusted predictions at profiles (APM/APR) with delta-method standard errors.

This function evaluates predictions at each row of the reference grid, providing
adjusted predictions at the mean (APM) or at representative values (APR).
"""
function _profile_predictions(engine::MarginsEngine{L}, reference_grid, scale) where L
    n_profiles = nrow(reference_grid)
    n_params = length(engine.β)
    
    # Reuse η_buf for predictions if possible (performance optimization)
    # NOTE: Buffer fallback is intentional and harmless - engines are often created with
    # minimal buffers (size=1) for flexibility, and gracefully allocate when needed.
    # This is working as designed, not an error.
    if length(engine.η_buf) >= n_profiles
        predictions = view(engine.η_buf, 1:n_profiles)
    else
        @info "Buffer allocation fallback: η_buf too small for $n_profiles profiles (size=$(length(engine.η_buf)))"
        predictions = Vector{Float64}(undef, n_profiles)
    end
    G = zeros(n_profiles, n_params)  # One row per profile
    
    # Compile with reference grid (like effects rework)
    # Convert reference grid to NamedTuple format for FormulaCompiler
    data_nt_raw = Tables.columntable(reference_grid)

    # Convert all numeric columns to Float64 for type stability
    data_nt = _convert_numeric_to_float64(data_nt_raw)

    # Single compilation with complete reference grid (using cache for performance)
    refgrid_compiled = get_or_compile_formula(engine.model, data_nt)

    # Single pass over profiles via helper that takes only concrete arguments
    _compute_profile_predictions!(predictions, G, refgrid_compiled, engine, data_nt, scale)
    
    # Safely use g_buf for SE computation if large enough (performance optimization)
    # NOTE: Buffer fallback is intentional and harmless - engines are often created with
    # minimal buffers (size=1) for flexibility, and gracefully allocate when needed.
    # This is working as designed, not an error.
    if length(engine.g_buf) >= n_profiles
        se_vals = view(engine.g_buf, 1:n_profiles)  # Reuse g_buf if large enough
    else
        @info "Buffer allocation fallback: g_buf too small for $n_profiles profiles (size=$(length(engine.g_buf)))"
        se_vals = Vector{Float64}(undef, n_profiles)  # Fall back to allocation
    end
    for i in 1:n_profiles
        se_vals[i] = sqrt((G[i:i, :] * engine.Σ * G[i:i, :]')[1, 1])
    end
    
    # Create results DataFrame with profile information (no variable/contrast for predictions)
    results = DataFrame()
    results.estimate = predictions
    results.se = se_vals
    results.t_stat = predictions ./ se_vals
    results.p_value = 2 .* (1 .- cdf.(Normal(), abs.(predictions ./ se_vals)))
    
    # Add profile columns to show which profile each prediction corresponds to
    for (col_name, col_data) in pairs(Tables.columns(reference_grid))
        results[!, Symbol("at_$(col_name)")] = col_data
    end
    
    return results, G
end

function _compute_profile_predictions!(
    predictions::AbstractVector{<:Float64},
    G::AbstractMatrix{<:Float64},
    compiled,
    engine::MarginsEngine,
    data_nt::NamedTuple,
    scale::Symbol
)
    # Extract engine fields once for cleaner code
    (; row_buf, β, link) = engine
    
    # Use centralized batch prediction computation (zero allocation)
    compute_predictions_batch!(predictions, G, compiled, data_nt, β, link, scale, row_buf)
    return nothing
end


"""
    _convert_profile_terms_to_strings(df::DataFrame)

Convert symbol terms + profile descriptions to user-friendly descriptive strings.
Internal computation uses symbols, but user display uses descriptive strings.

`df`` is only arg, since we assume it is filtered to model variables
"""
function _convert_profile_terms_to_strings(df::DataFrame)
    # Clean up contrast descriptions for better display
    descriptive_terms = String[]
    
    for i in 1:nrow(df)
        var = df.variable[i]
        contrast = df.contrast[i]
        
        # Clean up boolean contrasts: convert "value vs false" patterns to "true vs false"
        if contains(contrast, " vs false") && contrast != "true vs false"
            # This is a boolean variable with a numeric mean - standardize to "true vs false"
            cleaned_contrast = "true vs false"
        elseif contrast == "derivative"
            # Keep derivative as-is for continuous variables
            cleaned_contrast = contrast
        else
            # Keep other contrasts as-is (for future categorical support)
            cleaned_contrast = contrast
        end
        
        push!(descriptive_terms, cleaned_contrast)
    end
    
    # Create new DataFrame with clean contrast terms
    result_df = DataFrame(
        variable = df.variable,  # Include the "x" in dy/dx 
        contrast = descriptive_terms,
        estimate = df.estimate,
        se = df.se
    )
    
    return result_df
end

"""
    _mem_continuous_and_categorical(engine, reference_grid, scale, backend, measure, contrasts) -> (DataFrame, Matrix)

Profile Effects (MEM) properly decomposed from the original working implementation.
This is the correct decomposition that should have been completed during BREAKUP.md.
"""
function _mem_continuous_and_categorical(engine::MarginsEngine{L}, reference_grid, original_grid_vars::Set{Symbol}, scale::Symbol, backend::Symbol, measure::Symbol, contrasts::Symbol=:baseline) where L
    # Use the decomposed modular approach - break down the original 200-line function

    # Step 1: Calculate dimensions (extracted from original lines 1262-1286)
    n_profiles = nrow(reference_grid)
    continuous_vars = continuous_variables(engine.compiled, engine.data_nt)
    requested_vars = engine.vars
    continuous_requested = [v for v in requested_vars if v ∈ continuous_vars]
    categorical_requested_raw = [v for v in requested_vars if v ∉ continuous_vars]

    # Filter out variables that were ORIGINALLY in the reference grid (before completion)
    # Variables in original reference grid define the scenarios (conditioning variables), not analysis targets
    # Variables added by completion with typical values/mixtures ARE analysis targets
    refgrid_vars = original_grid_vars

    # Filter out mixture variables AND reference grid variables from categorical contrasts
    # Mixtures in reference grid are context/backdrop, not variables to contrast
    # Non-mixture reference grid variables are scenario-defining, not analysis targets
    refgrid_col_table = Tables.columntable(reference_grid)
    categorical_requested = Symbol[]
    for var in categorical_requested_raw
        refgrid_col = getproperty(refgrid_col_table, var)

        # Skip if it's a mixture
        if refgrid_col isa AbstractVector{<:CategoricalMixture}
            @info "Skipping contrasts for variable $var: specified as mixture in reference grid (use discrete levels if contrasts desired)"
            continue
        end

        # Skip if Bool variable has non-Bool values (indicates mixture specification)
        # Check original data type to detect Bool variables with Float64 mixture values
        orig_col = getproperty(engine.data_nt, var)
        if eltype(orig_col) <: Bool && !(eltype(refgrid_col) <: Bool)
            @info "Skipping contrasts for variable $var: specified as mixture in reference grid (use discrete levels if contrasts desired)"
            continue
        end

        # Skip if it's explicitly in the reference grid (scenario-defining variable)
        if var ∈ refgrid_vars
            @debug "Skipping contrasts for variable $var: specified in reference grid (scenario-defining variable)"
            continue
        end

        push!(categorical_requested, var)
    end

    # NOTE: DO NOT filter continuous variables from reference grid
    # Continuous variables in the grid just specify WHERE to evaluate the derivative
    # We still want to compute marginal effects for them at those points
    # Only categorical variables are filtered (to avoid contrasts for scenario-defining categories)

    # Calculate total terms based on ACTUAL variables to be processed (continuous + filtered categorical)
    actual_vars = vcat(continuous_requested, categorical_requested)
    total_terms = _calculate_total_profile_terms(actual_vars, categorical_requested, contrasts, engine, reference_grid, n_profiles)

    # Step 2: Pre-allocate using decomposed function (extracted from original lines 1288-1296)
    results, G = _build_profile_results_structure(total_terms, length(engine.β))

    # Step 3: Prepare reference grid compilation (extracted from original lines 1297-1303)
    refgrid_data = Tables.columntable(reference_grid)
    refgrid_compiled = get_or_compile_formula(engine.model, refgrid_data)
    refgrid_de = _build_refgrid_derivative_evaluator(refgrid_compiled, refgrid_data, continuous_requested, engine, backend)

    # Step 3.5: Build ContrastEvaluator for refgrid (Phase 4 migration)
    # Only build if there are categorical variables to process (after filtering mixtures)
    refgrid_contrast = if !isempty(categorical_requested)
        # IMPORTANT: For baseline contrasts, we need ContrastEvaluator to know about ALL
        # categorical levels from original data, not just levels in the reference grid.
        # Create hybrid data: categorical vars from original data, continuous from refgrid
        hybrid_data_dict = Dict{Symbol, Any}()
        for (k, v) in pairs(refgrid_data)
            if k in categorical_requested
                # Use original data's categorical structure (has all levels)
                hybrid_data_dict[k] = getproperty(engine.data_nt, k)
            else
                # Use refgrid values for continuous/other variables
                hybrid_data_dict[k] = v
            end
        end
        hybrid_data = NamedTuple(hybrid_data_dict)

        contrastevaluator(refgrid_compiled, hybrid_data, categorical_requested)
    else
        nothing  # No categorical contrasts needed
    end

    # Step 4: Pre-allocate profile computation buffers (PHASE 3 FIX)
    # These buffers are reused for ALL profiles to achieve 0 allocations
    n_params = length(engine.β)

    # Continuous variable buffers
    # IMPORTANT: Gβ second dimension MUST match length(refgrid_de.vars) for marginal_effects_eta!/mu!
    n_refgrid_vars = !isnothing(refgrid_de) ? length(refgrid_de.vars) : 0
    Gβ_continuous = n_refgrid_vars > 0 ? Matrix{Float64}(undef, n_params, n_refgrid_vars) : Matrix{Float64}(undef, 0, 0)
    output_buf = Vector{Float64}(undef, length(refgrid_compiled))  # For measure transformations

    # Categorical variable buffers
    contrast_buf = Vector{Float64}(undef, length(refgrid_compiled))
    η_baseline_buf = Vector{Float64}(undef, length(refgrid_compiled))
    gradient_buf = Vector{Float64}(undef, n_params)

    # Step 5: Process each profile (extracted from original lines 1305-1446)
    # Process only variables that should be computed (continuous + non-mixture categorical)
    row_idx = 1
    for profile_idx in 1:n_profiles
        # Process continuous variables
        for var in continuous_requested
            row_idx = _process_profile_continuous_variable!(results, G, row_idx, engine, var,
                                                          refgrid_de, refgrid_data, refgrid_compiled,
                                                          reference_grid, profile_idx,
                                                          scale, backend, measure, continuous_vars,
                                                          Gβ_continuous, output_buf)
        end

        # Process categorical variables (only non-mixtures)
        for var in categorical_requested
            row_idx = _process_profile_categorical_variable!(results, G, row_idx, engine, var,
                                                           refgrid_contrast, refgrid_compiled, refgrid_data,
                                                           reference_grid, profile_idx,
                                                           scale, backend, contrasts,
                                                           contrast_buf, η_baseline_buf, gradient_buf)
        end
    end

    # Check if we filled all rows correctly
    actual_rows_filled = row_idx - 1
    if actual_rows_filled != total_terms
        n_cont = length(continuous_requested)
        n_cat = length(categorical_requested)
        @warn "Profile results size mismatch" total_terms actual_rows_filled n_profiles n_cont n_cat contrasts continuous_requested categorical_requested
        # Trim to actual size to avoid UndefRefError
        results = results[1:actual_rows_filled, :]
        G = G[1:actual_rows_filled, :]
    end

    return (results, G)
end

# ==============================================================================
# DECOMPOSED HELPER FUNCTIONS (extracted from original _mem_continuous_and_categorical)
# ==============================================================================

"""
    _calculate_total_profile_terms(requested_vars, categorical_requested, contrasts, engine, reference_grid, n_profiles) -> Int

Calculate total number of terms needed for profile margins results.
The requested_vars should already be filtered (continuous + non-mixture categorical).
For categorical pairwise contrasts, uses levels in REFERENCE GRID, not original data.
Extracted from original lines 1270-1286.
"""
function _calculate_total_profile_terms(requested_vars, categorical_requested, contrasts, engine, reference_grid, n_profiles)
    total_terms = 0
    refgrid_data = Tables.columntable(reference_grid)

    for var in requested_vars
        if var ∈ categorical_requested && contrasts === :pairwise
            # Pairwise contrasts: n_choose_2 comparisons per profile
            # IMPORTANT: Use levels in reference grid, not original data
            # Variables added by completion may have only 1 level (all same value)
            refgrid_col = getproperty(refgrid_data, var)
            n_levels = length(unique(refgrid_col))

            # Only count pairs if there are actually multiple levels
            if n_levels >= 2
                n_pairs = (n_levels * (n_levels - 1)) ÷ 2
                total_terms += n_profiles * n_pairs
            end
            # else: skip this variable (0 pairs to compute with only 1 level)
        else
            # Continuous variable OR categorical baseline contrasts: 1 term per profile
            total_terms += n_profiles
        end
    end
    return total_terms
end

"""
    _build_profile_results_structure(total_terms, n_params) -> (DataFrame, Matrix)

Build results DataFrame and gradient matrix for profile margins.
Pre-allocates all vectors to avoid push! allocations.
Extracted from original lines 1288-1296.
"""
function _build_profile_results_structure(total_terms, n_params)
    # Pre-allocate all vectors with known size (avoids push! allocations)
    results = DataFrame(
        variable = Vector{String}(undef, total_terms),
        contrast = Vector{String}(undef, total_terms),
        estimate = Vector{Float64}(undef, total_terms),
        se = Vector{Float64}(undef, total_terms),
        profile_desc = Vector{Any}(undef, total_terms)  # Use Any to handle NamedTuple
    )
    G = Matrix{Float64}(undef, total_terms, n_params)
    return (results, G)
end

"""
    _build_refgrid_derivative_evaluator(refgrid_compiled, refgrid_data, continuous_requested, engine)

Build derivative evaluator for reference grid if needed.
Extracted from original lines 1300-1303.
"""
function _build_refgrid_derivative_evaluator(refgrid_compiled, refgrid_data, continuous_requested, engine, backend::Symbol)
    if !isempty(continuous_requested) && !isnothing(engine.de)
        continuous_vars = continuous_variables(engine.compiled, engine.data_nt)
        # Use FormulaCompiler's derivativeevaluator (dispatches to correct backend)
        return derivativeevaluator(backend, refgrid_compiled, refgrid_data, continuous_vars)
    else
        return nothing
    end
end

"""
    _process_profile_continuous_variable!(results, G, row_idx, engine, var, refgrid_de, refgrid_data,
                                        refgrid_compiled, reference_grid, profile_idx, scale, backend,
                                        measure, continuous_vars) -> next_row_idx

Process a single continuous variable at a single profile point.
Extracted from original lines 1311-1375.
"""
function _process_profile_continuous_variable!(results, G, row_idx, engine, var, refgrid_de, refgrid_data,
                                             refgrid_compiled, reference_grid, profile_idx, scale, backend,
                                             measure, continuous_vars, Gβ::Matrix{Float64}, output_buf::Vector{Float64})
    # Get marginal effects at profile point (extracted from lines 1312-1323)
    g_buf_view = @view engine.g_buf[1:length(refgrid_de.vars)]
    local_β = engine.β
    local_link = engine.link

    # Use passed Gβ buffer (0 bytes allocation - PHASE 3 FIX)

    if scale === :response
        marginal_effects_mu!(g_buf_view, Gβ, refgrid_de, local_β, local_link, profile_idx)
    else
        marginal_effects_eta!(g_buf_view, Gβ, refgrid_de, local_β, profile_idx)
    end

    var_idx = findfirst(==(var), continuous_vars)
    if !isnothing(var_idx)
        marginal_effect = g_buf_view[var_idx]

        # Apply measure transformations (extracted from lines 1325-1334)
        local estimate, transform_factor, ∂μ_∂β
        if measure === :effect
            estimate = marginal_effect
            transform_factor = 1.0
            ∂μ_∂β = nothing
        else
            var_value = refgrid_data[var][profile_idx]
            # Compute predicted value at profile point
            # Use passed output buffer (0 bytes allocation - PHASE 3 FIX)
            refgrid_compiled(output_buf, refgrid_data, profile_idx)
            η = dot(output_buf, local_β)  # Linear predictor = X'β
            # Apply link inverse if on response scale
            pred_value = scale === :response ? GLM.linkinv(local_link, η) : η
            (estimate, transform_factor) = apply_measure_transformation(marginal_effect, var_value, pred_value, measure)

            # Compute ∂μ/∂β at this profile point for quotient rule correction
            # For measures that divide by μ, we need this gradient for the quotient rule
            if measure === :elasticity || measure === :semielasticity_eydx
                ∂μ_∂β = if scale === :response
                    # For response scale: μ = g⁻¹(η), so ∂μ/∂β = g⁻¹'(η) × X
                    dμ_dη = GLM.mueta(local_link, η)
                    dμ_dη .* output_buf  # Element-wise: creates vector
                else
                    # For linear scale: η = X'β, so ∂η/∂β = X
                    copy(output_buf)  # X is already in output_buf
                end
            else
                ∂μ_∂β = nothing
            end
        end

        # Extract parameter gradients for this variable from Gβ matrix
        # Gβ is already computed above by marginal_effects_eta!/marginal_effects_mu!
        # Gβ has shape (n_params, n_vars), so column j contains gradient for variable j
        gradient_view = view(Gβ, :, var_idx)

        # Scale gradient by transformation factor AND apply quotient rule correction
        if transform_factor != 1.0
            # First term: k × ∂(∂μ/∂x)/∂β (existing transformation)
            engine.gβ_accumulator .= gradient_view .* transform_factor

            # Second term: quotient rule correction for measures that divide by μ
            # For ε = (x/μ) × ∂μ/∂x, the full quotient rule is:
            # ∂ε/∂β = (x/μ) × ∂(∂μ/∂x)/∂β - (ε/μ) × ∂μ/∂β
            if !isnothing(∂μ_∂β)
                quotient_correction = estimate / pred_value
                @inbounds for j in eachindex(engine.gβ_accumulator)
                    engine.gβ_accumulator[j] -= quotient_correction * ∂μ_∂β[j]
                end
            end
        else
            engine.gβ_accumulator .= gradient_view
        end

        # Store result (extracted from lines 1355-1374)
        se = compute_se_only(engine.gβ_accumulator, engine.Σ)

        profile_dict = Dict{Symbol,Any}()
        for k in names(reference_grid)
            val = reference_grid[profile_idx, k]
            if val isa CategoricalMixture
                profile_dict[Symbol(k)] = string(val)
            else
                profile_dict[Symbol(k)] = val
            end
        end
        profile_nt = NamedTuple(profile_dict)

        # Use column indexing for better performance
        results.variable[row_idx] = string(var)
        results.contrast[row_idx] = "derivative"
        results.estimate[row_idx] = estimate
        results.se[row_idx] = se
        results.profile_desc[row_idx] = profile_nt
        G[row_idx, :] = engine.gβ_accumulator

        return row_idx + 1
    end

    return row_idx
end

"""
    _process_profile_categorical_variable!(results, G, row_idx, engine, var, refgrid_contrast,
                                         refgrid_compiled, refgrid_data, reference_grid, profile_idx,
                                         scale, backend, contrasts) -> next_row_idx

Process a single categorical variable at a single profile point.
Phase 4 Migration: Uses ContrastEvaluator kernel (same as population path).
Extracted from original lines 1377-1444.
"""
function _process_profile_categorical_variable!(results, G, row_idx, engine, var, refgrid_contrast,
                                              refgrid_compiled, refgrid_data, reference_grid, profile_idx,
                                              scale, backend, contrasts,
                                              contrast_buf::Vector{Float64}, η_baseline_buf::Vector{Float64}, gradient_buf::Vector{Float64})
    profile_dict = Dict(Symbol(k) => reference_grid[profile_idx, k] for k in names(reference_grid))

    if contrasts === :baseline
        # Baseline contrast using ContrastEvaluator kernel
        current_level = profile_dict[var]
        baseline_level_str = _get_baseline_level(engine.model, var, engine.data_nt)

        # Convert to same type as current_level for ContrastEvaluator
        # Both levels must be from the categorical pool, not specific row values
        baseline_level = if current_level isa CategoricalValue
            # Get baseline CategoricalValue from the original data categorical pool
            orig_col = getproperty(engine.data_nt, var)
            idx_orig = findfirst(x -> string(x) == baseline_level_str, orig_col)
            if idx_orig !== nothing
                orig_col[idx_orig]
            else
                # Fallback: create from refgrid if baseline not in original data
                refgrid_col = getproperty(refgrid_data, var)
                idx = findfirst(x -> string(x) == baseline_level_str, refgrid_col)
                idx !== nothing ? refgrid_col[idx] : baseline_level_str
            end
        else
            baseline_level_str
        end

        # Use refgrid ContrastEvaluator with profile_idx as the row
        # Pass pre-allocated buffers (0 bytes allocation - PHASE 3 FIX)
        marginal_effect = _compute_row_specific_baseline_contrast(
            refgrid_contrast, refgrid_compiled, refgrid_data, profile_idx,
            var, baseline_level, current_level, engine.β, engine.link, scale,
            contrast_buf, η_baseline_buf
        )

        _row_specific_contrast_grad_beta!(
            engine.gβ_accumulator, refgrid_contrast, profile_idx,
            var, baseline_level, current_level, engine.β, engine.link, scale,
            gradient_buf
        )

        se = compute_se_only(engine.gβ_accumulator, engine.Σ)

        term_name = "$(current_level) - $(baseline_level)"

        profile_display_dict = Dict{Symbol,Any}()
        for k in names(reference_grid)
            val = reference_grid[profile_idx, k]
            if val isa CategoricalMixture
                profile_display_dict[Symbol(k)] = string(val)
            else
                profile_display_dict[Symbol(k)] = val
            end
        end
        profile_nt = NamedTuple(profile_display_dict)

        # Use column indexing for better performance
        results.variable[row_idx] = string(var)
        results.contrast[row_idx] = term_name
        results.estimate[row_idx] = marginal_effect
        results.se[row_idx] = se
        results.profile_desc[row_idx] = profile_nt
        G[row_idx, :] = engine.gβ_accumulator

        return row_idx + 1

    elseif contrasts === :pairwise
        # Pairwise contrasts using ContrastEvaluator kernel
        # Pass pre-allocated buffers (0 bytes allocation - PHASE 3 FIX)
        contrast_results = _compute_profile_pairwise_contrasts(
            refgrid_contrast, refgrid_compiled, refgrid_data, profile_idx,
            var, engine.β, engine.link, scale,
            contrast_buf, gradient_buf, η_baseline_buf
        )

        for (level1, level2, effect, gradient) in contrast_results
            se = sqrt(dot(gradient, engine.Σ, gradient))
            term_name = "$level2 - $level1"

            profile_display_dict = Dict{Symbol,Any}()
            for k in names(reference_grid)
                val = reference_grid[profile_idx, k]
                if val isa CategoricalMixture
                    profile_display_dict[Symbol(k)] = string(val)
                else
                    profile_display_dict[Symbol(k)] = val
                end
            end
            profile_nt = NamedTuple(profile_display_dict)

            # Use column indexing for better performance
            results.variable[row_idx] = string(var)
            results.contrast[row_idx] = term_name
            results.estimate[row_idx] = effect
            results.se[row_idx] = se
            results.profile_desc[row_idx] = profile_nt
            G[row_idx, :] = gradient

            row_idx += 1
        end

        return row_idx
    else
        throw(ArgumentError("Unsupported contrasts type: $contrasts. Use :baseline or :pairwise"))
    end
end
