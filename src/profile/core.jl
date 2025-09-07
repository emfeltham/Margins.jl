# profile/core.jl
# Main profile_margins() function with reference grid approach

using Distributions: Normal, cdf

"""
    profile_margins(model, data, reference_grid; type=:effects, vars=nothing, scale=:response, backend=:ad, measure=:effect, contrasts=:baseline, ci_alpha=0.05, vcov=GLM.vcov) -> MarginsResult

Compute profile marginal effects or adjusted predictions at specific covariate combinations.

This function evaluates effects/predictions at representative points or user-specified scenarios,
implementing the "Profile" approach from the 2×2 framework (Population vs Profile × Effects vs Predictions).
It provides marginal effects at the mean (MEM), marginal effects at representative values (MER),
or adjusted predictions at specific profiles (APM/APR).

# Arguments
- `model`: Fitted statistical model supporting `coef()` and `vcov()` methods
- `data`: Data table (DataFrame, NamedTuple, or any Tables.jl-compatible format)
- `reference_grid`: DataFrame specifying covariate combinations for analysis
  - Use reference grid builders: `means_grid(data)`, `balanced_grid(data; vars...)`, `cartesian_grid(data; vars...)`, `quantile_grid(data; vars...)`
  - Or provide custom DataFrame with desired covariate combinations

# Keyword Arguments
- `type::Symbol=:effects`: Analysis type
  - `:effects` - Marginal Effects at profiles (MEM/MER): derivatives/contrasts at specific points
  - `:predictions` - Adjusted Predictions at profiles (APM/APR): fitted values at specific points
- `vars=nothing`: Variables for effects analysis (Symbol, Vector{Symbol}, or :all_continuous)
  - Only required when `type=:effects`
  - Defaults to all continuous variables (numeric types except Bool)
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
`MarginsResult` containing:
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
result = profile_margins(model, data, cartesian_grid(data; x1=[0, 1], income=[25000, 50000]); 
                        type=:effects, vars=[:education])

# Semi-elasticities at specific profiles
result = profile_margins(model, data, cartesian_grid(data; x1=[-1, 0, 1]); 
                        type=:effects, vars=[:x2], measure=:semielasticity_dyex)

# Predictions at the mean (APM)
result = profile_margins(model, data, means_grid(data); type=:predictions)

# Balanced factorial designs (AsBalanced)
result = profile_margins(model, data, balanced_grid(data; education=:all, region=:all); type=:effects)

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

# Cartesian product for systematic exploration
result = profile_margins(model, data, 
                        cartesian_grid(data; age=[25, 35, 45], education=[12, 16]); 
                        type=:effects, vars=[:income], backend=:ad)
```

# Frequency-Weighted Categorical Defaults
Unspecified categorical variables use actual data composition:
```julia
# Your data: region = 75% Urban, 25% Rural
#           treated = 60% true, 40% false

# Effects "at means" now uses realistic population profile
result = profile_margins(model, data, means_grid(data); type=:effects)
# → income: sample mean
# → region: frequency-weighted (75% urban, 25% rural) 
# → treated: 0.6 (actual treatment rate)
# → Not arbitrary first levels!

# Override when needed for scenario analysis
reference_grid = DataFrame(treated=[1.0])  # 100% treatment scenario
result = profile_margins(model, data, reference_grid; type=:effects)
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
    _profile_margins(model, data_nt, reference_grid, type, vars, scale, backend, measure, vcov, at_spec) -> MarginsResult

Internal implementation for both profile_margins methods.
This eliminates code duplication between the convenience method and DataFrame method.
"""
function _profile_margins(model, data_nt::NamedTuple, reference_grid::DataFrame, type::Symbol, vars, scale::Symbol, backend::Symbol, measure::Symbol, vcov, at_spec, ci_alpha::Float64)
    # Handle vars parameter with improved validation - use same helper as population_margins
    if type === :effects
        vars = _process_vars_parameter(model, vars, data_nt)
    else # type === :predictions
        vars = nothing  # Not needed for predictions
    end
    
    # Build zero-allocation engine with ProfileUsage optimization (including vcov function)
    engine = get_or_build_engine(ProfileUsage, model, data_nt, vars === nothing ? Symbol[] : vars, vcov)
    
    if type === :effects
        # Use reference grid directly for efficient single-compilation approach
        # CategoricalMixture objects are handled natively by FormulaCompiler
        df, G = _mem_continuous_and_categorical_refgrid(engine, reference_grid; scale, backend, measure)  # → MEM/MER
        
        # Convert symbol terms + profile info to descriptive strings for user display
        df = _convert_profile_terms_to_strings(df)
        
        metadata = _build_metadata(; type, vars, scale, backend, measure, n_obs=length(first(data_nt)), 
                                  model_type=typeof(model), at_spec=at_spec)
        
        # Add analysis_type for format auto-detection
        metadata[:analysis_type] = :profile
        metadata[:n_profiles] = nrow(reference_grid)
        
        # Store confidence interval parameters in metadata
        metadata[:alpha] = ci_alpha
        
        # Extract raw components from DataFrame  
        estimates = df.estimate
        standard_errors = df.se
        terms = string.(df.term)  # Convert Symbol to String
        
        # Extract profile values from reference grid - expand to match result length
        profile_values = _extract_profile_values(reference_grid, length(estimates))
        
        return MarginsResult(estimates, standard_errors, terms, profile_values, nothing, G, metadata)
    else # :predictions  
        # Reference grid can contain CategoricalMixture objects directly - FormulaCompiler handles them
        df, G = _profile_predictions(engine, reference_grid; scale)  # → APM/APR
        metadata = _build_metadata(; type, vars=Symbol[], scale, backend, n_obs=length(first(data_nt)), 
                                  model_type=typeof(model), at_spec=at_spec)
        
        # Add analysis_type for format auto-detection  
        metadata[:analysis_type] = :profile
        metadata[:n_profiles] = nrow(reference_grid)
        
        # Store confidence interval parameters in metadata
        metadata[:alpha] = ci_alpha
        
        # Extract raw components from DataFrame
        estimates = df.estimate
        standard_errors = df.se
        terms = string.(df.term)  # Convert Symbol to String
        
        # Extract profile values from reference grid - expand to match result length
        profile_values = _extract_profile_values(reference_grid, length(estimates))
        
        return MarginsResult(estimates, standard_errors, terms, profile_values, nothing, G, metadata)
    end
end

"""
    profile_margins(model, data, reference_grid; type=:effects, vars=nothing, scale=:response, backend=:ad, measure=:effect, contrasts=:baseline, ci_alpha=0.05, vcov=GLM.vcov) -> MarginsResult

Compute profile marginal effects or adjusted predictions at specific covariate combinations.

This function evaluates effects/predictions at user-specified scenarios using reference grids,
implementing the "Profile" approach from the 2×2 framework (Population vs Profile × Effects vs Predictions).

# Arguments
- `model`: Fitted statistical model supporting `coef()` and `vcov()` methods
- `data`: Data table (DataFrame, NamedTuple, or any Tables.jl-compatible format)
- `reference_grid`: DataFrame specifying covariate combinations for analysis
  - Use reference grid builders: `means_grid(data)`, `balanced_grid(data; vars...)`, `cartesian_grid(data; vars...)`, `quantile_grid(data; vars...)`
  - Or provide custom DataFrame with desired covariate combinations

# Keyword Arguments
- `type::Symbol=:effects`: Analysis type
  - `:effects` - Marginal Effects at profiles: derivatives/contrasts at specific points
  - `:predictions` - Adjusted Predictions at profiles: fitted values at specific points
- `vars=nothing`: Variables for effects analysis (Symbol, Vector{Symbol}, or :all_continuous)
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

# Balanced factorial designs (AsBalanced)
result = profile_margins(model, data, balanced_grid(data; education=:all, region=:all); type=:effects)

# Effects at specific scenarios using cartesian grid
result = profile_margins(model, data, cartesian_grid(data; x1=[0, 1], income=[25000, 50000]); 
                        type=:effects, vars=[:education])

# Predictions using quantile-based grid
result = profile_margins(model, data, quantile_grid(data; age=[0.25, 0.5, 0.75]); type=:predictions)

# Direct reference grid specification (maximum control)
reference_grid = DataFrame(age=[25, 35, 45], education=[12, 16, 20], region=["North", "South", "North"])
result = profile_margins(model, data, reference_grid; type=:effects, vars=[:income])

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
    ci_alpha::Float64=0.05, vcov=GLM.vcov,
)
    # Convert data to NamedTuple for consistency
    data_nt = Tables.columntable(data)
    
    # Shared input validation for common parameters  
    validate_margins_common_inputs(model, data_nt, type, vars, scale, backend, measure, vcov)
    
    # Profile-specific validation for reference grid
    if isnothing(reference_grid)
        throw(ArgumentError("reference_grid cannot be nothing"))
    end
    
    # Reference grid specific validation
    if nrow(reference_grid) == 0
        throw(ArgumentError("reference_grid cannot be empty"))
    end
    
    # Validate data types in reference grid - explicit error policy
    for col_name in names(reference_grid)
        col = reference_grid[!, col_name]
        _validate_reference_grid_column_type(col, col_name)
    end
    
    if ncol(reference_grid) == 0
        throw(ArgumentError("reference_grid must have at least one column"))
    end
    
    # Route to single implementation with reference grid directly
    return _profile_margins(model, data_nt, reference_grid, type, vars, scale, backend, measure, vcov, reference_grid, ci_alpha)
end


"""
    _profile_predictions(engine, reference_grid; scale) -> (DataFrame, Matrix{Float64})

Compute adjusted predictions at profiles (APM/APR) with delta-method standard errors.

This function evaluates predictions at each row of the reference grid, providing
adjusted predictions at the mean (APM) or at representative values (APR).
"""
function _profile_predictions(engine::MarginsEngine{L}, reference_grid; scale=:response) where L
    n_profiles = nrow(reference_grid)
    n_params = length(engine.β)
    
    # Reuse η_buf for predictions if possible
    if length(engine.η_buf) >= n_profiles
        predictions = view(engine.η_buf, 1:n_profiles)
    else
        @info "Buffer allocation fallback: η_buf too small for $n_profiles profiles (size=$(length(engine.η_buf)))"
        predictions = Vector{Float64}(undef, n_profiles)
    end
    G = zeros(n_profiles, n_params)  # One row per profile
    
    # Compile with reference grid (like effects rework)
    # Convert reference grid to NamedTuple format for FormulaCompiler
    data_nt = Tables.columntable(reference_grid)
    
    # Single compilation with complete reference grid
    refgrid_compiled = FormulaCompiler.compile_formula(engine.model, data_nt)
    
    # Single pass over profiles via helper that takes only concrete arguments
    _compute_profile_predictions!(predictions, G, refgrid_compiled, engine, data_nt, scale)
    
    # Safely use g_buf for SE computation if large enough  
    if length(engine.g_buf) >= n_profiles
        se_vals = view(engine.g_buf, 1:n_profiles)  # Reuse g_buf if large enough
    else
        @info "Buffer allocation fallback: g_buf too small for $n_profiles profiles (size=$(length(engine.g_buf)))"
        se_vals = Vector{Float64}(undef, n_profiles)  # Fall back to allocation
    end
    for i in 1:n_profiles
        se_vals[i] = sqrt((G[i:i, :] * engine.Σ * G[i:i, :]')[1, 1])
    end
    
    # Create results DataFrame with profile information
    results = DataFrame()
    results.term = ["APM/APR" for _ in 1:n_profiles]
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
    # Create descriptive term strings
    descriptive_terms = String[]
    
    for i in 1:nrow(df)
        var = df.term[i]
        profile = df.profile_desc[i]
        
        # Build profile description
        profile_parts = ["$(k)=$(v)" for (k, v) in pairs(profile)]
        profile_desc = join(profile_parts, ", ")
        
        # For now, use simple format - we can enhance categorical detection later if needed
        term_name = "$(var) at $(profile_desc)"
        push!(descriptive_terms, term_name)
    end
    
    # Create new DataFrame with string terms and remove profile_desc column
    result_df = DataFrame(
        term = descriptive_terms,
        estimate = df.estimate,
        se = df.se
    )
    
    return result_df
end
