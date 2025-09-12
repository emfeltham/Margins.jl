# population/core.jl - Main population_margins() function with compilation caching

# Global cache for compiled formulas (MARGINS_GUIDE.md pattern)
# Unified caching system (see engine/caching.jl)

"""
    population_margins(model, data; type=:effects, vars=nothing, scale=:response, backend=:ad, scenarios=nothing, groups=nothing, measure=:effect, contrasts=:baseline, ci_alpha=0.05, vcov=GLM.vcov, weights=nothing) -> Union{EffectsResult, PredictionsResult}

Compute population-level marginal effects or adjusted predictions.

This function averages effects/predictions across the observed sample distribution,
providing true population parameters for your sample. It implements the "Population" 
approach from the 2×2 framework (Population vs Profile × Effects vs Predictions).

# Arguments
- `model`: Fitted statistical model supporting `coef()` and `vcov()` methods
- `data`: Data table (DataFrame, NamedTuple, or any Tables.jl-compatible format)

# Keyword Arguments
- `type::Symbol=:effects`: Analysis type
  - `:effects` - Average Marginal Effects (AME): population-averaged derivatives/contrasts
  - `:predictions` - Average Adjusted Predictions (AAP): population-averaged fitted values
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
- `scenarios=nothing`: Counterfactual scenarios (Dict mapping variables to values)
  - Example: `Dict(:x1 => 0, :x2 => [1, 2])` creates scenarios for all combinations
- `groups=nothing`: Grouping specification for stratified analysis
  - Simple: `:education` or `[:region, :gender]` for categorical grouping
  - Continuous: `(:income, 4)` for quartiles, `(:age, [25, 50, 75])` for thresholds
  - Nested: `:outer => :inner` for hierarchical grouping
- `contrasts::Symbol=:baseline`: Contrast type for categorical variables
  - `:baseline` - Compare each level to reference level
  - `:pairwise` - All pairwise comparisons between levels  
- `ci_alpha::Float64=0.05`: Type I error rate α for confidence intervals (confidence level = 1-α)
  - When specified, `ci_lower` and `ci_upper` columns are added to DataFrame output
- `vcov=GLM.vcov`: Covariance matrix function for standard errors
  - `GLM.vcov` - Model-based covariance matrix (default)
  - Custom function for robust/clustered standard errors
- `weights=nothing`: Observation weights for survey data or sampling corrections
  - `nothing` - No weights (equal weight for all observations, default)
  - `Symbol` - Column name containing weights (e.g., `:sampling_weight`)
  - `Vector` - Weight vector matching number of observations
  - Weights enable proper survey inference: `Σ(w_i * ∂ŷ_i/∂x_i) / Σ(w_i)`

# Returns
`EffectsResult` or `PredictionsResult` containing:
- Results DataFrame with estimates, standard errors, t-statistics, p-values
- Parameter gradients matrix for delta-method standard errors
- Analysis metadata (options used, model info, etc.)

# Statistical Notes
- Standard errors computed via delta method using full model covariance matrix
- Categorical variables use baseline contrasts vs reference levels
- All computations maintain statistical validity with zero tolerance for approximations

# Examples
```julia
# Average marginal effects for all explanatory variables
result = population_margins(model, data)
DataFrame(result)  # Convert to DataFrame

# Specific variables with response-scale effects
result = population_margins(model, data; vars=[:x1, :x2], scale=:response)

# Average elasticities
result = population_margins(model, data; vars=[:x1, :x2], measure=:elasticity)

# Semi-elasticities
result = population_margins(model, data; vars=[:x1], measure=:semielasticity_dyex)

# Average adjusted predictions  
result = population_margins(model, data; type=:predictions)

# Confidence intervals (99% confidence level)
result = population_margins(model, data; vars=[:x1], ci_alpha=0.01)
DataFrame(result)  # Includes ci_lower and ci_upper columns

# Counterfactual analysis: effects when x2 is set to 0 vs 1
result = population_margins(model, data; vars=[:x1], scenarios=Dict(:x2 => [0, 1]))

# Grouping examples
result = population_margins(model, data; groups=:education)  # By education level
result = population_margins(model, data; groups=(:income, 4))  # By income quartiles
result = population_margins(model, data; groups=:region => :gender)  # Nested grouping

# High-accuracy computation with automatic differentiation  
result = population_margins(model, data; backend=:ad, scale=:link)

# Survey data with sampling weights
result = population_margins(model, data; weights=:sampling_weight)

# Frequency weights for aggregated data
result = population_margins(model, data; weights=data.freq_weights)
```

# Frequency-Weighted Categorical Handling
Unspecified categorical variables automatically use population frequencies:
```julia
# Your data: education = 40% HS, 45% College, 15% Graduate
#           treated = 67% true, 33% false

result = population_margins(model, data; type=:effects)
# → Averages effects across actual population composition
# → Not arbitrary first levels or 50-50 assumptions
```

See also: [`profile_margins`](@ref) for effects at specific covariate combinations.
"""
function population_margins(
    model, data;
    type::Symbol=:effects, vars=nothing, scale::Symbol=:response,
    backend::Symbol=:ad, scenarios=nothing, groups=nothing, measure::Symbol=:effect,
    contrasts::Symbol=:baseline,
    ci_alpha::Float64=0.05, vcov=GLM.vcov, weights=nothing
)
    # Shared input validation for common parameters
    validate_margins_common_inputs(model, data, type, vars, scale, backend, measure, vcov)
    
    # Population-specific validation
    if !isnothing(scenarios)
        _validate_scenarios_specific(scenarios, vars, type)
    end
    if !isnothing(groups)
        _validate_groups_parameter(groups)
    end
    if !isnothing(weights)
        _validate_weights_parameter(weights, data)
    end
    # Single data conversion (consistent format throughout)
    data_nt = Tables.columntable(data)
    
    # Process weights parameter
    weights_vec = _process_weights_parameter(weights, data, data_nt)
    
    # Extract weight column name for variable filtering
    weight_col = weights isa Symbol ? weights : nothing
    
    # Handle vars parameter with improved validation
    if type === :effects
        vars = _process_vars_parameter(model, vars, data_nt, weight_col)
    else # type === :predictions
        vars = nothing  # Not needed for predictions
    end
    
    # Build zero-allocation engine with PopulationUsage optimization (including vcov function)
    engine = get_or_build_engine(PopulationUsage, model, data_nt, isnothing(vars) ? Symbol[] : vars, vcov)
    
    # Handle scenarios/groups parameters for counterfactual scenarios and grouping
    if !isnothing(scenarios) || !isnothing(groups)
        return _population_margins_with_contexts(engine, data_nt, vars, scenarios, groups, weights_vec, type, scale, backend, ci_alpha, measure)
    end
    
    if type === :effects
        df, G = _ame_continuous_and_categorical(engine, data_nt, scale, backend, measure; weights=weights_vec)  # → AME (both continuous and categorical)
        metadata = _build_metadata(; type, vars, scale, backend, measure, n_obs=length(first(data_nt)), model_type=typeof(model))
        
        # Store confidence interval parameters in metadata
        metadata[:alpha] = ci_alpha
        
        # Add analysis_type for format auto-detection
        metadata[:analysis_type] = :population
        
        # Store weights info in metadata
        metadata[:weighted] = !isnothing(weights_vec)
        
        # Extract raw components from DataFrame
        estimates = df.estimate
        standard_errors = df.se
        variables = df.variable  # The "x" in dy/dx
        terms = df.contrast
        
        return EffectsResult(estimates, standard_errors, variables, terms, nothing, nothing, G, metadata)
    else # :predictions  
        df, G = _population_predictions(engine, data_nt; scale, weights=weights_vec)  # → AAP
        metadata = _build_metadata(; type, vars=Symbol[], scale, backend, n_obs=length(first(data_nt)), model_type=typeof(model))
        
        # Store confidence interval parameters in metadata
        metadata[:alpha] = ci_alpha
        
        # Add analysis_type for format auto-detection
        metadata[:analysis_type] = :population
        
        # Store weights info in metadata  
        metadata[:weighted] = !isnothing(weights_vec)
        
        # Extract raw components from DataFrame
        estimates = df.estimate
        standard_errors = df.se
        
        return PredictionsResult(estimates, standard_errors, nothing, nothing, G, metadata)
    end
end


"""
    _get_model_formula_variables(model) -> Set{Symbol}

Extract explanatory variables from the model formula's right-hand side (RHS), excluding the dependent variable.

This function examines the model's formula specification and returns only variables that are 
actually used in the model, not all variables present in the data. This ensures that marginal
effects are computed only for variables that influence the model predictions.

# Arguments
- `model`: Fitted statistical model with a `.mf.f.rhs` formula component

# Returns
- `Set{Symbol}`: Set of variable names (symbols) from the model formula RHS

# Examples
```julia
# For model: y ~ x1 + x2 + region
variables = _get_model_formula_variables(model)  # -> Set([:x1, :x2, :region])
```

This filtering is critical for preventing errors when data contains extra columns
(e.g., ID variables, alternative specifications) that aren't part of the fitted model.
"""
function _get_model_formula_variables(model)
    # Extract all terms from the model formula RHS 
    formula_terms = StatsModels.termvars(model.mf.f.rhs)
    return Set{Symbol}(formula_terms)
end

"""
    _get_all_effect_variables(model, data_nt::NamedTuple, weight_col=nothing) -> Vector{Symbol}

Extract all explanatory variables from the model that can have marginal effects computed,
including both continuous and categorical variables.

This function performs three levels of filtering:
1. **Model filtering**: Only considers variables present in the model formula (not all data columns)
2. **Type filtering**: Only includes variables with computable effects (numeric, Bool, CategoricalArray)
3. **Weight filtering**: Excludes weight columns to prevent conflicts

# Arguments
- `model`: Fitted statistical model with formula specification
- `data_nt::NamedTuple`: Data in NamedTuple format (from Tables.columntable)
- `weight_col=nothing`: Weight column name to exclude (Symbol or nothing)

# Returns
- `Vector{Symbol}`: Variable names that can have marginal effects computed

# Variable Type Classification
- **Continuous variables**: `Int64`, `Float64` (but not `Bool`) → get derivatives
- **Categorical variables**: `Bool`, `CategoricalArray` → get discrete contrasts  
- **Excluded types**: `String`, `Date`, other non-numeric types

# Examples
```julia
# Model: y ~ x1 + x2 + education + treated
# Data also contains: id, date_collected, alternative_outcome
effect_vars = _get_all_effect_variables(model, data_nt, :sampling_weight)
# -> [:x1, :x2, :education, :treated]  (only model variables, no id/date/weight)
```

This filtering ensures statistical validity by computing effects only for variables
that actually influence the model predictions and have appropriate data types.
"""
function _get_all_effect_variables(model, data_nt::NamedTuple, weight_col=nothing)
    # Get variables that are actually in the model formula
    model_vars = _get_model_formula_variables(model)
    
    effect_vars = Symbol[]
    for (name, col) in pairs(data_nt)
        # CRITICAL FIX: Only process variables that are in the model formula
        if !(name in model_vars)
            continue
        end
        
        # Skip weight column - it's not a model variable
        if !isnothing(weight_col) && name == weight_col
            continue
        end
        
        # Include continuous variables (numeric types except Bool)
        # Include categorical variables (Bool, CategoricalArray, etc.)
        el_type = eltype(col)
        if (el_type <: Real && !(el_type <: Bool)) ||  # Continuous (Int, Float, but not Bool)
           el_type <: Bool ||                           # Bool (categorical)  
           hasproperty(col, :pool)                      # CategoricalArray
            push!(effect_vars, name)
        end
    end
    return effect_vars
end

"""
    _get_continuous_variables(model, data_nt::NamedTuple, weight_col=nothing) -> Vector{Symbol}

Extract continuous explanatory variables from the model that can have derivatives computed.

This function applies the same model filtering as `_get_all_effect_variables` but restricts
to continuous variables only, excluding categorical types (Bool, CategoricalArray).

# Arguments
- `model`: Fitted statistical model with formula specification  
- `data_nt::NamedTuple`: Data in NamedTuple format (from Tables.columntable)
- `weight_col=nothing`: Weight column name to exclude (Symbol or nothing)

# Returns
- `Vector{Symbol}`: Continuous variable names that can have derivatives computed

# Variable Type Classification
- **Included**: `Int64`, `Float64` (excluding `Bool`) → get derivatives via finite differences or AD
- **Excluded**: `Bool`, `CategoricalArray`, `String`, `Date`, etc. → not continuous

# Examples
```julia
# Model: y ~ x1 + x2 + education + treated  (x1,x2 continuous; education,treated categorical)
continuous_vars = _get_continuous_variables(model, data_nt)
# -> [:x1, :x2]  (only continuous model variables)
```

This is used for backward compatibility with `vars=:all_continuous` and when users
specifically want derivatives only (no discrete contrasts for categorical variables).
"""
function _get_continuous_variables(model, data_nt::NamedTuple, weight_col=nothing)
    # Get variables that are actually in the model formula
    model_vars = _get_model_formula_variables(model)
    
    continuous_vars = Symbol[]
    for (name, col) in pairs(data_nt)
        # CRITICAL FIX: Only process variables that are in the model formula
        if !(name in model_vars)
            continue
        end
        
        # Skip weight column - it's not a model variable
        if !isnothing(weight_col) && name == weight_col
            continue
        end
        
        # Continuous: numeric types except Bool
        if eltype(col) <: Real && !(eltype(col) <: Bool)
            push!(continuous_vars, name)
        end
    end
    return continuous_vars
end


"""
    _validate_vars_scenarios_overlap(vars, scenarios)

    Teaching validation that provides helpful error messages when users 
incorrectly specify the same variable in both vars and scenarios parameters.
"""
function _validate_vars_scenarios_overlap(vars, scenarios::Dict)
    # Convert vars to vector for consistent handling
    vars_vec = vars isa Symbol ? [vars] : (vars === :all_continuous ? Symbol[] : vars)
    
    # Find overlapping variables
    scenario_vars = Set(keys(scenarios))
    overlapping_vars = intersect(Set(vars_vec), scenario_vars)
    
    if !isempty(overlapping_vars)
        overlapping_list = join(overlapping_vars, ", ")
        
        # Create teaching error message
        error_msg = """
        Invalid parameter combination: Variables $(overlapping_list) appear in both 'vars' and 'scenarios'.
        
        What you're asking:
           vars = [$(join(vars_vec, ", "))]        -> "What's the marginal effect of these variables?"
           scenarios = Dict(...)  -> "Hold these variables constant at specific values"
        
        This is contradictory! You can't compute the effect of changing a variable while holding it constant.
        
        What you probably want:
        
        1. Effect of OTHER variables when $(overlapping_list) varies:
           population_margins(model, data; vars=[:other_var], scenarios=Dict(:$(overlapping_list) => [value1, value2]))
           
        2. Predicted outcomes when $(overlapping_list) varies:
           population_margins(model, data; type=:predictions, scenarios=Dict(:$(overlapping_list) => [value1, value2]))
           
        3. Effect of $(overlapping_list) within subgroups:
           population_margins(model, data; vars=[$(overlapping_list)], groups=:grouping_var)
           
        4. Just the effect of $(overlapping_list) (no scenarios):
           population_margins(model, data; vars=[$(overlapping_list)])
        
        Learn more about marginal effects vs predictions in the documentation.
        """
        
        throw(ArgumentError(error_msg))
    end
end

"""
    _validate_groups_parameter(groups)

Validate the unified groups parameter for population margins.
"""
function _validate_groups_parameter(groups)
    # Simple categorical grouping: :education or [:region, :gender]
    if groups isa Symbol 
        return
    end
    
    # Vector grouping (may contain mixed categorical and continuous specs)
    if groups isa AbstractVector
        # All symbols - simple categorical cross-tabulation
        if all(x -> x isa Symbol, groups)
            return
        end
        # Mixed vector - validate each element
        for spec in groups
            _validate_groups_parameter(spec)
        end
        return
    end
    
    # Continuous grouping: (:income, 4) or (:age, [25000, 50000])
    if groups isa Tuple && length(groups) == 2
        var, spec = groups
        if var isa Symbol
            # Quantile specification: (:income, 4)
            if spec isa Integer && spec > 0
                return
            end
            # Threshold specification: (:income, [25000, 50000])
            if spec isa AbstractVector && all(x -> x isa Real, spec)
                return
            end
        end
    end
    
    # Syntax for nested grouping: :region => :education
    if groups isa Pair
        outer_spec = groups.first
        inner_spec = groups.second
        # Recursively validate outer and inner specifications
        _validate_groups_parameter(outer_spec)
        # Inner specification can also be a Vector for multiple groupings
        if inner_spec isa AbstractVector
            for spec in inner_spec
                _validate_groups_parameter(spec)
            end
        else
            _validate_groups_parameter(inner_spec)
        end
        return
    end
    
    throw(ArgumentError("groups must be Symbol, Vector{Symbol}, continuous specification (Symbol, Int/Vector), or nested specification (outer => inner)"))
end

"""
    _process_vars_parameter(model, vars, data_nt::NamedTuple, weight_col=nothing) -> Vector{Symbol}

Process and validate the `vars` parameter for effects analysis, applying model-aware filtering.

This function handles all possible `vars` specifications and ensures only valid model variables
are returned for effects computation. It serves as the central validation point for variable
selection in both `population_margins` and `profile_margins`.

# Arguments
- `model`: Fitted statistical model with formula specification
- `vars`: User-specified variables (Symbol, Vector{Symbol}, :all_continuous, or nothing)
- `data_nt::NamedTuple`: Data in NamedTuple format (from Tables.columntable)  
- `weight_col=nothing`: Weight column name to exclude (Symbol or nothing)

# Returns
- `Vector{Symbol}`: Validated variable names for effects analysis

# Input Processing Rules
- `vars=nothing` → Auto-detect all model variables (continuous + categorical)
- `vars=:all_continuous` → Auto-detect continuous model variables only  
- `vars=:symbol` → Validate single variable exists in model and data
- `vars=[:x1, :x2]` → Validate all variables exist in model and data

# Validation Checks
1. **Model membership**: Variables must appear in model formula
2. **Data availability**: Variables must exist in provided data
3. **Weight conflicts**: Variables cannot conflict with weight column
4. **Non-empty results**: At least one valid variable must be found

# Examples
```julia
# Auto-detection (most common)
vars = _process_vars_parameter(model, nothing, data_nt)  # -> all model variables

# Explicit specification  
vars = _process_vars_parameter(model, [:x1, :x2], data_nt)  # -> [:x1, :x2] (if valid)

# Backward compatibility
vars = _process_vars_parameter(model, :all_continuous, data_nt)  # -> continuous only
```

# Errors Thrown
- `MarginsError`: When no valid variables found for effects analysis
- `ArgumentError`: When vars format is invalid or variables conflict with weights
"""
function _process_vars_parameter(model, vars, data_nt::NamedTuple, weight_col=nothing)
    if vars === nothing
        # Auto-detect ALL variables (both continuous and categorical) for comprehensive effects analysis
        effect_vars = _get_all_effect_variables(model, data_nt, weight_col)
        if isempty(effect_vars)
            throw(MarginsError("No explanatory variables found in data for effects analysis. Available variables: $(collect(keys(data_nt)))"))
        end
        return effect_vars
    elseif vars === :all_continuous
        # Backwards compatibility: only continuous variables
        continuous_vars = _get_continuous_variables(model, data_nt, weight_col)
        if isempty(continuous_vars)
            throw(MarginsError("No continuous explanatory variables found in data for effects analysis. Available variables: $(collect(keys(data_nt)))"))
        end
        return continuous_vars
    elseif vars isa Symbol
        vars_vec = [vars]
        _validate_variables(data_nt, vars_vec)
        _validate_vars_weight_conflict(vars_vec, weight_col)
        return vars_vec
    elseif vars isa Vector{Symbol}
        if isempty(vars)
            throw(ArgumentError("vars cannot be an empty vector. Use vars=nothing or vars=:all_continuous to auto-detect variables, or specify at least one variable."))
        end
        _validate_variables(data_nt, vars)
        _validate_vars_weight_conflict(vars, weight_col)
        return vars
    else
        throw(ArgumentError("vars must be Symbol, Vector{Symbol}, or :all_continuous"))
    end
end

"""
    _validate_vars_weight_conflict(vars, weight_col)

Validate that no variable in vars conflicts with the weight column.

# Arguments
- `vars`: Vector{Symbol} of variables for effects analysis
- `weight_col`: Weight column name (Symbol or nothing)

# Throws
- `ArgumentError`: If any variable in vars matches the weight column
"""
function _validate_vars_weight_conflict(vars::Vector{Symbol}, weight_col)
    if !isnothing(weight_col) && weight_col in vars
        throw(ArgumentError(
            "Variable :$weight_col cannot be used both as a weight (weights=:$weight_col) " *
            "and as a variable for effects analysis (vars includes :$weight_col). " *
            "A column cannot serve both roles simultaneously. " *
            "Choose different columns for weights and effects, or remove :$weight_col from vars."
        ))
    end
end

"""
    _validate_weights_parameter(weights, data)

Validate the weights parameter for population_margins().

# Arguments
- `weights`: The weights specification (Symbol, Vector, or nothing)
- `data`: Original data (for Tables.jl compatibility check)

# Throws
- `ArgumentError`: If weights specification is invalid
"""
function _validate_weights_parameter(weights, data)
    if weights isa Symbol
        # Check that the column exists in the data
        if !Tables.columnaccess(data) || !haskey(Tables.columntable(data), weights)
            throw(ArgumentError("weights column :$weights not found in data"))
        end
    elseif weights isa AbstractVector
        # Check that weights vector has right length
        n_data = Tables.rowaccess(data) ? length(Tables.rows(data)) : length(first(Tables.columns(data)))
        if length(weights) != n_data
            throw(ArgumentError("weights vector length ($(length(weights))) must match number of observations ($n_data)"))
        end
        
        # Check that all weights are non-negative
        if !all(w >= 0 for w in weights)
            throw(ArgumentError("all weights must be non-negative"))
        end
        
        # Check that at least some weights are positive
        if all(w == 0 for w in weights)
            throw(ArgumentError("at least some weights must be positive"))
        end
    else
        throw(ArgumentError("weights must be Symbol (column name) or Vector{<:Real}"))
    end
end

"""
    _process_weights_parameter(weights, data, data_nt) -> Vector{Float64} or nothing

Process the weights parameter into a standardized Float64 vector.

# Arguments
- `weights`: The weights specification (Symbol, Vector, or nothing)
- `data`: Original data (unused but kept for API consistency)
- `data_nt`: Data in NamedTuple format

# Returns
- `Vector{Float64}`: Processed weights vector
- `nothing`: If no weights specified

# Examples
```julia
# No weights
weights_vec = _process_weights_parameter(nothing, data, data_nt)  # -> nothing

# Column name
weights_vec = _process_weights_parameter(:sampling_weight, data, data_nt)  # -> Float64 vector

# Direct vector
weights_vec = _process_weights_parameter([1.0, 2.0, 1.5], data, data_nt)  # -> Float64 vector
```
"""
function _process_weights_parameter(weights, data, data_nt)
    if isnothing(weights)
        return nothing
    elseif weights isa Symbol
        # Extract weights column from data
        weights_col = getproperty(data_nt, weights)
        return Float64.(weights_col)  # Convert to Float64
    elseif weights isa AbstractVector
        return Float64.(weights)  # Convert to Float64
    else
        throw(ArgumentError("Unexpected weights type: $(typeof(weights))"))  # Should not reach here due to validation
    end
end