# population/core.jl - Main population_margins() function with compilation caching

# Removed module imports - functions now in main namespace

# Global cache for compiled formulas (MARGINS_GUIDE.md pattern)
# Unified caching system (see engine/caching.jl)
# Removed: const COMPILED_CACHE = Dict{UInt64, Any}()  # Now unified in engine/caching.jl

"""
    population_margins(model, data; type=:effects, vars=nothing, target=:mu, backend=:auto, scenarios=nothing, groups=nothing, measure=:effect, vcov=GLM.vcov) -> MarginsResult

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
  - Defaults to all continuous variables (numeric types except Bool)
- `target::Symbol=:mu`: Target scale for computation
  - `:mu` - Response scale (default, applies inverse link function)  
  - `:eta` - Linear predictor scale (link scale)
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
  - Nested: `(main=:region, within=:gender)` for hierarchical grouping

# Returns
`MarginsResult` containing:
- Results DataFrame with estimates, standard errors, t-statistics, p-values
- Parameter gradients matrix for delta-method standard errors
- Analysis metadata (options used, model info, etc.)

# Statistical Notes
- Standard errors computed via delta method using full model covariance matrix
- Categorical variables use baseline contrasts vs reference levels
- All computations maintain statistical validity with zero tolerance for approximations

# Examples
```julia
# Average marginal effects for all continuous variables
result = population_margins(model, data)
DataFrame(result)  # Convert to DataFrame

# Specific variables with response-scale effects
result = population_margins(model, data; vars=[:x1, :x2], target=:mu)

# Average elasticities (NEW in Phase 3)
result = population_margins(model, data; vars=[:x1, :x2], measure=:elasticity)

# Semi-elasticities (NEW in Phase 3)
result = population_margins(model, data; vars=[:x1], measure=:semielasticity_dyex)

# Average adjusted predictions  
result = population_margins(model, data; type=:predictions)

# Counterfactual analysis: effects when x2 is set to 0 vs 1
result = population_margins(model, data; vars=[:x1], scenarios=Dict(:x2 => [0, 1]))

# Grouping examples (NEW unified syntax)
result = population_margins(model, data; groups=:education)  # By education level
result = population_margins(model, data; groups=(:income, 4))  # By income quartiles  
result = population_margins(model, data; groups=(main=:region, within=:gender))  # Nested

# High-performance production use with finite differences
result = population_margins(model, data; backend=:ad, target=:eta)
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
function population_margins(model, data; type::Symbol=:effects, vars=nothing, target::Symbol=:mu, backend::Symbol=:auto, scenarios=nothing, groups=nothing, measure::Symbol=:effect, vcov=GLM.vcov)
    # Input validation
    _validate_population_inputs(model, data, type, vars, target, backend, scenarios, measure, groups, vcov)
    # Single data conversion (consistent format throughout)
    data_nt = Tables.columntable(data)
    
    # Handle vars parameter with improved validation
    if type === :effects
        vars = _process_vars_parameter(model, vars, data_nt)
    else # type === :predictions
        vars = nothing  # Not needed for predictions
    end
    
    # Proper backend selection
    # Population margins default to :ad for consistency across all functions
    recommended_backend = backend === :auto ? :ad : backend
    
    # Build zero-allocation engine with caching (including vcov function)
    engine = get_or_build_engine(model, data_nt, vars === nothing ? Symbol[] : vars, vcov)
    
    # Handle scenarios/groups parameters for counterfactual scenarios and grouping
    if scenarios !== nothing || groups !== nothing
        return _population_margins_with_contexts(engine, data_nt, vars, scenarios, groups; type, target, backend=recommended_backend)
    end
    
    if type === :effects
        df, G = _ame_continuous_and_categorical(engine, data_nt; target, backend=recommended_backend, measure)  # → AME (both continuous and categorical)
        metadata = _build_metadata(; type, vars, target, backend=recommended_backend, measure, n_obs=length(first(data_nt)), model_type=typeof(model))
        
        # Add analysis_type for format auto-detection
        metadata[:analysis_type] = :population
        
        # Extract raw components from DataFrame
        estimates = df.estimate
        standard_errors = df.se
        terms = df.term
        
        return MarginsResult(estimates, standard_errors, terms, nothing, nothing, G, metadata)
    else # :predictions  
        df, G = _population_predictions(engine, data_nt; target)  # → AAP
        metadata = _build_metadata(; type, vars=Symbol[], target, backend=recommended_backend, n_obs=length(first(data_nt)), model_type=typeof(model))
        
        # Add analysis_type for format auto-detection
        metadata[:analysis_type] = :population
        
        # Extract raw components from DataFrame
        estimates = df.estimate
        standard_errors = df.se
        terms = df.term
        
        return MarginsResult(estimates, standard_errors, terms, nothing, nothing, G, metadata)
    end
end


"""
    _get_continuous_variables(model, data_nt) -> Vector{Symbol}

Extract continuous explanatory variables from data, filtering out categorical types and the dependent variable.
"""
function _get_continuous_variables(model, data_nt::NamedTuple)
    # Get dependent variable from model formula
    dependent_var = Symbol(model.mf.f.lhs)
    
    continuous_vars = Symbol[]
    for (name, col) in pairs(data_nt)
        # Skip dependent variable - we only want explanatory variables for marginal effects
        if name == dependent_var
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
    _validate_population_inputs(model, data, type, vars, target, backend, scenarios, measure, groups, vcov)

Validate inputs to population_margins() with clear Julia-style error messages.
"""
function _validate_population_inputs(model, data, type::Symbol, vars, target::Symbol, backend::Symbol, scenarios, measure::Symbol, groups, vcov)
    # Validate required arguments
    if model === nothing
        throw(ArgumentError("model cannot be nothing"))
    end
    
    if data === nothing
        throw(ArgumentError("data cannot be nothing"))
    end
    
    # Use centralized validation for common parameters
    validate_population_parameters(type, target, backend, measure, vars)
    
    # Validate vcov parameter
    validate_vcov_parameter(vcov, model)
    
    # Validate vars parameter for effects
    if type === :effects && vars !== nothing
        if !(vars isa Symbol || vars isa Vector{Symbol} || vars === :all_continuous)
            throw(ArgumentError("vars must be Symbol, Vector{Symbol}, or :all_continuous for effects analysis"))
        end
    end
    
    # Validate scenarios parameter (replaces 'at' for population margins)
    if scenarios !== nothing && !(scenarios isa Dict)
        throw(ArgumentError("scenarios parameter must be a Dict specifying counterfactual scenarios"))
    end
    
    # Teaching validation: Check for vars/scenarios overlap
    if scenarios !== nothing && vars !== nothing && type === :effects
        _validate_vars_scenarios_overlap(vars, scenarios)
    end
    
    # Validate groups parameter (unified grouping system)
    if groups !== nothing
        _validate_groups_parameter(groups)
    end
    
    # Validate model has required methods
    try
        coef(model)
    catch e
        throw(ArgumentError("model must support coef() method (fitted statistical model required)"))
    end
    
    try
        vcov(model)
    catch e
        throw(ArgumentError("model must support vcov() method (covariance matrix required for standard errors)"))
    end
end

"""
    _validate_vars_scenarios_overlap(vars, scenarios)

Phase 5: Teaching validation that provides helpful error messages when users 
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
Phase 3: Clean syntax with no legacy compatibility.
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
    
    # Phase 3: => syntax for nested grouping: :region => :education
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
    _process_vars_parameter(model, vars, data_nt) -> Vector{Symbol}

Process and validate the vars parameter with model awareness to exclude dependent variable.
"""
function _process_vars_parameter(model, vars, data_nt::NamedTuple)
    if vars === nothing || vars === :all_continuous
        continuous_vars = _get_continuous_variables(model, data_nt)
        if isempty(continuous_vars)
            throw(MarginsError("No continuous explanatory variables found in data for effects analysis. Available variables: $(collect(keys(data_nt)))"))
        end
        return continuous_vars
    elseif vars isa Symbol
        vars_vec = [vars]
        _validate_variables(data_nt, vars_vec)
        return vars_vec
    elseif vars isa Vector{Symbol}
        if isempty(vars)
            throw(ArgumentError("vars cannot be an empty vector. Use vars=nothing or vars=:all_continuous to auto-detect variables, or specify at least one variable."))
        end
        _validate_variables(data_nt, vars)
        return vars
    else
        throw(ArgumentError("vars must be Symbol, Vector{Symbol}, or :all_continuous"))
    end
end