# Centralized parameter validation for Margins.jl API functions.
#
# This file provides consistent parameter validation across the entire API,
# ensuring uniform error messages and eliminating duplicate validation code.
# All validation functions follow the principle of failing fast with clear,
# actionable error messages.

"""
    validate_type_parameter(type::Symbol)

Validate the `type` parameter for margins functions.
Must be either `:effects` or `:predictions`.
"""
function validate_type_parameter(type::Symbol)
    type ∉ (:effects, :predictions) && 
        throw(ArgumentError("type must be :effects or :predictions, got :$(type)"))
end

"""
    validate_scale_parameter(scale::Symbol)

Validate the `scale` parameter for margins functions.
Must be either `:link` (link scale) or `:response` (response scale).

"""
function validate_scale_parameter(scale::Symbol)  
    scale ∉ (:link, :response) && 
        throw(ArgumentError("scale must be :link or :response, got :$(scale)"))
end

"""
    _validate_reference_grid_column_type(col, col_name::Symbol)

Validate that a column in a reference grid has a supported data type.
Implements explicit error policy - fails fast with clear message for unsupported types.

# Supported types:
- Numeric: Int64, Float64 (continuous variables)
- Bool (categorical via frequency mixture)
- CategoricalArray (categorical variables)
- AbstractString (limited support in non-optimized paths only)

# Throws
- `MarginsError`: For unsupported data types that could produce invalid statistical results
"""
function _validate_reference_grid_column_type(col, col_name)
    eltype_val = eltype(col)
    
    # Check for supported types - order matters, check specific cases first
    if col isa CategoricalArray
        # CategoricalArray - supported (check this first before eltype checks)
        return
    elseif any(x -> x isa CategoricalMixture, col)
        # Column contains CategoricalMixture objects - supported for fractional specifications
        return
    elseif eltype_val <: CategoricalArrays.CategoricalValue
        # Vector of CategoricalValue objects (extracted from CategoricalArray) - supported
        return
    elseif eltype_val <: Real && !(eltype_val <: Bool)
        # Numeric types (Int64, Float64, etc.) - supported
        return
    elseif eltype_val <: Bool
        # Bool - supported as categorical
        return  
    elseif eltype_val <: AbstractString
        # String - limited support but allowed in reference grids
        return
    else
        # Unsupported types - explicit error
        throw(MarginsError("Unsupported data type $(eltype_val) for variable '$col_name' in reference grid. " *
                          "Statistical correctness cannot be guaranteed for unknown data types. " *
                          "Supported types: numeric (Int64, Float64), Bool, CategoricalArray, AbstractString, CategoricalMixture. " *
                          "Consider converting complex types to supported categorical or numeric formats."))
    end
end

"""
    validate_backend_parameter(backend::Symbol)

Validate the `backend` parameter for derivative computation.
Must be `:ad` (automatic differentiation) or `:fd` (finite differences).
"""
function validate_backend_parameter(backend::Symbol)
    backend ∉ (:ad, :fd) && 
        throw(ArgumentError("backend must be :ad or :fd, got :$(backend)"))
end

"""
    validate_measure_parameter(measure::Symbol, type::Symbol)

Validate the `measure` parameter for effects computation.
Must be one of `:effect`, `:elasticity`, `:semielasticity_dyex`, `:semielasticity_eydx`.
Only applies when `type=:effects`.
"""
function validate_measure_parameter(measure::Symbol, type::Symbol)
    valid_measures = (:effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx)
    measure ∉ valid_measures && 
        throw(ArgumentError("measure must be one of $(valid_measures), got :$(measure)"))
    
    # Measure only applies to effects
    if type === :predictions && measure !== :effect
        throw(ArgumentError("measure parameter only applies when type=:effects"))
    end
end

"""
    validate_vars_parameter(vars, type::Symbol)

Validate the `vars` parameter for variable selection.
Throws error if vars is specified for predictions (where it's invalid).
"""
function validate_vars_parameter(vars, type::Symbol)
    if !isnothing(vars) && type === :predictions
        throw(ArgumentError("vars parameter is incompatible with type=:predictions. Remove vars parameter or change type to :effects."))
    end
end

"""
    validate_grouping_parameters(over, within, by)

Validate grouping parameters for consistency.
Ensures no conflicts between different grouping specifications.
"""
function validate_grouping_parameters(over, within, by)
    # Check for mutual exclusivity where appropriate
    if !isnothing(over) && !isnothing(within)
        # This is actually allowed - within creates nested grouping
        # No validation error needed
    end
    
    # Validate that grouping variables are Symbols or Vector{Symbol}
    for (name, param) in [(:over, over), (:within, within), (:by, by)]
        if !isnothing(param)
            if !(param isa Symbol || (param isa AbstractVector && all(x -> x isa Symbol, param)))
                throw(ArgumentError("$(name) must be Symbol or Vector{Symbol}, got $(typeof(param))"))
            end
        end
    end
end

"""
    validate_common_parameters(type, scale, backend, measure=:effect, vars=nothing)

Validate all common API parameters in a single call.
This is the main validation entry point for most margins functions.

# Arguments
- `type::Symbol`: Effect type (`:effects` or `:predictions`)  
- `scale::Symbol`: Target scale (`:link` or `:response`)
- `backend::Symbol`: Computation backend (`:ad` or `:fd`)
- `measure::Symbol`: Effect measure (default `:effect`)
- `vars`: Variable selection (default `nothing`)

# Examples
```julia
# Basic validation
validate_common_parameters(:effects, :response, :ad)

# Full validation with all parameters  
validate_common_parameters(:effects, :link, :ad, :elasticity, [:x1, :x2])
```
"""
function validate_common_parameters(type, scale, backend, measure=:effect, vars=nothing)
    validate_type_parameter(type)
    validate_scale_parameter(scale)  # Use new scale validation
    validate_backend_parameter(backend)
    validate_measure_parameter(measure, type)
    validate_vars_parameter(vars, type)
end

"""
    validate_profile_parameters(at, type, scale, backend, measure=:effect, vars=nothing)

Validate parameters specific to profile margins functions.
Includes all common parameter validation plus profile-specific checks.
"""
function validate_profile_parameters(at, type, scale, backend, measure=:effect, vars=nothing)
    validate_at_parameter(at)
    validate_common_parameters(type, scale, backend, measure, vars)
end

"""
    validate_population_parameters(type, scale, backend, measure=:effect, vars=nothing)

Validate parameters specific to population margins functions.
Currently identical to common validation but provided for API symmetry.
"""
function validate_population_parameters(type, scale, backend, measure=:effect, vars=nothing)
    validate_common_parameters(type, scale, backend, measure, vars)
end

"""
    validate_vcov_parameter(vcov, model)

Validate the `vcov` parameter for margins functions.
Must be a function that takes a model and returns a covariance matrix.

# Arguments
- `vcov`: Function that computes covariance matrix from model
- `model`: Statistical model to test the vcov function against

# Examples
```julia
validate_vcov_parameter(GLM.vcov, model)
validate_vcov_parameter(CovarianceMatrices.HC1, model)
```
"""
function validate_vcov_parameter(vcov, model)
    # Create cache key for this specific vcov + model combination
    vcov_key = hash((vcov, typeof(model), coef(model)))
    
    # Skip expensive validation if we've already validated this combination
    if vcov_key in Margins.VALIDATED_MODELS_CACHE
        return nothing
    end
    
    # Test that vcov works with the model
    try
        if isa(vcov, Function)
            # vcov is a function like GLM.vcov
            vcov_result = vcov(model)
        else
            # vcov is likely a CovarianceMatrices estimator like HC1()
            # Use Base.invokelatest to handle the optional dependency
            vcov_module = Base.require(Main, :CovarianceMatrices)
            vcov_result = Base.invokelatest(vcov_module.vcov, vcov, model)
        end
        
        if !isa(vcov_result, AbstractMatrix)
            throw(ArgumentError("vcov must produce a matrix when applied to the model"))
        end
    catch e
        throw(ArgumentError("vcov failed when applied to provided model: $e"))
    end
    
    # Cache successful validation
    push!(Margins.VALIDATED_MODELS_CACHE, vcov_key)
    return nothing
end

"""
    _filter_vars_for_measure(vars, data_nt, measure, vars_explicit) -> Vector{Symbol}

Enforce that econometric measures other than `:effect` are only requested for continuous
variables, and return the variable list to actually compute.

Elasticities and semi-elasticities are defined through a derivative ∂y/∂x scaled by x̄ and/or
ȳ. A discrete level contrast has no such derivative and no meaningful x̄, so applying an
elasticity factor to it is not defined. Stata rejects factor variables in
`eyex()`/`dyex()`/`eydx()`; R's marginaleffects applies slope transformations to numeric
predictors only. Margins.jl previously computed the raw contrast and labeled the result
`AME (elasticity)`, which silently misreported it.

Behaviour:
- A discrete variable the caller named explicitly is an error — the request cannot be
  honoured as stated.
- A discrete variable that was auto-detected (`vars=nothing`) is dropped with an `@info`,
  so `population_margins(model, data; measure=:elasticity)` still works and returns only
  the rows it can define. If nothing continuous remains, that is an error.

# Arguments
- `vars::Vector{Symbol}`: Resolved variable list from `_process_vars_parameter`
- `data_nt::NamedTuple`: Data columns (used for continuity detection)
- `measure::Symbol`: Requested measure
- `vars_explicit::Bool`: Whether the caller supplied `vars` (vs. auto-detection)

# Returns
- `Vector{Symbol}`: `vars` unchanged for `measure === :effect`, otherwise continuous only
"""
function _filter_vars_for_measure(vars, data_nt::NamedTuple, measure::Symbol, vars_explicit::Bool)
    (measure === :effect || isnothing(vars)) && return vars

    discrete = [v for v in vars if !_is_continuous_variable(getproperty(data_nt, v))]
    isempty(discrete) && return vars

    discrete_list = join(string.(discrete), ", ")
    if vars_explicit
        throw(ArgumentError(
            "measure=:$measure is only defined for continuous variables, but vars includes " *
            "discrete variable(s): $discrete_list. Elasticities scale a derivative ∂y/∂x by " *
            "x̄ and/or ȳ; a discrete level contrast has no such derivative. Request the " *
            "discrete variables in a separate call with measure=:effect."
        ))
    end

    continuous = [v for v in vars if !(v in discrete)]
    if isempty(continuous)
        throw(ArgumentError(
            "measure=:$measure requires at least one continuous variable, but all detected " *
            "variables are discrete: $discrete_list. Use measure=:effect for discrete contrasts."
        ))
    end

    @info "Skipping measure=:$measure for discrete variable(s): $discrete_list " *
          "(elasticity measures are defined for continuous variables only)."
    return continuous
end

# End of validation.jl
