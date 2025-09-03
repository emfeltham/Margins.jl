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

# Scale conversion functions removed - now using scale directly throughout

# Legacy support for transition period
"""
    validate_target_parameter(target::Symbol)

DEPRECATED: Use `validate_scale_parameter()` instead.
Validates legacy `:eta`/`:mu` parameters and converts to `:link`/`:response`.
"""
function validate_target_parameter(target::Symbol)  
    if target ∈ (:eta, :mu)
        # Legacy support - convert silently
        return target === :eta ? :link : :response
    elseif target ∈ (:link, :response)
        # Already using new terminology
        return target
    else
        throw(ArgumentError("scale must be :link or :response (or legacy :eta/:mu), got :$(target)"))
    end
end

"""
    validate_backend_parameter(backend::Symbol)

Validate the `backend` parameter for derivative computation.
Must be `:ad` (automatic differentiation), `:fd` (finite differences), or `:auto`.
"""
function validate_backend_parameter(backend::Symbol)
    backend ∉ (:ad, :fd, :auto) && 
        throw(ArgumentError("backend must be :ad, :fd, or :auto, got :$(backend)"))
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
    if vars !== nothing && type === :predictions
        throw(ArgumentError("vars parameter is incompatible with type=:predictions. Remove vars parameter or change type to :effects."))
    end
end

"""
    validate_at_parameter(at)

Validate the `at` parameter for profile specifications.
Must be `:means`, a Dict, or Vector{Dict}.
"""
function validate_at_parameter(at)
    if !(at === :means || at isa Dict || (at isa Vector && !isempty(at) && all(x -> x isa Dict, at)))
        throw(ArgumentError("at must be :means, Dict{Symbol,Any}, or Vector{Dict{Symbol,Any}}, got $(typeof(at))"))
    end
end

"""
    validate_grouping_parameters(over, within, by)

Validate grouping parameters for consistency.
Ensures no conflicts between different grouping specifications.
"""
function validate_grouping_parameters(over, within, by)
    # Check for mutual exclusivity where appropriate
    if over !== nothing && within !== nothing
        # This is actually allowed - within creates nested grouping
        # No validation error needed
    end
    
    # Validate that grouping variables are Symbols or Vector{Symbol}
    for (name, param) in [(:over, over), (:within, within), (:by, by)]
        if param !== nothing
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
- `backend::Symbol`: Computation backend (`:ad`, `:fd`, or `:auto`)
- `measure::Symbol`: Effect measure (default `:effect`)
- `vars`: Variable selection (default `nothing`)

# Examples
```julia
# Basic validation
validate_common_parameters(:effects, :response, :auto)

# Full validation with all parameters  
validate_common_parameters(:effects, :link, :fd, :elasticity, [:x1, :x2])
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
end

# End of validation.jl