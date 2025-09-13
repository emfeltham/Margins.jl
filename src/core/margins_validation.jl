# Shared validation utilities for margins computation
# Contains only genuinely common validation logic between population_margins() and profile_margins()

# Cache for validated models to avoid repeated expensive vcov() calls
const VALIDATED_MODELS_CACHE = Set{UInt64}()

"""
    validate_margins_common_inputs(model, data, type, vars, scale, backend, measure, vcov)

Shared input validation for both population_margins and profile_margins.
Only validates the truly common requirements - each function handles its specific validation separately.
"""
function validate_margins_common_inputs(model, data, type, vars, scale, backend, measure, vcov)
    validate_required_args(model, data)
    validate_model_methods(model)
    validate_population_parameters(type, scale, backend, measure, vars)
    validate_vcov_parameter(vcov, model)
end

"""
    validate_required_args(model, data)

Check that required arguments are not nothing.
Eliminates duplicated null checks between both margin functions.
"""
function validate_required_args(model, data)
    model === nothing && throw(ArgumentError("model cannot be nothing"))
    data === nothing && throw(ArgumentError("data cannot be nothing"))
end

"""
    validate_model_methods(model)

Validate that model supports required methods for margins computation.
Uses caching to avoid repeated expensive vcov() calls for large models.
Eliminates the duplicated try/catch blocks between both margin functions.
"""
function validate_model_methods(model)
    # Create a lightweight cache key for the model
    # Use hash of model type and coefficient vector (not the full model object)
    model_key = hash((typeof(model), coef(model)))
    
    # Skip expensive validation if we've already validated this model
    if model_key in VALIDATED_MODELS_CACHE
        return nothing
    end
    
    try
        coef(model)
    catch e
        throw(ArgumentError("model must support coef() method (fitted statistical model required)"))
    end
    
    try
        GLM.vcov(model)
    catch e
        throw(ArgumentError("model must support vcov() method (covariance matrix required for standard errors)"))
    end
    
    # Cache successful validation
    push!(VALIDATED_MODELS_CACHE, model_key)
    return nothing
end

"""
    _validate_scenarios_specific(scenarios, vars, type)

Population-specific validation for scenarios parameter.
"""
function _validate_scenarios_specific(scenarios, vars, type)
    if !(scenarios isa NamedTuple)
        throw(ArgumentError("scenarios parameter must be a NamedTuple specifying counterfactual scenarios. Example: scenarios=(treatment=[0, 1], income=[30000, 50000])"))
    end
    
    # Teaching validation: Check for vars/scenarios overlap
    if !isnothing(vars) && type == :effects
        _validate_vars_scenarios_overlap(vars, scenarios)
    end
end