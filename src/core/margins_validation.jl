# Shared validation utilities for margins computation
# Contains only genuinely common validation logic between population_margins() and profile_margins()

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
Eliminates the duplicated try/catch blocks between both margin functions.
"""
function validate_model_methods(model)
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
end

"""
    _validate_scenarios_specific(scenarios, vars, type)

Population-specific validation for scenarios parameter.
"""
function _validate_scenarios_specific(scenarios, vars, type)
    if !(scenarios isa Dict)
        throw(ArgumentError("scenarios parameter must be a Dict specifying counterfactual scenarios"))
    end
    
    # Teaching validation: Check for vars/scenarios overlap
    if !isnothing(vars) && type == :effects
        _validate_vars_scenarios_overlap(vars, scenarios)
    end
end