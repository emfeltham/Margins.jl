# Shared validation utilities for margins computation
# Contains only genuinely common validation logic between population_margins() and profile_margins()

"""
    validate_margins_common_inputs(model, data, type, vars, scale, backend, measure, vcov) -> Matrix{Float64}

Shared input validation for both population_margins and profile_margins.
Only validates the truly common requirements - each function handles its specific validation separately.
Validation runs on every call (nothing is cached), and returns the covariance matrix Σ that
`vcov` gives for `model` (`validate_vcov_parameter`), which the caller passes to `build_engine`.
"""
function validate_margins_common_inputs(model, data, type, vars, scale, backend, measure, vcov)
    validate_required_args(model, data)
    validate_model_methods(model)
    validate_population_parameters(type, scale, backend, measure, vars)
    return validate_vcov_parameter(vcov, model)
end

"""
    validate_required_args(model, data)

Check that required arguments are not nothing.
Eliminates duplicated null checks between both margin functions.
"""
function validate_required_args(model, data)
    isnothing(model) && throw(ArgumentError("model cannot be nothing"))
    isnothing(data) && throw(ArgumentError("data cannot be nothing"))
end

"""
    validate_model_methods(model)

Validate that model supports required methods for margins computation. Its covariance is
checked where it is evaluated, once per call (`validate_vcov_parameter`).
Eliminates the duplicated try/catch blocks between both margin functions.
"""
function validate_model_methods(model)
    try
        coef(model)
    catch e
        throw(ArgumentError("model must support coef() method (fitted statistical model required)"))
    end
    return nothing
end

"""
    _validate_scenarios_specific(scenarios, vars, type)

Population-specific validation for scenarios parameter.
Accepts both NamedTuple and Tuple of Pairs syntax.
"""
function _validate_scenarios_specific(scenarios, vars, type)
    # Accept both NamedTuple and Tuple of Pairs
    if !(scenarios isa NamedTuple) && !_is_tuple_of_pairs(scenarios)
        throw(ArgumentError(
            "scenarios parameter must be either:\n" *
            "  1. NamedTuple: scenarios=(treatment=[0, 1], income=[30000, 50000])\n" *
            "  2. Tuple of Pairs: scenarios=(:treatment => [0, 1], :income => [30000, 50000])"
        ))
    end

    # Validate that all Pairs have Symbol keys (if using Pair syntax)
    if _is_tuple_of_pairs(scenarios)
        for (i, item) in enumerate(scenarios)
            if !(item isa Pair)
                throw(ArgumentError("scenarios element $i is not a Pair. All elements must be Pair{Symbol, Any} when using Pair syntax."))
            end
            if !(item.first isa Symbol)
                throw(ArgumentError("scenarios Pair $i has non-Symbol key '$(item.first)'. Keys must be Symbols (e.g., :predicate => values)."))
            end
        end
    end

    # Teaching validation: Check for vars/scenarios overlap
    # Convert to NamedTuple for consistent validation
    scenarios_nt = _normalize_scenarios_to_namedtuple(scenarios)
    if !isnothing(vars) && type == :effects
        _validate_vars_scenarios_overlap(vars, scenarios_nt)
    end
end

"""
    _is_tuple_of_pairs(scenarios) -> Bool

Check if scenarios is a Tuple potentially containing Pairs.
Returns true for non-NamedTuple tuples, false otherwise.
"""
function _is_tuple_of_pairs(scenarios)
    # NamedTuples are also Tuples in Julia, so we need to exclude them explicitly
    return scenarios isa Tuple && !(scenarios isa NamedTuple)
end