# workspace.jl
# Enhanced version with FormulaCompiler integration and performance tracking

mutable struct MarginalEffectsWorkspace
    # FormulaCompiler components
    compiled_formula::CompiledFormula
    derivative_formulas::Dict{Symbol,CompiledDerivativeFormula}
    column_data::NamedTuple
    
    # Pre-allocated buffers
    model_row_buffer::Vector{Float64}
    derivative_buffer::Vector{Float64}
    gradient_accumulator::Vector{Float64}
    computation_buffer::Vector{Float64}
    
    # Link functions (ADD THESE)
    inverse_link_function::Function
    first_derivative::Function
    second_derivative::Function
    
    # Performance tracking
    total_model_evaluations::Int
    total_derivative_evaluations::Int
    analytical_derivative_successes::Int
    forwarddiff_fallbacks::Int
    creation_time_ns::Int
end

function MarginalEffectsWorkspace(model::StatisticalModel, data, focal_variables::Vector{Symbol})
    start_time = time_ns()
    
    # Convert to efficient column-table format and validate
    column_data = Tables.columntable(data)
    validate_data_structure(column_data)
    
    # Compile model formula using FormulaCompiler.jl
    compiled_formula = compile_formula(model)
    parameter_count = length(compiled_formula)
    
    # Extract link functions from model
    inverse_link, first_deriv, second_deriv = extract_link_functions(model)
    
    # Pre-compile analytical derivatives using FormulaCompiler.jl
    derivative_formulas = Dict{Symbol,CompiledDerivativeFormula}()
    
    for focal_var in focal_variables
        try
            derivative_compiled = compile_derivative_formula(compiled_formula, focal_var)
            derivative_formulas[focal_var] = derivative_compiled
        catch e
            @warn "Failed to compile analytical derivatives for $focal_var: $e. Will use fallback during evaluation."
        end
    end
    
    creation_time = time_ns() - start_time
    
    return MarginalEffectsWorkspace(
        compiled_formula,
        derivative_formulas,
        column_data,
        Vector{Float64}(undef, parameter_count),
        Vector{Float64}(undef, parameter_count),
        Vector{Float64}(undef, parameter_count),
        Vector{Float64}(undef, parameter_count),
        inverse_link,      # Store the link functions
        first_deriv,
        second_deriv,
        0, 0, 0, 0, Int(creation_time)
    )
end

# Simple accessor function
function get_inverse_link_function(workspace::MarginalEffectsWorkspace)
    return workspace.inverse_link_function
end

# Convenience constructor
function MarginalEffectsWorkspace(model::StatisticalModel, data)
    column_data = Tables.columntable(data)
    continuous_vars = filter(var -> is_continuous_variable(var, column_data), collect(keys(column_data)))
    return MarginalEffectsWorkspace(model, data, continuous_vars)
end

"""
    evaluate_model_row!(workspace, observation_index; variable_overrides)

Zero-allocation model row evaluation using FormulaCompiler.
"""
function evaluate_model_row!(workspace::MarginalEffectsWorkspace, 
                            observation_index::Int; 
                            variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}())
    workspace.total_model_evaluations += 1
    
    if isempty(variable_overrides)
        modelrow!(workspace.model_row_buffer, workspace.compiled_formula, workspace.column_data, observation_index)
    else
        scenario = create_scenario("evaluation", workspace.column_data; variable_overrides...)
        modelrow!(workspace.model_row_buffer, workspace.compiled_formula, scenario.data, observation_index)
    end
    return workspace.model_row_buffer
end

"""
    evaluate_model_derivative!(workspace, observation_index, focal_variable; variable_overrides)

Zero-allocation analytical derivative evaluation with automatic ForwardDiff fallback.
"""
function evaluate_model_derivative!(workspace::MarginalEffectsWorkspace, 
                                   observation_index::Int, 
                                   focal_variable::Symbol;
                                   variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}())
    workspace.total_derivative_evaluations += 1
    
    # Try analytical derivatives first
    if haskey(workspace.derivative_formulas, focal_variable)
        try
            derivative_formula = workspace.derivative_formulas[focal_variable]
            
            if isempty(variable_overrides)
                modelrow!(workspace.derivative_buffer, derivative_formula, workspace.column_data, observation_index)
            else
                scenario = create_scenario("derivative_evaluation", workspace.column_data; variable_overrides...)
                modelrow!(workspace.derivative_buffer, derivative_formula, scenario.data, observation_index)
            end
            
            workspace.analytical_derivative_successes += 1
            return workspace.derivative_buffer
            
        catch e
            workspace.forwarddiff_fallbacks += 1
            @warn "Analytical derivative failed for $focal_variable, using ForwardDiff fallback: $e" maxlog=3
            return _evaluate_derivative_forwarddiff_fallback!(workspace, observation_index, focal_variable, variable_overrides)
        end
    else
        workspace.forwarddiff_fallbacks += 1
        return _evaluate_derivative_forwarddiff_fallback!(workspace, observation_index, focal_variable, variable_overrides)
    end
end

function _evaluate_derivative_forwarddiff_fallback!(workspace::MarginalEffectsWorkspace,
                                                   observation_index::Int,
                                                   focal_variable::Symbol,
                                                   variable_overrides::Dict{Symbol,Any})
    
    function model_eval_func(var_value::Real)
        temp_overrides = merge(variable_overrides, Dict(focal_variable => var_value))
        
        if isempty(temp_overrides)
            modelrow!(workspace.derivative_buffer, workspace.compiled_formula, workspace.column_data, observation_index)
        else
            scenario = create_scenario("forwarddiff_fallback", workspace.column_data; temp_overrides...)
            modelrow!(workspace.derivative_buffer, workspace.compiled_formula, scenario.data, observation_index)
        end
        
        return copy(workspace.derivative_buffer)
    end
    
    if haskey(variable_overrides, focal_variable)
        current_value = Float64(variable_overrides[focal_variable])
    else
        current_value = Float64(workspace.column_data[focal_variable][observation_index])
    end
    
    try
        derivative_result = ForwardDiff.derivative(model_eval_func, current_value)
        workspace.derivative_buffer .= derivative_result
        return workspace.derivative_buffer
    catch e
        @error "ForwardDiff fallback failed for $focal_variable: $e"
        fill!(workspace.derivative_buffer, NaN)
        return workspace.derivative_buffer
    end
end

# Utility functions
get_observation_count(workspace::MarginalEffectsWorkspace) = length(first(workspace.column_data))
get_parameter_count(workspace::MarginalEffectsWorkspace) = length(workspace.compiled_formula)
get_variable_names(workspace::MarginalEffectsWorkspace) = collect(keys(workspace.column_data))

function validate_data_structure(column_data::NamedTuple)
    if isempty(column_data)
        throw(ArgumentError("Data cannot be empty"))
    end
    
    reference_length = length(first(column_data))
    if reference_length == 0
        throw(ArgumentError("Data cannot have zero observations"))
    end
    
    for (variable_name, variable_values) in pairs(column_data)
        current_length = length(variable_values)
        if current_length != reference_length
            throw(DimensionMismatch(
                "Variable $variable_name has $current_length observations, " *
                "expected $reference_length observations"
            ))
        end
    end
    
    return true
end

function is_continuous_variable(variable::Symbol, column_data::NamedTuple)
    if !haskey(column_data, variable)
        return false
    end
    
    values = column_data[variable]
    element_type = eltype(values)
    
    if element_type <: Bool || values isa CategoricalArray
        return false
    end
    
    return element_type <: Real
end

function all_finite_and_reasonable(values...)
    return all(is_finite_and_reasonable, values)
end

function is_finite_and_reasonable(value::Real)
    return isfinite(value) && abs(value) < 1e10
end

function compute_standard_error_from_gradient(workspace::MarginalEffectsWorkspace,
                                            cholesky_covariance::LinearAlgebra.Cholesky)
    try
        mul!(workspace.computation_buffer, cholesky_covariance.U, workspace.gradient_accumulator)
        variance_estimate = dot(workspace.computation_buffer, workspace.computation_buffer)
        
        if variance_estimate >= 0 && isfinite(variance_estimate)
            return sqrt(variance_estimate)
        else
            @warn "Invalid variance estimate: $variance_estimate"
            return NaN
        end
    catch exception
        @warn "Standard error computation failed: $exception"
        return NaN
    end
end

"""
    get_workspace_performance_summary(workspace::MarginalEffectsWorkspace) -> NamedTuple

Get comprehensive performance summary for the workspace.
"""
function get_workspace_performance_summary(workspace::MarginalEffectsWorkspace)
    total_evaluations = workspace.total_derivative_evaluations
    analytical_rate = total_evaluations > 0 ? workspace.analytical_derivative_successes / total_evaluations : 0.0
    
    return (
        creation_time_ms = workspace.creation_time_ns / 1e6,
        total_model_evaluations = workspace.total_model_evaluations,
        total_derivative_evaluations = workspace.total_derivative_evaluations,
        analytical_successes = workspace.analytical_derivative_successes,
        forwarddiff_fallbacks = workspace.forwarddiff_fallbacks,
        analytical_success_rate = analytical_rate,
        compiled_derivatives = length(workspace.derivative_formulas),
        zero_allocation_capable = analytical_rate > 0.9,
        performance_score = _compute_performance_score(workspace)
    )
end

function _compute_performance_score(workspace::MarginalEffectsWorkspace)
    # Simple performance score based on analytical derivative usage
    total_evals = workspace.total_derivative_evaluations
    if total_evals == 0
        return 100.0  # No evaluations yet
    end
    
    analytical_rate = workspace.analytical_derivative_successes / total_evals
    return round(analytical_rate * 100, digits=1)
end

"""
    diagnose_workspace_efficiency(workspace::MarginalEffectsWorkspace) -> NamedTuple

Analyze workspace efficiency and provide recommendations.
"""
function diagnose_workspace_efficiency(workspace::MarginalEffectsWorkspace)
    summary = get_workspace_performance_summary(workspace)
    
    recommendations = String[]
    warnings = String[]
    
    # Analyze analytical derivative usage
    if summary.analytical_success_rate < 0.5
        push!(warnings, "Low analytical derivative success rate ($(round(summary.analytical_success_rate * 100, digits=1))%)")
        push!(recommendations, "Check model formula complexity - some terms may not support analytical derivatives")
    elseif summary.analytical_success_rate < 0.9
        push!(warnings, "Moderate analytical derivative usage ($(round(summary.analytical_success_rate * 100, digits=1))%)")
        push!(recommendations, "Consider simplifying model formula for better performance")
    end
    
    # Analyze compilation time
    if summary.creation_time_ms > 1000
        push!(warnings, "Slow workspace creation ($(round(summary.creation_time_ms, digits=1))ms)")
        push!(recommendations, "Consider pre-compiling workspace for repeated use")
    end
    
    # Analyze ForwardDiff fallbacks
    if summary.forwarddiff_fallbacks > summary.analytical_successes * 0.1
        push!(warnings, "High ForwardDiff fallback usage ($(summary.forwarddiff_fallbacks) calls)")
        push!(recommendations, "Review model terms for analytical derivative compatibility")
    end
    
    # Overall assessment
    efficiency_rating = if summary.performance_score >= 90
        "Excellent"
    elseif summary.performance_score >= 70
        "Good"
    elseif summary.performance_score >= 50
        "Fair"
    else
        "Poor"
    end
    
    return (
        efficiency_rating = efficiency_rating,
        performance_score = summary.performance_score,
        analytical_success_rate = summary.analytical_success_rate,
        warnings = warnings,
        recommendations = recommendations,
        summary = summary
    )
end

function Base.show(io::IO, ::MIME"text/plain", workspace::MarginalEffectsWorkspace)
    summary = get_workspace_performance_summary(workspace)
    
    println(io, "MarginalEffectsWorkspace")
    println(io, "━" ^ 40)
    println(io, "Parameters: $(length(workspace.compiled_formula))")
    println(io, "Compiled derivatives: $(summary.compiled_derivatives)")
    println(io, "Creation time: $(round(summary.creation_time_ms, digits=1))ms")
    println(io, "Performance score: $(summary.performance_score)%")
    
    if summary.total_derivative_evaluations > 0
        println(io, "Usage statistics:")
        println(io, "  Model evaluations: $(summary.total_model_evaluations)")
        println(io, "  Derivative evaluations: $(summary.total_derivative_evaluations)")
        println(io, "  Analytical success rate: $(round(summary.analytical_success_rate * 100, digits=1))%")
        println(io, "  ForwardDiff fallbacks: $(summary.forwarddiff_fallbacks)")
    end
end
