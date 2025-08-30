# gradient_collection.jl - Utilities for collecting and merging gradients from multiple computation types

# Import required types
using DataFrames
import ..AbstractMarginsGradients, ..ContinuousGradients, ..CategoricalGradients, ..PredictionGradients

"""
    GradientCollector

Utility for collecting gradients from multiple computational functions and merging them
into a single AbstractMarginsGradients object for storage in MarginsResult.
"""
mutable struct GradientCollector
    gradients::Vector{AbstractMarginsGradients}
    computation_type::Symbol  # :population or :profile
    target::Symbol           # :mu or :eta
    backend::Symbol          # :fd or :ad
end

"""
    GradientCollector(computation_type, target, backend)

Create a new gradient collector for the specified computation parameters.
"""
function GradientCollector(computation_type::Symbol, target::Symbol, backend::Symbol)
    return GradientCollector(AbstractMarginsGradients[], computation_type, target, backend)
end

"""
    add_gradients!(collector, gradients)

Add gradients from a computational function to the collector.
"""
function add_gradients!(collector::GradientCollector, gradients::AbstractMarginsGradients)
    push!(collector.gradients, gradients)
    return collector
end

"""
    merge_gradients(collector)

Merge all collected gradients into a single AbstractMarginsGradients object.
Handles different computation types (continuous, categorical, predictions) appropriately.
"""
function merge_gradients(collector::GradientCollector)
    if isempty(collector.gradients)
        error("No gradients to merge")
    end
    
    if length(collector.gradients) == 1
        # Single gradient type - return as-is
        return collector.gradients[1]
    end
    
    # Multiple gradient types - need to merge
    return _merge_multiple_gradients(collector.gradients, collector.computation_type, collector.target, collector.backend)
end

"""
    _merge_multiple_gradients(gradients_list, computation_type, target, backend)

Merge gradients from different computation types into a unified representation.
This is complex because different gradient types have different Dict key formats.
"""
function _merge_multiple_gradients(gradients_list::Vector{AbstractMarginsGradients}, computation_type::Symbol, target::Symbol, backend::Symbol)
    # Strategy: Create a MixedGradients type that can hold multiple gradient types
    # This preserves the individual gradient structures while providing unified access
    
    # Collect variables from all gradient objects
    all_variables = Symbol[]
    for grads in gradients_list
        append!(all_variables, grads.variables)
    end
    unique!(all_variables)
    
    # Collect profile coordinates (use from first non-nothing)
    profile_coords = nothing
    for grads in gradients_list
        if grads.profile_coordinates !== nothing
            profile_coords = grads.profile_coordinates
            break
        end
    end
    
    # Create a mixed gradients container
    return MixedGradients(gradients_list, computation_type, target, backend, all_variables, profile_coords)
end

"""
    MixedGradients <: AbstractMarginsGradients

Container for gradients from multiple computation types (e.g., continuous + categorical).
Preserves the individual gradient structures while providing unified access.
"""
struct MixedGradients <: AbstractMarginsGradients
    # Store individual gradient objects
    gradient_components::Vector{AbstractMarginsGradients}
    
    # Unified metadata
    computation_type::Symbol
    target::Symbol
    backend::Symbol
    variables::Vector{Symbol}
    profile_coordinates::Union{Nothing, DataFrame}
end

# Add helper function for gradient extraction from MixedGradients
function _get_gradient_dict(grads::MixedGradients)
    # Merge all gradient dicts into one
    # This is complex because different types have different key formats
    merged_dict = Dict{Any, Vector{Float64}}()
    
    for component in grads.gradient_components
        component_dict = _get_gradient_dict(component)
        for (key, value) in component_dict
            # Prefix keys with computation type to avoid conflicts
            prefixed_key = (typeof(component), key)
            merged_dict[prefixed_key] = value
        end
    end
    
    return merged_dict
end

"""
    collect_and_merge_gradients(computation_type, target, backend, gradient_objects...)

Convenience function to collect and merge gradients in one call.
"""
function collect_and_merge_gradients(computation_type::Symbol, target::Symbol, backend::Symbol, gradient_objects...)
    collector = GradientCollector(computation_type, target, backend)
    for grads in gradient_objects
        add_gradients!(collector, grads)
    end
    return merge_gradients(collector)
end