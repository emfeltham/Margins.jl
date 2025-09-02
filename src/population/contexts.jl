# population/contexts.jl - Handle at/over parameters for population contexts (Stata-compatible)

"""
    _population_margins_with_contexts(engine, data_nt, vars, at, over; type, target, backend, kwargs...)

Handle population margins with at/over contexts (Stata-compatible).

This function implements the complex logic for counterfactual scenarios (at) and 
subgroup analysis (over) in population margins computation.
"""
function _population_margins_with_contexts(engine, data_nt, vars, at, over; type, target, backend, kwargs...)
    results = DataFrame()
    gradients_list = Matrix{Float64}[]
    
    # Parse specifications
    at_specs = at === nothing ? [Dict()] : _parse_at_specification(at)
    over_specs = over === nothing ? [Dict()] : _parse_over_specification(over, data_nt)
    
    # Create all combinations of contexts
    for at_spec in at_specs, over_spec in over_specs
        context_data = _create_context_data(data_nt, at_spec, over_spec)
        
        if type === :effects
            # Process each variable
            for var in vars
                # Skip if this var appears in at/over (conflict resolution)
                if haskey(at_spec, var) || haskey(over_spec, var)
                    continue
                end
                
                # Compute effect in this context
                var_result, var_gradients = _compute_population_effect_in_context(engine, context_data, var, target, backend)
                
                # Add context identifiers
                for (ctx_var, ctx_val) in merge(at_spec, over_spec)
                    var_result[!, Symbol("at_$(ctx_var)")] = ctx_val
                end
                
                append!(results, var_result)
                push!(gradients_list, var_gradients)
            end
        else # :predictions
            # Compute prediction in this context
            pred_result, pred_gradients = _compute_population_prediction_in_context(engine, context_data, target)
            
            # Add context identifiers
            for (ctx_var, ctx_val) in merge(at_spec, over_spec)
                pred_result[!, Symbol("at_$(ctx_var)")] = ctx_val
            end
            
            append!(results, pred_result)
            push!(gradients_list, pred_gradients)
        end
    end
    
    # Combine gradients
    G = isempty(gradients_list) ? Matrix{Float64}(undef, 0, length(engine.β)) : vcat(gradients_list...)
    
    # Build metadata
    metadata = _build_metadata(; type, vars, target, backend, n_obs=length(first(data_nt)), 
                              model_type=typeof(engine.model), has_contexts=true, kwargs...)
    
    # Add analysis_type for format auto-detection
    metadata[:analysis_type] = :population  # This is still population-level, just with contexts
    
    # Extract raw components from DataFrame
    estimates = results.estimate
    standard_errors = results.se
    terms = results.term
    
    # Extract profile values from at_ columns (if any)
    profile_values = _extract_context_profile_values(results)
    
    return MarginsResult(estimates, standard_errors, terms, profile_values, nothing, G, metadata)
end

"""
    _extract_context_profile_values(results::DataFrame) -> Union{Nothing, NamedTuple}

Extract profile values from at_ columns in results DataFrame.
Returns Nothing if no at_ columns exist.
"""
function _extract_context_profile_values(results::DataFrame)
    at_cols = [col for col in names(results) if startswith(String(col), "at_")]
    
    if isempty(at_cols)
        return nothing
    end
    
    # Extract the profile data, removing the "at_" prefix
    profile_dict = Dict{Symbol, Vector}()
    for col in at_cols
        var_name = Symbol(String(col)[4:end])  # Remove "at_" prefix
        profile_dict[var_name] = results[!, col]
    end
    
    return NamedTuple(profile_dict)
end

"""
    _parse_at_specification(at) -> Vector{Dict}

Parse at specification (counterfactual scenarios).
Creates all combinations of at values for Cartesian product expansion.
"""
function _parse_at_specification(at)
    if at isa Dict
        # Create all combinations of at values
        var_names = collect(keys(at))
        var_values = [at[k] isa Vector ? at[k] : [at[k]] for k in var_names]  # Ensure vectors
        contexts = []
        for combo in Iterators.product(var_values...)
            context = Dict(zip(var_names, combo))
            push!(contexts, context)
        end
        return contexts
    else
        error("at parameter must be a Dict specifying variable values")
    end
end

"""
    _parse_over_specification(over, data_nt) -> Vector{Dict}

Parse over specification (subgroup analysis).
Supports both NamedTuple and Vector syntax with flexible value specification.
"""
function _parse_over_specification(over, data_nt)
    if over isa NamedTuple
        # Enhanced flexible syntax
        contexts = []
        
        for (var, vals) in pairs(over)
            if vals === nothing
                # Unspecified - must be categorical, use all levels
                if _is_continuous_variable(data_nt[var])
                    error("Continuous variable $var in over() must specify values")
                end
                contexts = isempty(contexts) ? [Dict(var => v) for v in unique(data_nt[var])] : 
                          [merge(ctx, Dict(var => v)) for ctx in contexts for v in unique(data_nt[var])]
            else
                # Specified values
                if _is_continuous_variable(data_nt[var])
                    # For continuous: create subgroups around specified values
                    subgroups = _create_continuous_subgroups(data_nt[var], vals)
                    contexts = isempty(contexts) ? [Dict(var => sg) for sg in subgroups] :
                              [merge(ctx, Dict(var => sg)) for ctx in contexts for sg in subgroups]
                else
                    # For categorical: use specified subset
                    val_list = vals isa Vector ? vals : [vals]
                    contexts = isempty(contexts) ? [Dict(var => v) for v in val_list] :
                              [merge(ctx, Dict(var => v)) for ctx in contexts for v in val_list]
                end
            end
        end
        
        return contexts
    elseif over isa Vector
        # Simple vector syntax - all categorical
        contexts = [Dict()]
        for var in over
            if _is_continuous_variable(data_nt[var])
                error("Continuous variable $var in over() must specify values. Use over=($var => [values])")
            end
            new_contexts = []
            for ctx in contexts, val in unique(data_nt[var])
                push!(new_contexts, merge(ctx, Dict(var => val)))
            end
            contexts = new_contexts
        end
        return contexts
    elseif over isa Symbol
        # Single variable syntax - convert to vector and recurse
        return _parse_over_specification([over], data_nt)
    else
        error("over parameter must be a NamedTuple, Vector, or Symbol")
    end
end

"""
    _create_continuous_subgroups(col, specified_values) -> Vector

Create subgroups around specified continuous values.
Uses ±2.5 unit ranges for subgroup definition.
"""
function _create_continuous_subgroups(col, specified_values)
    val_list = specified_values isa Vector ? specified_values : [specified_values]
    subgroups = []
    for val in val_list
        # Create subgroup of indices within range of this value
        indices = findall(x -> abs(x - val) <= 2.5, col)  # ±2.5 units
        push!(subgroups, (center=val, indices=indices, label="$(val)±2.5"))
    end
    return subgroups
end

"""
    _create_context_data(data_nt, at_spec, over_spec) -> NamedTuple

Create context data (counterfactual overrides + subgroup filtering).
"""
function _create_context_data(data_nt, at_spec, over_spec)
    # Start with full data
    context_data = data_nt  # Don't deepcopy for performance
    
    # Apply counterfactual overrides (at)
    for (var, val) in at_spec
        if haskey(context_data, var)
            # Override all values with specified value
            n_rows = length(first(context_data))
            override_col = fill(val, n_rows)
            context_data = merge(context_data, NamedTuple{(var,)}((override_col,)))
        end
    end
    
    # Apply subgroup filtering (over)
    indices_to_keep = collect(1:length(first(context_data)))
    for (var, spec) in over_spec
        if haskey(context_data, var)
            if spec isa NamedTuple && haskey(spec, :indices)
                # Continuous subgroup - intersect with these indices
                indices_to_keep = intersect(indices_to_keep, spec.indices)
            else
                # Categorical - filter by value
                var_indices = findall(==(spec), context_data[var])
                indices_to_keep = intersect(indices_to_keep, var_indices)
            end
        end
    end
    
    # Subset context_data to selected indices
    if length(indices_to_keep) < length(first(context_data))
        subset_data = NamedTuple()
        for (var, col) in pairs(context_data)
            subset_data = merge(subset_data, NamedTuple{(var,)}((col[indices_to_keep],)))
        end
        return subset_data
    end
    
    return context_data
end

"""
    _compute_population_effect_in_context(engine, context_data, var, target, backend) -> (DataFrame, Matrix{Float64})

Compute population marginal effect for a single variable in a specific context.
"""
function _compute_population_effect_in_context(engine::MarginsEngine, context_data::NamedTuple, var::Symbol, target::Symbol, backend::Symbol)
    # Use existing AME computation but with context data
    n_obs = length(first(context_data))
    
    # For now, return a simple result - this should delegate to FormulaCompiler
    df = DataFrame(
        term = [string(var)],
        estimate = [0.0],  # Placeholder
        se = [0.0],        # Placeholder
        t_stat = [0.0],
        p_value = [1.0]
    )
    
    G = zeros(1, length(engine.β))  # Placeholder gradient
    
    return df, G
end

"""
    _compute_population_prediction_in_context(engine, context_data, target) -> (DataFrame, Matrix{Float64})

Compute population average prediction in a specific context.
"""
function _compute_population_prediction_in_context(engine::MarginsEngine, context_data::NamedTuple, target::Symbol)
    # Use existing prediction computation but with context data
    n_obs = length(first(context_data))
    
    # For now, return a simple result
    df = DataFrame(
        term = ["AAP"],
        estimate = [0.0],  # Placeholder
        se = [0.0],        # Placeholder
        t_stat = [0.0],
        p_value = [1.0]
    )
    
    G = zeros(1, length(engine.β))  # Placeholder gradient
    
    return df, G
end