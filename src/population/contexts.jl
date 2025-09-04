# population/contexts.jl - Handle at/over parameters for population contexts (Stata-compatible)

"""
    _population_margins_with_contexts(engine, data_nt, vars, scenarios, groups; type, scale, backend)

Handle population margins with scenarios/groups contexts (unified API).

This function implements the complex logic for counterfactual scenarios (scenarios) and 
subgroup analysis (groups) in population margins computation.
"""
function _population_margins_with_contexts(engine, data_nt, vars, scenarios, groups; type, scale, backend)
    results = DataFrame()
    gradients_list = Matrix{Float64}[]
    
    # Parse specifications (unified API)
    scenario_specs = scenarios === nothing ? [Dict()] : _parse_at_specification(scenarios)
    group_specs = groups === nothing ? [Dict()] : _parse_groups_specification(groups, data_nt)
    
    # Create all combinations of contexts
    total_combinations = length(scenario_specs) * length(group_specs)
    
    # Warn about potential combination explosion
    if total_combinations > 1000
        error("Combination explosion detected ($total_combinations combinations). " *
              "This would likely exhaust system memory. " *
              "Please reduce the number of groups or scenarios. " *
              "Maximum recommended: 1000 combinations.")
    elseif total_combinations > 100
        @warn "Large number of combinations detected ($total_combinations). " *
              "This may result in slow computation and large output. " *
              "Consider reducing grouping complexity or scenario count if performance is poor."
    end
    
    for scenario_spec in scenario_specs, group_spec in group_specs
        context_data, context_indices = _create_context_data(data_nt, scenario_spec, group_spec)
        
        if type === :effects
            # Process each variable
            for var in vars
                # Skip if this var appears in scenarios/groups (conflict resolution)
                if haskey(scenario_spec, var) || haskey(group_spec, var)
                    continue
                end
                
                # Compute effect in this context
                var_result, var_gradients = _compute_population_effect_in_context(engine, context_data, context_indices, var, scale, backend)
                
                # Add context identifiers
                for (ctx_var, ctx_val) in merge(scenario_spec, group_spec)
                    # Handle continuous binning values specially
                    if ctx_val isa NamedTuple && haskey(ctx_val, :label)
                        display_val = ctx_val.label
                    else
                        display_val = ctx_val
                    end
                    var_result[!, Symbol("at_$(ctx_var)")] = fill(display_val, nrow(var_result))
                end
                
                results = _append_results_with_missing_columns(results, var_result)
                push!(gradients_list, var_gradients)
            end
        else # :predictions
            # Compute prediction in this context
            pred_result, pred_gradients = _compute_population_prediction_in_context(engine, context_data, context_indices, scale)
            
            # Add context identifiers
            for (ctx_var, ctx_val) in merge(scenario_spec, group_spec)
                # Handle continuous binning values specially
                if ctx_val isa NamedTuple && haskey(ctx_val, :label)
                    display_val = ctx_val.label
                else
                    display_val = ctx_val
                end
                pred_result[!, Symbol("at_$(ctx_var)")] = fill(display_val, nrow(pred_result))
            end
            
            results = _append_results_with_missing_columns(results, pred_result)
            push!(gradients_list, pred_gradients)
        end
    end
    
    # Combine gradients
    G = isempty(gradients_list) ? Matrix{Float64}(undef, 0, length(engine.β)) : vcat(gradients_list...)
    
    # Build metadata
    metadata = _build_metadata(; type, vars, scale, backend, n_obs=length(first(data_nt)), 
                              model_type=typeof(engine.model), has_contexts=true)
    
    # Add analysis_type for format auto-detection
    metadata[:analysis_type] = :population  # This is still population-level, just with contexts
    
    # Extract raw components from DataFrame
    estimates = results.estimate
    standard_errors = results.se
    terms = string.(results.term)  # Convert Symbol to String
    
    # CRITICAL: Preserve actual subgroup sizes from computation
    if "n" in names(results)
        # Store the actual subgroup n values in metadata
        metadata[:subgroup_n_values] = results.n
        metadata[:has_subgroup_n] = true
    end
    
    # Extract profile values from at_ columns (if any)
    profile_values = _extract_context_profile_values(results)
    
    return MarginsResult(estimates, standard_errors, terms, profile_values, nothing, G, metadata)
end

"""
    _extract_context_profile_values(results) -> Union{Nothing, NamedTuple}

Extract profile values from at_ columns in results DataFrame.
Returns Nothing if no at_ columns exist.
"""
function _extract_context_profile_values(results)
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
        # Mixed vector syntax - handle both categorical and continuous variables
        contexts = [Dict()]
        for var in over
            if _is_continuous_variable(data_nt[var])
                # For continuous variables in vector syntax, use automatic quartile binning
                quantile_groups = _create_quantile_groups(data_nt[var], var, 4)  # Default to quartiles
                new_contexts = []
                for ctx in contexts, qg in quantile_groups
                    push!(new_contexts, merge(ctx, qg))
                end
                contexts = new_contexts
            else
                # For categorical variables, use all unique levels
                new_contexts = []
                for ctx in contexts, val in unique(data_nt[var])
                    push!(new_contexts, merge(ctx, Dict(var => val)))
                end
                contexts = new_contexts
            end
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
    _parse_groups_specification(groups, data_nt) -> Vector{Dict}

Parse unified groups specification for stratified analysis.
Supports the unified grouping syntax from POP_GROUPING.md Phase 3.
"""
function _parse_groups_specification(groups, data_nt)
    # Simple categorical grouping: :education
    if groups isa Symbol
        return _parse_over_specification([groups], data_nt)
    end
    
    # Vector grouping: may be all symbols or mixed specs  
    if groups isa AbstractVector
        if all(x -> x isa Symbol, groups)
            # All symbols - simple categorical cross-tabulation
            return _parse_over_specification(groups, data_nt)
        else
            # Mixed vector - handle each spec and create cross-product
            all_groups = []
            for spec in groups
                spec_groups = _parse_groups_specification(spec, data_nt)
                if isempty(all_groups)
                    all_groups = spec_groups
                else
                    # Cross-product with existing groups
                    new_groups = []
                    for existing_group in all_groups
                        for spec_group in spec_groups
                            combined_group = merge(existing_group, spec_group)
                            push!(new_groups, combined_group)
                        end
                    end
                    all_groups = new_groups
                end
            end
            return all_groups
        end
    end
    
    # Continuous grouping: (:income, 4) or (:age, [25000, 50000])
    if groups isa Tuple && length(groups) == 2
        var, spec = groups
        if var isa Symbol
            # Quantile specification: (:income, 4)
            if spec isa Integer && spec > 0
                return _create_quantile_groups(data_nt[var], var, spec)
            end
            # Threshold specification: (:income, [25000, 50000])
            if spec isa AbstractVector && all(x -> x isa Real, spec)
                return _create_threshold_groups(data_nt[var], var, spec)
            end
        end
    end
    
    # Phase 3: => syntax for nested grouping: :region => :education
    if groups isa Pair
        outer_spec = groups.first
        inner_spec = groups.second
        return _create_nested_strata(outer_spec, inner_spec, data_nt)
    end
    
    error("Invalid groups specification. Supported syntax: Symbol, Vector{Symbol}, (Symbol, Int), (Symbol, Vector), or outer => inner")
end

"""
    _create_nested_strata(outer_spec, inner_spec, data_nt) -> Vector{Dict}

Phase 3: Create nested strata using => syntax.
Implements outer => inner pattern: outer first, then inner within each outer.
Supports inner_spec as Vector for parallel inner groupings.
"""
function _create_nested_strata(outer_spec, inner_spec, data_nt)
    # Parse outer specification
    outer_groups = _parse_groups_specification(outer_spec, data_nt)
    
    # Handle inner specification - can be single spec or Vector of specs
    if inner_spec isa AbstractVector
        # Multiple inner specifications - create parallel groups within each outer
        nested_groups = []
        for outer_group in outer_groups
            for inner_single_spec in inner_spec
                inner_groups = _parse_groups_specification(inner_single_spec, data_nt)
                for inner_group in inner_groups
                    # Merge outer and inner groups
                    combined_group = merge(outer_group, inner_group)
                    push!(nested_groups, combined_group)
                end
            end
        end
        return nested_groups
    else
        # Single inner specification
        inner_groups = _parse_groups_specification(inner_spec, data_nt)
        
        # Create nested combinations: outer => inner means inner within each outer
        nested_groups = []
        for outer_group in outer_groups
            for inner_group in inner_groups
                # Merge outer and inner groups
                combined_group = merge(outer_group, inner_group)
                push!(nested_groups, combined_group)
            end
        end
        
        return nested_groups
    end
end

"""
    _create_quantile_groups(col, var, n_quantiles) -> Vector{Dict}

Create groups based on quantiles of a continuous variable.
Phase 3: Enhanced with proper quartile/tertile labeling.
"""
function _create_quantile_groups(col, var, n_quantiles)
    quantiles = [quantile(col, i/n_quantiles) for i in 0:n_quantiles]
    groups = []
    for i in 1:(n_quantiles)
        lower, upper = quantiles[i], quantiles[i+1]
        
        # Generate descriptive labels based on n_quantiles
        if n_quantiles == 4
            label = ["Q1", "Q2", "Q3", "Q4"][i]
        elseif n_quantiles == 3
            label = ["T1", "T2", "T3"][i]
        elseif n_quantiles == 5
            label = ["P1", "P2", "P3", "P4", "P5"][i]  # Quintiles
        else
            label = "Bin$i"
        end
        
        # Create group with range information for filtering
        group = Dict(
            var => (lower=lower, upper=upper, label=label, bin=i),
            Symbol("$(var)_bin") => i
        )
        push!(groups, group)
    end
    return groups
end

"""
    _create_threshold_groups(col, var, thresholds) -> Vector{Dict}

Create groups based on specified threshold values.
Phase 3: Enhanced with proper range-based grouping.
"""
function _create_threshold_groups(col, var, thresholds)
    groups = []
    n_groups = length(thresholds) + 1
    
    for i in 1:n_groups
        if i == 1
            # First group: [min, threshold1)
            lower = minimum(col)
            upper = thresholds[1]
            label = "< $(thresholds[1])"
        elseif i == n_groups
            # Last group: [threshold_last, max]
            lower = thresholds[end]
            upper = maximum(col)
            label = ">= $(thresholds[end])"
        else
            # Middle groups: [threshold_{i-1}, threshold_i)
            lower = thresholds[i-1]
            upper = thresholds[i]
            label = "[$(lower), $(upper))"
        end
        
        group = Dict(
            var => (lower=lower, upper=upper, label=label, bin=i),
            Symbol("$(var)_bin") => i
        )
        push!(groups, group)
    end
    
    return groups
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
    _create_context_data(data_nt, at_spec, over_spec) -> (NamedTuple, Vector{Int})

Create context data (counterfactual overrides + subgroup filtering).
Phase 3: Enhanced for continuous binning with range-based filtering.
Returns the filtered data and the indices of the original rows that were kept.
"""
function _create_context_data(data_nt, at_spec, over_spec)
    # Start with full data
    context_data = data_nt  # Don't deepcopy for performance
    
    # Apply counterfactual overrides (at/scenarios)
    for (var, val) in at_spec
        if haskey(context_data, var)
            # Override all values with specified value
            n_rows = length(first(context_data))
            override_col = fill(val, n_rows)
            context_data = merge(context_data, NamedTuple{(var,)}((override_col,)))
        end
    end
    
    # Apply subgroup filtering (groups)
    indices_to_keep = collect(1:length(first(context_data)))
    for (var, spec) in over_spec
        if haskey(context_data, var)
            if spec isa NamedTuple && haskey(spec, :lower) && haskey(spec, :upper)
                # Phase 3: Range-based continuous filtering
                col = context_data[var]
                if haskey(spec, :bin) && spec.bin == 1
                    # First bin: include lower boundary
                    range_indices = findall(x -> spec.lower <= x < spec.upper, col)
                elseif haskey(spec, :bin) && spec.bin > 1
                    # Other bins: exclude lower boundary to avoid overlap
                    range_indices = findall(x -> spec.lower < x <= spec.upper, col)
                else
                    # Fallback: include both boundaries
                    range_indices = findall(x -> spec.lower <= x <= spec.upper, col)
                end
                indices_to_keep = intersect(indices_to_keep, range_indices)
            else
                # Categorical - filter by value
                var_indices = findall(==(spec), context_data[var])
                indices_to_keep = intersect(indices_to_keep, var_indices)
            end
        elseif occursin("_bin", String(var)) && haskey(over_spec, var)
            # Skip bin columns - they're handled by their parent variable
            continue
        end
    end
    
    # Subset context_data to selected indices
    if length(indices_to_keep) < length(first(context_data))
        subset_data = NamedTuple()
        for (var, col) in pairs(context_data)
            subset_data = merge(subset_data, NamedTuple{(var,)}((col[indices_to_keep],)))
        end
        return subset_data, indices_to_keep
    end
    
    return context_data, indices_to_keep
end

"""
    _compute_population_effect_in_context(engine, context_data, context_indices, var, scale, backend) -> (DataFrame, Matrix{Float64})

Compute population marginal effect for a single variable in a specific context.
"""
function _compute_population_effect_in_context(engine::MarginsEngine{L}, context_data::NamedTuple, context_indices::Vector{Int}, var::Symbol, scale::Symbol, backend::Symbol) where L
    # Create a temporary engine for this single variable to reuse existing AME infrastructure
    n_obs = length(first(context_data))
    n_params = length(engine.β)
    
    # CRITICAL: Check for empty subgroups - violates statistical correctness principles
    if n_obs == 0
        error("Cannot compute marginal effects for empty subgroup. " *
              "This violates statistical correctness principles - " *
              "marginal effects require at least one observation. " *
              "Consider using coarser grouping or removing empty categories.")
    end
    
    # For single variable, we need to compute the AME for just this var
    # First check if this variable is continuous or categorical
    col = context_data[var]
    is_categorical = eltype(col) <: Bool || eltype(col) <: CategoricalValue
    
    if is_categorical
        # Use existing categorical AME computation
        ame_val, gβ_avg = _compute_categorical_baseline_ame_context(engine, var, context_data, context_indices, scale, backend)
    else
        # Use existing continuous AME computation for single variable
        ame_val, gβ_avg = _compute_continuous_ame_context(engine, var, context_data, context_indices, scale, backend)
    end
    
    # Compute delta-method SE
    se = sqrt(dot(gβ_avg, engine.Σ, gβ_avg))
    
    # Create results DataFrame
    df = DataFrame(
        term = [string(var)],
        estimate = [ame_val],
        se = [se],
        t_stat = [ame_val / se],
        p_value = [2 * (1 - cdf(Normal(), abs(ame_val / se)))],
        n = [n_obs]  # Add sample size for this subgroup
    )
    
    # Reshape gradient to matrix form (1×p)
    G = reshape(gβ_avg, 1, length(gβ_avg))
    
    return df, G
end

"""
    _compute_continuous_ame_context(engine, var, context_data, context_indices, scale, backend) -> (Float64, Vector{Float64})

Compute continuous AME for a single variable in context using FormulaCompiler.
"""
function _compute_continuous_ame_context(engine::MarginsEngine{L}, var::Symbol, context_data::NamedTuple, context_indices::Vector{Int}, scale::Symbol, backend::Symbol) where L
    # Get variable index for AME computation
    var_idx = findfirst(v -> v == var, engine.de.vars)
    if var_idx === nothing
        throw(ArgumentError("Variable $var not found in derivatives engine"))
    end
    
    # Use the context_indices to compute AME only for the filtered subgroup
    rows = context_indices  # Now we use only the rows for this specific group!
    
    n_params = length(engine.β)
    
    # Clear accumulator buffer
    fill!(engine.gβ_accumulator, 0.0)
    
    # Use FormulaCompiler to compute AME gradient with original data and engine
    FormulaCompiler.accumulate_ame_gradient!(
        engine.gβ_accumulator, engine.de, engine.β, rows, var;
        link=(scale === :response ? engine.link : GLM.IdentityLink()), 
        backend=backend
    )
    
    # The gradient is already averaged by accumulate_ame_gradient!
    gβ_avg = copy(engine.gβ_accumulator)  # Copy to avoid mutation
    
    # Compute AME value
    ame_val = _accumulate_me_value(engine.g_buf, engine.de, engine.β, engine.link, rows, scale, backend, var_idx)
    ame_val /= length(rows)  # Average across context observations
    
    return (ame_val, gβ_avg)
end

"""
    _compute_categorical_baseline_ame_context(engine, var, context_data, context_indices, scale, backend) -> (Float64, Vector{Float64})

Compute categorical baseline AME for a single variable in context using FormulaCompiler.
"""
function _compute_categorical_baseline_ame_context(engine::MarginsEngine{L}, var::Symbol, context_data::NamedTuple, context_indices::Vector{Int}, scale::Symbol, backend::Symbol) where L
    # For now, delegate to continuous implementation (placeholder)
    # TODO: Implement proper categorical contrast computation
    return _compute_continuous_ame_context(engine, var, context_data, context_indices, scale, backend)
end

"""
    _compute_population_prediction_in_context(engine, context_data, context_indices, scale) -> (DataFrame, Matrix{Float64})

Compute population average prediction in a specific context.
"""
function _compute_population_prediction_in_context(engine::MarginsEngine{L}, context_data::NamedTuple, context_indices::Vector{Int}, scale::Symbol) where L
    # Use the same logic as _population_predictions but with context_data
    n_obs = length(first(context_data))
    n_params = length(engine.β)
    
    # Use pre-allocated η_buf as working buffer; size to n_obs
    work = view(engine.η_buf, 1:n_obs)
    G = zeros(1, n_params)  # Single row for population average
    
    # Delegate hot loop to the same helper but with context data
    mean_prediction = _compute_population_predictions!(G, work, engine, context_data, scale, nothing)
    
    # Delta-method SE (G is 1×p, Σ is p×p)
    se = sqrt((G * engine.Σ * G')[1, 1])
    
    # Create results DataFrame
    df = DataFrame(
        term = ["AAP"],
        estimate = [mean_prediction],
        se = [se],
        t_stat = [mean_prediction / se],
        p_value = [2 * (1 - cdf(Normal(), abs(mean_prediction / se)))]
    )
    
    return df, G
end

"""
    _append_results_with_missing_columns(results, new_result) -> DataFrame

Helper function to append DataFrames that may have different column structures.
This is needed for complex parallel grouping where different group specifications
create results with different at_ columns.
"""
function _append_results_with_missing_columns(results::DataFrame, new_result::DataFrame)
    if nrow(results) == 0
        return new_result
    end
    
    # Simple approach: use vcat with cols=:union to let DataFrames handle missing columns
    try
        return vcat(results, new_result; cols=:union)
    catch e
        # Fallback: manual column alignment with string-based missing values
        all_cols = union(names(results), names(new_result))
        
        # Ensure all columns exist in both DataFrames, using string "missing" for consistency
        for col in all_cols
            if !(col in names(results))
                results[!, col] = fill("missing", nrow(results))
            end
        end
        
        new_result_copy = copy(new_result)
        for col in all_cols
            if !(col in names(new_result_copy))
                new_result_copy[!, col] = fill("missing", nrow(new_result_copy))
            end
        end
        
        # Reorder columns to match
        results = results[!, all_cols]
        new_result_copy = new_result_copy[!, all_cols]
        
        # Now append
        append!(results, new_result_copy)
        return results
    end
end