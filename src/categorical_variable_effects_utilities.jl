# categorical_variable_effects_extra.jl

###############################################################################
# Factor Level Utilities
###############################################################################

"""
    validate_categorical_variable_structure(focal_variable::Symbol, 
                                           column_data::NamedTuple) -> Bool

Validate that a variable is appropriate for categorical marginal effects computation.
"""
function validate_categorical_variable_structure(focal_variable::Symbol, column_data::NamedTuple)
    if !haskey(column_data, focal_variable)
        throw(ArgumentError("Variable $focal_variable not found in data"))
    end
    
    values = column_data[focal_variable]
    unique_values = unique(values)
    unique_count = length(unique_values)
    total_count = length(values)
    
    # Check for appropriate number of levels
    if unique_count < 2
        throw(ArgumentError("Variable $focal_variable has fewer than 2 unique values"))
    elseif unique_count == total_count
        @warn "Variable $focal_variable has all unique values - consider treating as continuous"
    elseif unique_count > 50
        @warn "Variable $focal_variable has $unique_count levels - consider grouping levels for computational efficiency"
    end
    
    return true
end

###############################################################################
# Diagnostics and Performance Monitoring
###############################################################################

"""
    CategoricalEffectDiagnostics

Diagnostic information for categorical variable marginal effect computation.
"""
struct CategoricalEffectDiagnostics
    focal_variable::Symbol
    factor_levels::Vector
    level_frequencies::Dict
    observation_count::Int
    parameter_count::Int
    contrast_count::Int
    sample_contrasts::Vector{NamedTuple}
    variable_overrides::Dict{Symbol,Any}
    computation_warnings::Vector{String}
end

"""
    diagnose_categorical_effect_computation(focal_variable::Symbol, 
                                           workspace::MarginalEffectsWorkspace,
                                           coefficient_vector::AbstractVector,
                                           covariance_matrix::AbstractMatrix,
                                           inverse_link_function::Function,
                                           first_derivative::Function;
                                           variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}(),
                                           sample_contrast_count::Int = 3) -> CategoricalEffectDiagnostics

Diagnostic analysis of categorical variable marginal effect computation.
"""
function diagnose_categorical_effect_computation(
    focal_variable::Symbol, 
    workspace::MarginalEffectsWorkspace,
    coefficient_vector::AbstractVector,
    covariance_matrix::AbstractMatrix,
    inverse_link_function::Function,
    first_derivative::Function;
    variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}(),
    sample_contrast_count::Int = 3
)
    
    observation_count = get_observation_count(workspace)
    parameter_count = get_parameter_count(workspace)
    warnings = String[]
    
    # Extract factor information
    factor_column = workspace.column_data[focal_variable]
    factor_levels = extract_factor_levels(factor_column)
    
    # Compute level frequencies
    level_frequencies = Dict()
    for level in factor_levels
        count = sum(==(level), factor_column)
        level_frequencies[level] = count
    end
    
    # Sample a few contrasts for detailed analysis
    sample_contrasts = NamedTuple[]
    contrast_pairs = [(factor_levels[1], factor_levels[2])]  # Start with first pair
    
    # Add more pairs if available
    if length(factor_levels) > 2
        push!(contrast_pairs, (factor_levels[1], factor_levels[3]))
    end
    if length(factor_levels) > 3
        push!(contrast_pairs, (factor_levels[2], factor_levels[3]))
    end
    
    # Limit to requested sample count
    contrast_pairs = contrast_pairs[1:min(sample_contrast_count, length(contrast_pairs))]
    
    for (ref_level, comp_level) in contrast_pairs
        try
            # Compute sample contrast
            effect, se, gradient = compute_factor_level_contrast(
                focal_variable, ref_level, comp_level,
                workspace, coefficient_vector, covariance_matrix,
                inverse_link_function, first_derivative;
                variable_overrides=variable_overrides
            )
            
            # Sample a few observations to show prediction differences
            sample_predictions = NamedTuple[]
            for obs_idx in 1:min(3, observation_count)
                # Prediction at reference level
                ref_overrides = merge(variable_overrides, Dict(focal_variable => ref_level))
                evaluate_model_row!(workspace, obs_idx; variable_overrides=ref_overrides)
                ref_eta = dot(workspace.model_row_buffer, coefficient_vector)
                ref_pred = inverse_link_function(ref_eta)
                
                # Prediction at comparison level
                comp_overrides = merge(variable_overrides, Dict(focal_variable => comp_level))
                evaluate_model_row!(workspace, obs_idx; variable_overrides=comp_overrides)
                comp_eta = dot(workspace.model_row_buffer, coefficient_vector)
                comp_pred = inverse_link_function(comp_eta)
                
                push!(sample_predictions, (
                    observation = obs_idx,
                    reference_prediction = ref_pred,
                    comparison_prediction = comp_pred,
                    difference = comp_pred - ref_pred,
                    original_level = factor_column[obs_idx]
                ))
            end
            
            push!(sample_contrasts, (
                reference_level = ref_level,
                comparison_level = comp_level,
                marginal_effect = effect,
                standard_error = se,
                gradient_norm = norm(gradient),
                sample_predictions = sample_predictions
            ))
            
        catch exception
            push!(warnings, "Contrast ($ref_level, $comp_level) failed: $exception")
        end
    end
    
    total_contrasts = length(factor_levels) * (length(factor_levels) - 1) ÷ 2
    
    return CategoricalEffectDiagnostics(
        focal_variable,
        factor_levels,
        level_frequencies,
        observation_count,
        parameter_count,
        total_contrasts,
        sample_contrasts,
        variable_overrides,
        warnings
    )
end

"""
    benchmark_categorical_effect_computation(focal_variable::Symbol,
                                            workspace::MarginalEffectsWorkspace,
                                            coefficient_vector::AbstractVector,
                                            covariance_matrix::AbstractMatrix,
                                            inverse_link_function::Function,
                                            first_derivative::Function;
                                            variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}(),
                                            benchmark_samples::Int = 5) -> NamedTuple

Benchmark performance of categorical variable marginal effect computation.
"""
function benchmark_categorical_effect_computation(
    focal_variable::Symbol,
    workspace::MarginalEffectsWorkspace,
    coefficient_vector::AbstractVector,
    covariance_matrix::AbstractMatrix,
    inverse_link_function::Function,
    first_derivative::Function;
    variable_overrides::Dict{Symbol,Any} = Dict{Symbol,Any}(),
    benchmark_samples::Int = 5
)
    
    observation_count = get_observation_count(workspace)
    parameter_count = get_parameter_count(workspace)
    
    # Extract factor information
    factor_levels = extract_factor_levels(workspace.column_data[focal_variable])
    level_count = length(factor_levels)
    
    if level_count < 2
        return (error = "Variable $focal_variable has fewer than 2 levels",)
    end
    
    # Benchmark single contrast computation
    reference_level = factor_levels[1]
    comparison_level = factor_levels[2]
    
    contrast_times = Float64[]
    allocation_counts = Int[]
    
    # Warm-up runs
    for _ in 1:3
        compute_factor_level_contrast(
            focal_variable, reference_level, comparison_level,
            workspace, coefficient_vector, covariance_matrix,
            inverse_link_function, first_derivative;
            variable_overrides=variable_overrides
        )
    end
    
    # Actual benchmark runs
    for _ in 1:benchmark_samples
        timing_result = @timed compute_factor_level_contrast(
            focal_variable, reference_level, comparison_level,
            workspace, coefficient_vector, covariance_matrix,
            inverse_link_function, first_derivative;
            variable_overrides=variable_overrides
        )
        push!(contrast_times, timing_result.time)
        push!(allocation_counts, timing_result.bytes)
    end
    
    # Estimate total computation time for all contrasts
    total_contrasts = level_count * (level_count - 1) ÷ 2  # All pairs
    estimated_total_time = mean(contrast_times) * total_contrasts
    
    return (
        focal_variable = focal_variable,
        observations = observation_count,
        parameters = parameter_count,
        factor_levels = level_count,
        total_contrasts = total_contrasts,
        benchmark_samples = benchmark_samples,
        
        # Single contrast timing
        mean_contrast_time_seconds = mean(contrast_times),
        minimum_contrast_time_seconds = minimum(contrast_times),
        time_per_observation_nanoseconds = mean(contrast_times) * 1e9 / observation_count,
        
        # Allocation statistics
        mean_allocations_bytes = mean(allocation_counts),
        minimum_allocations_bytes = minimum(allocation_counts),
        zero_allocation_percentage = 100 * count(==(0), allocation_counts) / length(allocation_counts),
        
        # Total computation estimates
        estimated_total_computation_seconds = estimated_total_time,
        contrasts_per_second = 1.0 / mean(contrast_times),
        
        # Memory efficiency
        workspace_memory_per_contrast = (sizeof(workspace.model_row_buffer) + 
                                        sizeof(workspace.derivative_buffer) + 
                                        sizeof(workspace.gradient_accumulator) + 
                                        sizeof(workspace.computation_buffer)),
        
        # Representative values info
        override_count = length(variable_overrides)
    )
end

###############################################################################
# Pretty Printing for Diagnostics
###############################################################################

function Base.show(io::IO, diag::CategoricalEffectDiagnostics)
    println(io, "CategoricalEffectDiagnostics for $(diag.focal_variable)")
    println(io, "━" ^ 50)
    println(io, "Observations: $(diag.observation_count)")
    println(io, "Parameters: $(diag.parameter_count)")
    println(io, "Factor levels: $(length(diag.factor_levels))")
    println(io, "Total contrasts: $(diag.contrast_count)")
    
    if !isempty(diag.variable_overrides)
        println(io, "Variable overrides: $(diag.variable_overrides)")
    end
    
    println(io, "\nLevel frequencies:")
    for (level, freq) in diag.level_frequencies
        percentage = round(100 * freq / diag.observation_count, digits=1)
        println(io, "  $level: $freq ($percentage%)")
    end
    
    if !isempty(diag.sample_contrasts)
        println(io, "\nSample contrasts:")
        for (i, contrast) in enumerate(diag.sample_contrasts)
            println(io, "  $i. $(contrast.reference_level) → $(contrast.comparison_level):")
            println(io, "     Effect: $(round(contrast.marginal_effect, digits=4))")
            println(io, "     SE: $(round(contrast.standard_error, digits=4))")
            println(io, "     Gradient norm: $(round(contrast.gradient_norm, digits=4))")
        end
    end
    
    if !isempty(diag.computation_warnings)
        println(io, "\nWarnings:")
        for warning in diag.computation_warnings
            println(io, "  ⚠ $warning")
        end
    end
end

###############################################################################
# Additional Utility Functions for Categorical Effects
###############################################################################

"""
    estimate_categorical_computation_complexity(focal_variable::Symbol, 
                                               workspace::MarginalEffectsWorkspace,
                                               factor_contrasts::Symbol) -> NamedTuple

Estimate computational complexity for categorical variable marginal effects.
"""
function estimate_categorical_computation_complexity(
    focal_variable::Symbol, 
    workspace::MarginalEffectsWorkspace,
    factor_contrasts::Symbol
)
    
    factor_levels = extract_factor_levels(workspace.column_data[focal_variable])
    level_count = length(factor_levels)
    observation_count = get_observation_count(workspace)
    parameter_count = get_parameter_count(workspace)
    
    # Calculate number of contrasts
    contrast_count = if factor_contrasts == :baseline_contrasts
        max(0, level_count - 1)  # All levels vs first level
    else  # :all_pairs
        level_count * (level_count - 1) ÷ 2  # All pairwise combinations
    end
    
    # Estimate operations
    # Each contrast requires 2n model evaluations (n for each level)
    # Plus gradient computations and linear algebra
    model_evaluations = contrast_count * 2 * observation_count
    gradient_operations = contrast_count * observation_count * parameter_count
    linear_algebra_operations = contrast_count * parameter_count^2
    
    return (
        focal_variable = focal_variable,
        factor_levels = level_count,
        observations = observation_count,
        parameters = parameter_count,
        contrast_strategy = factor_contrasts,
        total_contrasts = contrast_count,
        
        # Operation counts
        model_evaluations = model_evaluations,
        gradient_operations = gradient_operations,
        linear_algebra_operations = linear_algebra_operations,
        
        # Memory requirements (in Float64 elements)
        workspace_memory_elements = 4 * parameter_count,  # 4 buffers
        peak_memory_bytes = 4 * parameter_count * sizeof(Float64),
        
        # Estimated timing (rough estimates based on typical performance)
        estimated_seconds_optimistic = model_evaluations * 100e-9,  # 100ns per evaluation
        estimated_seconds_realistic = model_evaluations * 200e-9,   # 200ns per evaluation
        estimated_seconds_pessimistic = model_evaluations * 500e-9, # 500ns per evaluation
    )
end

"""
    optimize_categorical_contrast_strategy(focal_variable::Symbol,
                                          workspace::MarginalEffectsWorkspace;
                                          max_contrasts::Int = 100) -> Symbol

Recommend optimal contrast strategy based on factor characteristics.
"""
function optimize_categorical_contrast_strategy(
    focal_variable::Symbol,
    workspace::MarginalEffectsWorkspace;
    max_contrasts::Int = 100
)
    
    factor_levels = extract_factor_levels(workspace.column_data[focal_variable])
    level_count = length(factor_levels)
    
    if level_count < 2
        throw(ArgumentError("Variable $focal_variable has fewer than 2 levels"))
    end
    
    # Calculate contrast counts for each strategy
    baseline_contrasts = level_count - 1
    all_pairs_contrasts = level_count * (level_count - 1) ÷ 2
    
    # Decision logic
    if level_count == 2
        # For binary variables, both strategies are equivalent
        return :baseline_contrasts
    elseif level_count <= 5
        # For small number of levels, all pairs is manageable
        return :all_pairs
    elseif all_pairs_contrasts <= max_contrasts
        # If all pairs is within computational budget, prefer it for completeness
        return :all_pairs
    else
        # For many levels, baseline contrasts are more practical
        @warn "Variable $focal_variable has $level_count levels ($all_pairs_contrasts total contrasts). " *
              "Recommending baseline contrasts for computational efficiency."
        return :baseline_contrasts
    end
end

"""
    validate_factor_levels_for_contrasts(focal_variable::Symbol,
                                        workspace::MarginalEffectsWorkspace;
                                        min_observations_per_level::Int = 5) -> Bool

Validate that factor levels have sufficient observations for reliable contrasts.
"""
function validate_factor_levels_for_contrasts(
    focal_variable::Symbol,
    workspace::MarginalEffectsWorkspace;
    min_observations_per_level::Int = 5
)
    
    factor_column = workspace.column_data[focal_variable]
    factor_levels = extract_factor_levels(factor_column)
    
    warnings = String[]
    valid = true
    
    # Check each level for sufficient observations
    for level in factor_levels
        count = sum(==(level), factor_column)
        if count < min_observations_per_level
            push!(warnings, "Level '$level' has only $count observations (minimum recommended: $min_observations_per_level)")
            if count == 0
                valid = false
            end
        end
    end
    
    # Check for extreme imbalance
    level_counts = [sum(==(level), factor_column) for level in factor_levels]
    max_count = maximum(level_counts)
    min_count = minimum(level_counts)
    
    if min_count > 0 && max_count / min_count > 100
        push!(warnings, "Severe class imbalance detected (largest/smallest = $(round(max_count/min_count, digits=1)))")
    end
    
    # Display warnings
    for warning in warnings
        @warn "Factor validation for $focal_variable: $warning"
    end
    
    return valid
end

###############################################################################
# Advanced Categorical Effects Analysis
###############################################################################

"""
    analyze_factor_separation(focal_variable::Symbol,
                             workspace::MarginalEffectsWorkspace,
                             coefficient_vector::AbstractVector,
                             inverse_link_function::Function) -> NamedTuple

Analyze how well factor levels are separated in prediction space.
Helps understand whether contrasts will be meaningful.
"""
function analyze_factor_separation(
    focal_variable::Symbol,
    workspace::MarginalEffectsWorkspace,
    coefficient_vector::AbstractVector,
    inverse_link_function::Function
)
    
    factor_column = workspace.column_data[focal_variable]
    factor_levels = extract_factor_levels(factor_column)
    observation_count = get_observation_count(workspace)
    
    # Compute predictions for each level across all observations
    level_predictions = Dict()
    
    for level in factor_levels
        predictions = Float64[]
        
        for obs_idx in 1:observation_count
            # Evaluate model with this level
            level_overrides = Dict(focal_variable => level)
            evaluate_model_row!(workspace, obs_idx; variable_overrides=level_overrides)
            linear_predictor = dot(workspace.model_row_buffer, coefficient_vector)
            predicted_value = inverse_link_function(linear_predictor)
            push!(predictions, predicted_value)
        end
        
        level_predictions[level] = predictions
    end
    
    # Compute separation statistics
    level_means = Dict(level => mean(preds) for (level, preds) in level_predictions)
    level_stds = Dict(level => std(preds) for (level, preds) in level_predictions)
    
    # Overall statistics
    all_predictions = vcat(collect(values(level_predictions))...)
    overall_mean = mean(all_predictions)
    overall_std = std(all_predictions)
    
    # Between-level variance vs within-level variance
    between_level_variance = var(collect(values(level_means)))
    within_level_variance = mean(values(level_stds).^2)
    separation_ratio = between_level_variance / (within_level_variance + 1e-10)
    
    # Pairwise separations
    pairwise_separations = Dict()
    for i in 1:length(factor_levels)-1
        for j in i+1:length(factor_levels)
            level_i, level_j = factor_levels[i], factor_levels[j]
            mean_diff = abs(level_means[level_i] - level_means[level_j])
            pooled_std = sqrt((level_stds[level_i]^2 + level_stds[level_j]^2) / 2)
            effect_size = pooled_std > 0 ? mean_diff / pooled_std : Inf
            pairwise_separations[(level_i, level_j)] = (
                mean_difference = mean_diff,
                effect_size = effect_size
            )
        end
    end
    
    return (
        focal_variable = focal_variable,
        factor_levels = factor_levels,
        observations = observation_count,
        
        # Level-specific statistics
        level_means = level_means,
        level_standard_deviations = level_stds,
        
        # Overall separation
        overall_mean = overall_mean,
        overall_standard_deviation = overall_std,
        between_level_variance = between_level_variance,
        within_level_variance = within_level_variance,
        separation_ratio = separation_ratio,
        
        # Pairwise comparisons
        pairwise_separations = pairwise_separations,
        
        # Interpretation
        well_separated = separation_ratio > 1.0,
        max_effect_size = maximum(sep.effect_size for sep in values(pairwise_separations)),
        min_effect_size = minimum(sep.effect_size for sep in values(pairwise_separations))
    )
end

###############################################################################
# Factor Level Display and Summary Functions
###############################################################################

"""
    summarize_factor_levels(focal_variable::Symbol, workspace::MarginalEffectsWorkspace) -> NamedTuple

Create a summary of factor level characteristics for a categorical variable.
"""
function summarize_factor_levels(focal_variable::Symbol, workspace::MarginalEffectsWorkspace)
    factor_column = workspace.column_data[focal_variable]
    factor_levels = extract_factor_levels(factor_column)
    observation_count = get_observation_count(workspace)
    
    # Compute frequencies and proportions
    level_frequencies = Dict()
    level_proportions = Dict()
    
    for level in factor_levels
        count = sum(==(level), factor_column)
        level_frequencies[level] = count
        level_proportions[level] = count / observation_count
    end
    
    # Summary statistics
    most_frequent_level = argmax(level_frequencies)
    least_frequent_level = argmin(level_frequencies)
    max_frequency = maximum(values(level_frequencies))
    min_frequency = minimum(values(level_frequencies))
    
    # Balance metrics
    frequency_ratio = max_frequency / max(min_frequency, 1)
    entropy = -sum(p * log(p + 1e-10) for p in values(level_proportions))
    max_entropy = log(length(factor_levels))
    normalized_entropy = entropy / max_entropy
    
    return (
        focal_variable = focal_variable,
        total_observations = observation_count,
        factor_levels = factor_levels,
        level_count = length(factor_levels),
        
        # Frequency information
        level_frequencies = level_frequencies,
        level_proportions = level_proportions,
        most_frequent_level = most_frequent_level,
        least_frequent_level = least_frequent_level,
        
        # Balance metrics
        frequency_ratio = frequency_ratio,
        entropy = entropy,
        normalized_entropy = normalized_entropy,
        
        # Classification
        is_balanced = frequency_ratio < 2.0,
        is_binary = length(factor_levels) == 2,
        has_rare_levels = min_frequency < max(5, observation_count * 0.01)
    )
end

"""
    display_factor_summary(focal_variable::Symbol, workspace::MarginalEffectsWorkspace)

Display a formatted summary of factor level characteristics.
"""
function display_factor_summary(focal_variable::Symbol, workspace::MarginalEffectsWorkspace)
    summary = summarize_factor_levels(focal_variable, workspace)
    
    println("Factor Summary: $(summary.focal_variable)")
    println("═" ^ 40)
    println("Total observations: $(summary.total_observations)")
    println("Factor levels: $(summary.level_count)")
    println("Balance: $(summary.is_balanced ? "Balanced" : "Imbalanced") (ratio: $(round(summary.frequency_ratio, digits=1)))")
    println("Entropy: $(round(summary.normalized_entropy, digits=3)) (normalized)")
    
    if summary.has_rare_levels
        println("⚠ Warning: Some levels have very few observations")
    end
    
    println("\nLevel frequencies:")
    for level in summary.factor_levels
        freq = summary.level_frequencies[level]
        prop = summary.level_proportions[level]
        bar_length = max(1, round(Int, prop * 40))
        bar = "█" ^ bar_length
        println("  $(rpad(string(level), 15)) $freq ($(round(prop*100, digits=1))%) $bar")
    end
end

###############################################################################
# Categorical Effects Validation
###############################################################################

"""
    validate_categorical_effects_setup(focal_variable::Symbol,
                                      workspace::MarginalEffectsWorkspace,
                                      factor_contrasts::Symbol;
                                      min_observations_per_level::Int = 5,
                                      max_contrasts::Int = 100) -> NamedTuple

Comprehensive validation of categorical effects setup before computation.
"""
function validate_categorical_effects_setup(
    focal_variable::Symbol,
    workspace::MarginalEffectsWorkspace,
    factor_contrasts::Symbol;
    min_observations_per_level::Int = 5,
    max_contrasts::Int = 100
)
    
    validation_results = []
    warnings = String[]
    errors = String[]
    
    # 1. Basic variable validation
    try
        validate_categorical_variable_structure(focal_variable, workspace.column_data)
        push!(validation_results, "✅ Variable structure valid")
    catch e
        push!(errors, "❌ Variable structure invalid: $e")
    end
    
    # 2. Factor level validation
    try
        levels_valid = validate_factor_levels_for_contrasts(
            focal_variable, workspace; 
            min_observations_per_level=min_observations_per_level
        )
        if levels_valid
            push!(validation_results, "✅ Factor levels have sufficient observations")
        else
            push!(warnings, "⚠ Some factor levels have insufficient observations")
        end
    catch e
        push!(errors, "❌ Factor level validation failed: $e")
    end
    
    # 3. Contrast strategy validation
    try
        recommended_strategy = optimize_categorical_contrast_strategy(
            focal_variable, workspace; max_contrasts=max_contrasts
        )
        if recommended_strategy == factor_contrasts
            push!(validation_results, "✅ Contrast strategy optimal")
        else
            push!(warnings, "⚠ Consider using $recommended_strategy instead of $factor_contrasts")
        end
    catch e
        push!(errors, "❌ Contrast strategy validation failed: $e")
    end
    
    # 4. Computational complexity check
    try
        complexity = estimate_categorical_computation_complexity(focal_variable, workspace, factor_contrasts)
        if complexity.total_contrasts <= max_contrasts
            push!(validation_results, "✅ Computational complexity reasonable ($(complexity.total_contrasts) contrasts)")
        else
            push!(warnings, "⚠ High computational complexity ($(complexity.total_contrasts) contrasts)")
        end
        
        if complexity.estimated_seconds_realistic < 60.0
            push!(validation_results, "✅ Estimated computation time reasonable")
        else
            push!(warnings, "⚠ Long estimated computation time ($(round(complexity.estimated_seconds_realistic, digits=1))s)")
        end
    catch e
        push!(errors, "❌ Complexity estimation failed: $e")
    end
    
    # 5. Factor separation analysis
    try
        # This requires coefficient_vector and inverse_link_function, so we'll skip for now
        # Could be added if those are available in the validation context
        push!(validation_results, "ℹ Factor separation analysis skipped (requires fitted model)")
    catch e
        push!(warnings, "⚠ Factor separation analysis failed: $e")
    end
    
    # Overall assessment
    has_errors = !isempty(errors)
    has_warnings = !isempty(warnings)
    
    overall_status = if has_errors
        "❌ FAILED"
    elseif has_warnings
        "⚠ PASSED WITH WARNINGS"
    else
        "✅ PASSED"
    end
    
    return (
        focal_variable = focal_variable,
        factor_contrasts = factor_contrasts,
        overall_status = overall_status,
        validation_results = validation_results,
        warnings = warnings,
        errors = errors,
        has_errors = has_errors,
        has_warnings = has_warnings,
        ready_for_computation = !has_errors
    )
end

function Base.show(io::IO, validation::NamedTuple{(:focal_variable, :factor_contrasts, :overall_status, :validation_results, :warnings, :errors, :has_errors, :has_warnings, :ready_for_computation)})
    println(io, "Categorical Effects Validation: $(validation.focal_variable)")
    println(io, "Contrast strategy: $(validation.factor_contrasts)")
    println(io, "Overall status: $(validation.overall_status)")
    println(io, "━" ^ 50)
    
    if !isempty(validation.validation_results)
        println(io, "Validation results:")
        for result in validation.validation_results
            println(io, "  $result")
        end
    end
    
    if !isempty(validation.warnings)
        println(io, "\nWarnings:")
        for warning in validation.warnings
            println(io, "  $warning")
        end
    end
    
    if !isempty(validation.errors)
        println(io, "\nErrors:")
        for error in validation.errors
            println(io, "  $error")
        end
    end
    
    println(io, "\nReady for computation: $(validation.ready_for_computation ? "Yes" : "No")")
end

###############################################################################
# Categorical Effects Performance Optimization
###############################################################################

"""
    optimize_categorical_computation(focal_variable::Symbol,
                                   workspace::MarginalEffectsWorkspace,
                                   factor_contrasts::Symbol) -> NamedTuple

Suggest optimizations for categorical variable marginal effects computation.
"""
function optimize_categorical_computation(
    focal_variable::Symbol,
    workspace::MarginalEffectsWorkspace,
    factor_contrasts::Symbol
)
    
    factor_levels = extract_factor_levels(workspace.column_data[focal_variable])
    level_count = length(factor_levels)
    observation_count = get_observation_count(workspace)
    
    suggestions = String[]
    performance_tips = String[]
    
    # Analyze current setup
    complexity = estimate_categorical_computation_complexity(focal_variable, workspace, factor_contrasts)
    
    # Level count optimizations
    if level_count > 20
        push!(suggestions, "Consider grouping factor levels - $level_count levels may be excessive")
        push!(performance_tips, "Group rare levels into 'Other' category to reduce contrasts")
    elseif level_count > 10 && factor_contrasts == :all_pairs
        push!(suggestions, "Consider using :baseline_contrasts instead of :all_pairs")
        baseline_contrasts = level_count - 1
        all_pairs_contrasts = complexity.total_contrasts
        time_savings = (all_pairs_contrasts - baseline_contrasts) / all_pairs_contrasts * 100
        push!(performance_tips, "Would reduce contrasts from $all_pairs_contrasts to $baseline_contrasts ($(round(time_savings, digits=1))% time savings)")
    end
    
    # Observation count optimizations
    if observation_count > 1_000_000
        push!(suggestions, "For very large datasets, consider stratified sampling for initial exploration")
        push!(performance_tips, "Sample ~100K observations to validate approach before full computation")
    end
    
    # Memory optimizations
    workspace_memory_mb = complexity.peak_memory_bytes / (1024^2)
    if workspace_memory_mb > 100
        push!(suggestions, "High memory usage detected ($(round(workspace_memory_mb, digits=1)) MB)")
        push!(performance_tips, "Consider processing subsets of variables separately")
    end
    
    # Computational time optimizations
    if complexity.estimated_seconds_realistic > 300  # 5 minutes
        push!(suggestions, "Long computation time estimated ($(round(complexity.estimated_seconds_realistic/60, digits=1)) minutes)")
        push!(performance_tips, "Consider parallel processing or cloud computing for large-scale analysis")
    end
    
    return (
        focal_variable = focal_variable,
        current_strategy = factor_contrasts,
        factor_levels = level_count,
        total_contrasts = complexity.total_contrasts,
        estimated_time_seconds = complexity.estimated_seconds_realistic,
        memory_usage_mb = workspace_memory_mb,
        suggestions = suggestions,
        performance_tips = performance_tips,
        optimization_potential = !isempty(suggestions)
    )
end

# export validate_categorical_variable_structure
# export estimate_categorical_computation_complexity, optimize_categorical_contrast_strategy
# export validate_factor_levels_for_contrasts, analyze_factor_separation
# export summarize_factor_levels, display_factor_summary
# export validate_categorical_effects_setup, optimize_categorical_computation
# export CategoricalEffectDiagnostics
# export diagnose_categorical_effect_computation, benchmark_categorical_effect_computation

