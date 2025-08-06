# marginal_effects_results.jl
# Result types and display methods

using Printf
using Distributions: Normal, TDist, cdf, quantile

"""
    MarginalEffectsResult{EffectType}

Container for marginal effects computation results with complete metadata.

# Type Parameter
- `EffectType`: Either `:dydx` for marginal effects or `:predictions` for predicted values

# Fields
- `focal_variables::Vector{Symbol}`: Variables for which effects were computed
- `representative_values::AbstractDict`: Representative value combinations used
- `effect_estimates::AbstractDict`: Effect estimates (scalar or Dict{Tuple,Float64})
- `standard_errors::AbstractDict`: Standard errors for each effect
- `gradient_vectors::AbstractDict`: Delta-method gradients for each effect
- `n_observations::Int`: Number of observations in original data
- `df_residual::Real`: Model degrees of freedom
- `model_family::String`: Model distribution family
- `model_link::String`: Model link function

# Result Structure
- When `representative_values` is empty: effects are scalars for each focal variable
- When `representative_values` is non-empty: effects are Dict{Tuple,Float64} mapping combinations to estimates
- For categorical variables: effects are always Dict{Tuple,Float64} mapping level pairs to contrasts
"""
struct MarginalEffectsResult{EffectType} <: AbstractMarginalEffectsResult
    focal_variables::Vector{Symbol}
    representative_values::AbstractDict
    effect_estimates::AbstractDict
    standard_errors::AbstractDict
    gradient_vectors::AbstractDict
    n_observations::Int
    df_residual::Real
    model_family::String
    model_link::String
end

# Abstract supertype for results
abstract type AbstractMarginalEffectsResult end

"""
    get_effect_type(result::MarginalEffectsResult{T}) -> Symbol

Extract the effect type from a MarginalEffectsResult.
"""
get_effect_type(::MarginalEffectsResult{T}) where T = T

###############################################################################
# Result Access Functions
###############################################################################

"""
    get_focal_variables(result::MarginalEffectsResult) -> Vector{Symbol}

Get the focal variables for which effects were computed.
"""
get_focal_variables(result::MarginalEffectsResult) = result.focal_variables

"""
    get_effect_estimate(result::MarginalEffectsResult, variable::Symbol)

Get the effect estimate for a specific variable.
Returns either a scalar (no representative values) or Dict{Tuple,Float64} (with representative values).
"""
function get_effect_estimate(result::MarginalEffectsResult, variable::Symbol)
    if variable ∉ result.focal_variables
        throw(ArgumentError("Variable $variable not found in focal variables"))
    end
    return result.effect_estimates[variable]
end

"""
    get_standard_error(result::MarginalEffectsResult, variable::Symbol)

Get the standard error for a specific variable's effect.
"""
function get_standard_error(result::MarginalEffectsResult, variable::Symbol)
    if variable ∉ result.focal_variables
        throw(ArgumentError("Variable $variable not found in focal variables"))
    end
    return result.standard_errors[variable]
end

"""
    get_gradient_vector(result::MarginalEffectsResult, variable::Symbol)

Get the delta-method gradient vector for a specific variable's effect.
"""
function get_gradient_vector(result::MarginalEffectsResult, variable::Symbol)
    if variable ∉ result.focal_variables
        throw(ArgumentError("Variable $variable not found in focal variables"))
    end
    return result.gradient_vectors[variable]
end

"""
    has_representative_values(result::MarginalEffectsResult) -> Bool

Check if the result includes representative value computations.
"""
has_representative_values(result::MarginalEffectsResult) = !isempty(result.representative_values)

"""
    get_representative_variables(result::MarginalEffectsResult) -> Vector{Symbol}

Get the variables that were set to representative values.
"""
function get_representative_variables(result::MarginalEffectsResult)
    return collect(keys(result.representative_values))
end

###############################################################################
# Statistical Inference
###############################################################################

"""
    compute_confidence_intervals(result::MarginalEffectsResult; confidence_level::Real = 0.95)

Compute confidence intervals for all effects in the result.

# Arguments
- `result`: MarginalEffectsResult containing estimates and standard errors
- `confidence_level`: Confidence level (default: 0.95 for 95% intervals)

# Returns
Dict with same structure as effect_estimates, containing (lower_bound, upper_bound) tuples
"""
function compute_confidence_intervals(result::MarginalEffectsResult; confidence_level::Real = 0.95)
    if !(0 < confidence_level < 1)
        throw(ArgumentError("confidence_level must be between 0 and 1, got $confidence_level"))
    end
    
    # Compute critical value (use t-distribution for small samples, normal for large)
    alpha = 1 - confidence_level
    if result.df_residual < 30
        critical_value = quantile(TDist(result.df_residual), 1 - alpha/2)
    else
        critical_value = quantile(Normal(), 1 - alpha/2)
    end
    
    confidence_intervals = Dict{Symbol,Any}()
    
    for variable in result.focal_variables
        estimate = result.effect_estimates[variable]
        standard_error = result.standard_errors[variable]
        
        if estimate isa Number
            # Scalar effect
            margin_of_error = critical_value * standard_error
            lower_bound = estimate - margin_of_error
            upper_bound = estimate + margin_of_error
            confidence_intervals[variable] = (lower_bound, upper_bound)
            
        else
            # Dictionary of effects (representative values or categorical contrasts)
            variable_intervals = Dict{Tuple,Tuple{Float64,Float64}}()
            
            for (key, effect_value) in estimate
                se_value = standard_error[key]
                margin_of_error = critical_value * se_value
                lower_bound = effect_value - margin_of_error
                upper_bound = effect_value + margin_of_error
                variable_intervals[key] = (lower_bound, upper_bound)
            end
            
            confidence_intervals[variable] = variable_intervals
        end
    end
    
    return confidence_intervals
end

"""
    compute_z_statistics(result::MarginalEffectsResult)

Compute z-statistics for all effects (estimate / standard_error).
"""
function compute_z_statistics(result::MarginalEffectsResult)
    z_statistics = Dict{Symbol,Any}()
    
    for variable in result.focal_variables
        estimate = result.effect_estimates[variable]
        standard_error = result.standard_errors[variable]
        
        if estimate isa Number
            # Scalar effect
            z_stat = standard_error > 0 ? estimate / standard_error : NaN
            z_statistics[variable] = z_stat
            
        else
            # Dictionary of effects
            variable_z_stats = Dict{Tuple,Float64}()
            
            for (key, effect_value) in estimate
                se_value = standard_error[key]
                z_stat = se_value > 0 ? effect_value / se_value : NaN
                variable_z_stats[key] = z_stat
            end
            
            z_statistics[variable] = variable_z_stats
        end
    end
    
    return z_statistics
end

"""
    compute_p_values(result::MarginalEffectsResult)

Compute two-tailed p-values for all effects.
"""
function compute_p_values(result::MarginalEffectsResult)
    z_stats = compute_z_statistics(result)
    p_values = Dict{Symbol,Any}()
    
    # Use appropriate distribution
    if result.df_residual < 30
        dist = TDist(result.df_residual)
    else
        dist = Normal()
    end
    
    for variable in result.focal_variables
        z_stat = z_stats[variable]
        
        if z_stat isa Number
            # Scalar z-statistic
            p_val = isfinite(z_stat) ? 2 * (1 - cdf(dist, abs(z_stat))) : NaN
            p_values[variable] = p_val
            
        else
            # Dictionary of z-statistics
            variable_p_values = Dict{Tuple,Float64}()
            
            for (key, z_value) in z_stat
                p_val = isfinite(z_value) ? 2 * (1 - cdf(dist, abs(z_value))) : NaN
                variable_p_values[key] = p_val
            end
            
            p_values[variable] = variable_p_values
        end
    end
    
    return p_values
end

###############################################################################
# Pretty Printing
###############################################################################

"""
    format_p_value(p::Real) -> String

Format p-value for display with appropriate precision.
"""
function format_p_value(p::Real)
    if !isfinite(p)
        return "---"
    elseif p < 1e-16
        return "<1e-16"
    elseif p < 1e-8
        return "<1e-08"
    elseif p < 1e-4
        return "<0.0001"
    elseif p < 0.001
        return @sprintf("%.3e", p)
    else
        return @sprintf("%.4f", p)
    end
end

"""
    format_coefficient(coef::Real, digits::Int = 4) -> String

Format coefficient for display with consistent width.
"""
function format_coefficient(coef::Real, digits::Int = 4)
    if !isfinite(coef)
        return @sprintf("%*s", digits + 3, "---")
    else
        return @sprintf("%*.4f", digits + 3, coef)
    end
end

function Base.show(io::IO, ::MIME"text/plain", result::MarginalEffectsResult)
    effect_type = get_effect_type(result)
    
    # Header
    title = if effect_type == :dydx
        "Marginal Effects"
    elseif effect_type == :predictions
        "Average Predictions"
    else
        "Effects"
    end
    
    println(io, title)
    println(io, "Model: $(result.model_family), Link: $(result.model_link)")
    println(io, "Observations: $(result.n_observations), df: $(result.df_residual)")
    println(io, "═" ^ 80)
    
    # Representative values header if applicable
    if has_representative_values(result)
        rep_vars = get_representative_variables(result)
        println(io, "Representative values set for: $(join(rep_vars, ", "))")
        println(io)
        
        # Print column headers for representative values
        for var in rep_vars
            @printf(io, "%-12s ", string(var))
        end
    end
    
    @printf(io, "%-15s %12s %12s %8s %10s %20s\n",
            "Variable", "Estimate", "Std.Error", "z-stat", "P>|z|", "95% CI")
    println(io, "─" ^ 80)
    
    # Compute statistics
    confidence_intervals = compute_confidence_intervals(result; confidence_level=0.95)
    z_statistics = compute_z_statistics(result)
    p_values = compute_p_values(result)
    
    # Print results for each focal variable
    for variable in result.focal_variables
        estimate = result.effect_estimates[variable]
        standard_error = result.standard_errors[variable]
        z_stat = z_statistics[variable]
        p_val = p_values[variable]
        ci = confidence_intervals[variable]
        
        if estimate isa Number
            # Scalar effect - simple case
            if has_representative_values(result)
                # Print empty cells for representative value columns
                for _ in get_representative_variables(result)
                    @printf(io, "%-12s ", "")
                end
            end
            
            @printf(io, "%-15s %12s %12s %8s %10s [%8s, %8s]\n",
                    string(variable),
                    format_coefficient(estimate),
                    format_coefficient(standard_error),
                    format_coefficient(z_stat),
                    format_p_value(p_val),
                    format_coefficient(ci[1]),
                    format_coefficient(ci[2]))
            
        else
            # Dictionary of effects - representative values or categorical contrasts
            sorted_keys = sort(collect(keys(estimate)))
            
            for (i, key) in enumerate(sorted_keys)
                effect_value = estimate[key]
                se_value = standard_error[key]
                z_value = z_stat[key]
                p_value = p_val[key]
                ci_value = ci[key]
                
                # Print representative value combination or contrast levels
                if has_representative_values(result) && length(key) >= length(get_representative_variables(result))
                    # Representative values case
                    rep_count = length(get_representative_variables(result))
                    for j in 1:rep_count
                        @printf(io, "%-12s ", string(key[j]))
                    end
                    
                    # Extract contrast part if this is also a categorical variable
                    if length(key) > rep_count
                        contrast_part = key[rep_count+1:end]
                        variable_label = "$(variable)[$(join(contrast_part, "→"))]"
                    else
                        variable_label = string(variable)
                    end
                else
                    # Categorical contrast case (no representative values)
                    if has_representative_values(result)
                        # Print empty cells for representative value columns
                        for _ in get_representative_variables(result)
                            @printf(io, "%-12s ", "")
                        end
                    end
                    variable_label = "$(variable)[$(join(key, "→"))]"
                end
                
                @printf(io, "%-15s %12s %12s %8s %10s [%8s, %8s]\n",
                        variable_label,
                        format_coefficient(effect_value),
                        format_coefficient(se_value),
                        format_coefficient(z_value),
                        format_p_value(p_value),
                        format_coefficient(ci_value[1]),
                        format_coefficient(ci_value[2]))
            end
        end
    end
    
    println(io, "═" ^ 80)
end

###############################################################################
# DataFrame Conversion
###############################################################################

"""
    DataFrame(result::MarginalEffectsResult) -> DataFrame

Convert MarginalEffectsResult to DataFrame for further analysis.
"""
function DataFrame(result::MarginalEffectsResult)
    rows = []
    
    # Compute additional statistics
    confidence_intervals = compute_confidence_intervals(result)
    z_statistics = compute_z_statistics(result)
    p_values = compute_p_values(result)
    
    for variable in result.focal_variables
        estimate = result.effect_estimates[variable]
        standard_error = result.standard_errors[variable]
        z_stat = z_statistics[variable]
        p_val = p_values[variable]
        ci = confidence_intervals[variable]
        
        if estimate isa Number
            # Scalar effect
            row = Dict{Symbol,Any}(
                :variable => variable,
                :estimate => estimate,
                :std_error => standard_error,
                :z_statistic => z_stat,
                :p_value => p_val,
                :ci_lower => ci[1],
                :ci_upper => ci[2]
            )
            
            # Add empty representative value columns if needed
            if has_representative_values(result)
                for rep_var in get_representative_variables(result)
                    row[Symbol("rep_", rep_var)] = missing
                end
            end
            
            push!(rows, row)
            
        else
            # Dictionary of effects
            for (key, effect_value) in estimate
                se_value = standard_error[key]
                z_value = z_stat[key]
                p_value = p_val[key]
                ci_value = ci[key]
                
                row = Dict{Symbol,Any}(
                    :variable => variable,
                    :estimate => effect_value,
                    :std_error => se_value,
                    :z_statistic => z_value,
                    :p_value => p_value,
                    :ci_lower => ci_value[1],
                    :ci_upper => ci_value[2]
                )
                
                # Add representative value columns and contrast information
                if has_representative_values(result)
                    rep_vars = get_representative_variables(result)
                    rep_count = length(rep_vars)
                    
                    for (i, rep_var) in enumerate(rep_vars)
                        if i <= length(key)
                            row[Symbol("rep_", rep_var)] = key[i]
                        else
                            row[Symbol("rep_", rep_var)] = missing
                        end
                    end
                    
                    # Add contrast information if this is a categorical variable
                    if length(key) > rep_count
                        contrast_levels = key[rep_count+1:end]
                        if length(contrast_levels) >= 2
                            row[:contrast_from] = contrast_levels[1]
                            row[:contrast_to] = contrast_levels[2]
                        end
                    end
                else
                    # Pure categorical contrast
                    if length(key) >= 2
                        row[:contrast_from] = key[1]
                        row[:contrast_to] = key[2]
                    end
                end
                
                push!(rows, row)
            end
        end
    end
    
    return DataFrame(rows)
end
