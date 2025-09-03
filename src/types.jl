# types.jl - Result types, error types, and display methods

"""
    MarginsResult

Container for marginal effects results with flexible table formatting.

Fields:
- `estimates::Vector{Float64}`: Point estimates
- `standard_errors::Vector{Float64}`: Standard errors
- `terms::Vector{Symbol}`: Variable names (clean, no suffixes)
- `profile_values::Union{Nothing, NamedTuple}`: Reference grid values for profile margins
- `group_values::Union{Nothing, NamedTuple}`: Grouping variable values
- `gradients::Matrix{Float64}`: Parameter gradients (G matrix) for delta-method
- `metadata::Dict{Symbol, Any}`: Analysis metadata (model info, options used, etc.)

# Examples
```julia
result = population_margins(model, data; type=:effects, vars=[:x1, :x2])
DataFrame(result)                     # Auto-detect format (:standard for population)
DataFrame(result; format=:compact)    # Just estimates and SEs
DataFrame(result; format=:confidence) # Confidence intervals only

# Confidence intervals with ci_alpha parameter
result = population_margins(model, data; type=:effects, vars=[:x1, :x2], ci_alpha=0.05)
DataFrame(result)                     # Includes ci_lower and ci_upper columns

result.estimates      # Access raw estimates
result.gradients      # Access parameter gradients
result.metadata       # Access analysis metadata (includes :alpha when ci_alpha specified)
```
"""
struct MarginsResult
    # Core statistical results
    estimates::Vector{Float64}
    standard_errors::Vector{Float64}  
    terms::Vector{String}  # Clean variable names (converted to strings for display)
    
    # Structural information
    profile_values::Union{Nothing, NamedTuple}  # Reference grid (profile margins only)
    group_values::Union{Nothing, NamedTuple}    # Grouping columns
    
    # Statistical metadata  
    gradients::Matrix{Float64}
    metadata::Dict{Symbol, Any}
end

import DataFrames: DataFrame # explicit import to extend method

# Analysis-type aware format dispatch with validation
function DataFrame(mr::MarginsResult; format::Symbol=:auto)
    # Auto-detect natural format based on analysis type
    if format == :auto
        analysis_type = get(mr.metadata, :analysis_type, :unknown)
        format = analysis_type == :profile ? :profile : :standard
    end
    
    # Validate format compatibility
    if format == :profile && mr.profile_values === nothing
        throw(ArgumentError("format=:profile requires profile-based results (from profile_margins)"))
    end
    
    if format == :standard
        return _standard_table(mr)
    elseif format == :compact
        return _compact_table(mr)
    elseif format == :confidence
        return _confidence_table(mr)
    elseif format == :profile
        return _profile_table(mr)
    elseif format == :stata
        return _stata_table(mr)
    else
        error("Unknown format: $format")
    end
end

# Tables.jl interface implementation - uses auto format
Tables.istable(::Type{MarginsResult}) = true
Tables.rowaccess(::Type{MarginsResult}) = true
Tables.rows(mr::MarginsResult) = Tables.rows(DataFrame(mr))
Tables.schema(mr::MarginsResult) = Tables.schema(DataFrame(mr))

# DataFrame conversion compatibility
Base.convert(::Type{DataFrame}, mr::MarginsResult) = DataFrame(mr)

# Stata-style display methods
function Base.show(io::IO, mr::MarginsResult)
    n_results = length(mr.estimates)
    analysis_type = get(mr.metadata, :analysis_type, :unknown)
    
    # Header
    if analysis_type == :profile
        n_profiles = get(mr.metadata, :n_profiles, 1)
        println(io, "MarginsResult: $n_results results at $n_profiles profiles")
    else
        println(io, "MarginsResult: $n_results population results")  
    end
    
    # Stata-style table with horizontal lines
    _show_stata_table(io, mr)
end

function Base.show(io::IO, ::MIME"text/plain", mr::MarginsResult)
    show(io, mr)
    # unpleasant to show this every time?
    # println(io, "\nUse DataFrame(result; format=...) for different table formats")
end

function _show_stata_table(io::IO, mr::MarginsResult)
    # Calculate confidence intervals
    alpha = get(mr.metadata, :alpha, 0.05)
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha)
    
    # Column widths for alignment
    var_width = max(8, maximum(length.(mr.terms)) + 1)
    num_width = 12
    
    # Header line
    println(io, "─"^(var_width + 4*num_width + 3))
    
    # Column headers - dynamic based on measure
    measure = get(mr.metadata, :measure, :effect)
    header_text = measure === :effect ? "dy/dx" :
                  measure === :elasticity ? "eyex" :
                  measure === :semielasticity_dyex ? "dyex" :
                  measure === :semielasticity_eydx ? "eydx" : "dy/dx"
    
    print(io, rpad("", var_width))
    print(io, lpad(header_text, num_width))
    print(io, lpad("Std. Err.", num_width))
    print(io, lpad("[$(Int(100*(1-alpha)))% Conf.", num_width))
    print(io, lpad("Interval]", num_width))
    println(io)
    
    # Separator line  
    println(io, "─"^(var_width + 4*num_width + 3))
    
    # Data rows
    for i in 1:length(mr.estimates)
        print(io, rpad(mr.terms[i], var_width))
        print(io, lpad(sprintf("%.6f", mr.estimates[i]), num_width))
        print(io, lpad(sprintf("%.6f", mr.standard_errors[i]), num_width))
        print(io, lpad(sprintf("%.6f", lower[i]), num_width))
        print(io, lpad(sprintf("%.6f", upper[i]), num_width))
        println(io)
    end
    
    # Bottom line
    println(io, "─"^(var_width + 4*num_width + 3))
end

# Simple sprintf-like formatting
function sprintf(fmt::String, x::Float64)
    if fmt == "%.6f"
        return @sprintf("%.6f", x)
    else
        return string(x)
    end
end

# Utility function for confidence interval calculation
function _calculate_confidence_intervals(estimates::Vector{Float64}, standard_errors::Vector{Float64}, alpha::Float64)
    z = quantile(Normal(), 1 - alpha/2)
    lower = estimates .- z .* standard_errors
    upper = estimates .+ z .* standard_errors
    return lower, upper
end

# Format-specific table builders

function _standard_table(mr::MarginsResult)
    t_stats = mr.estimates ./ mr.standard_errors
    p_values = 2 .* (1 .- cdf.(Normal(), abs.(t_stats)))
    
    df = DataFrame(
        term = mr.terms,  # Clean variable names, not "x1_effect"
        estimate = mr.estimates,
        se = mr.standard_errors, 
        t_stat = t_stats,
        p_value = p_values
    )
    
    # Add confidence intervals if alpha is specified in metadata
    if haskey(mr.metadata, :alpha)
        alpha = mr.metadata[:alpha]
        lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha)
        df[!, :ci_lower] = lower
        df[!, :ci_upper] = upper
    end
    
    # CRITICAL: Use actual subgroup sizes, never fallback to incorrect quantities
    if haskey(mr.metadata, :has_subgroup_n) && haskey(mr.metadata, :subgroup_n_values)
        # Grouped case: use the actual subgroup sizes from computation
        df[!, :n] = mr.metadata[:subgroup_n_values]
    else
        # Simple case: use overall sample size  
        n_obs = get(mr.metadata, :n_obs, missing)
        df[!, :n] = fill(n_obs, length(mr.estimates))
    end
    
    _add_structural_columns!(df, mr)
    return df
end

function _compact_table(mr::MarginsResult)
    df = DataFrame(
        term = mr.terms,
        estimate = mr.estimates,
        se = mr.standard_errors
    )
    _add_structural_columns!(df, mr)
    return df
end

function _confidence_table(mr::MarginsResult)
    alpha = get(mr.metadata, :alpha, 0.05)
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha)
    
    df = DataFrame(
        term = mr.terms,
        estimate = mr.estimates,
        lower = lower,
        upper = upper
    )
    _add_structural_columns!(df, mr)
    return df
end

function _profile_table(mr::MarginsResult)
    # Profile-first organization: reference grid with results attached
    alpha = get(mr.metadata, :alpha, 0.05)
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha)
    
    df = DataFrames.DataFrame()
    
    # Add profile columns first (primary organization) - just variable names
    if mr.profile_values !== nothing
        for (k, v) in pairs(mr.profile_values)
            df[!, k] = v
        end
    end
    
    # Add results columns
    df[!, :term] = mr.terms  # Clean variable names
    df[!, :estimate] = mr.estimates
    df[!, :se] = mr.standard_errors
    df[!, :lower] = lower
    df[!, :upper] = upper
    
    # Add sample size from metadata
    n_obs = get(mr.metadata, :n_obs, missing)
    df[!, :n] = fill(n_obs, length(mr.estimates))
    
    # Add grouping columns if present
    if mr.group_values !== nothing
        for (k, v) in pairs(mr.group_values)
            df[!, k] = v
        end
    end
    
    return df
end

function _stata_table(mr::MarginsResult)
    t_stats = mr.estimates ./ mr.standard_errors  
    p_values = 2 .* (1 .- cdf.(Normal(), abs.(t_stats)))
    
    # Get sample size from metadata
    n_obs = get(mr.metadata, :n_obs, missing)
    
    df = DataFrame(
        margin = mr.estimates,      # Stata uses "margin" not "estimate"
        std_err = mr.standard_errors, # "std_err" not "se" 
        t = t_stats,
        P_t = p_values,            # "P>|t|" equivalent
        N = fill(n_obs, length(mr.estimates))  # Add sample size (Stata uses uppercase N)
    )
    _add_structural_columns!(df, mr)
    return df
end

function _add_structural_columns!(df, mr::MarginsResult)
    # Add profile columns (at_x1, at_x2, etc.) - only for non-profile formats
    if mr.profile_values !== nothing
        for (k, v) in pairs(mr.profile_values)
            df[!, Symbol("at_", k)] = v
        end
    end
    
    # Add grouping columns from over/by parameters
    if mr.group_values !== nothing
        for (k, v) in pairs(mr.group_values)
            df[!, k] = v
        end
    end
end

# Custom error types for clear user feedback
struct MarginsError <: Exception
    msg::String
end

struct StatisticalValidityError <: Exception
    msg::String
end

# Make error messages display cleanly
Base.showerror(io::IO, e::MarginsError) = print(io, "MarginsError: ", e.msg)
Base.showerror(io::IO, e::StatisticalValidityError) = print(io, "StatisticalValidityError: ", e.msg)