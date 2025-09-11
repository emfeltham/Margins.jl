# types.jl - Result types, error types, and display methods

# Global display settings (defined here so it's available to formatting functions)
const DISPLAY_DIGITS = Ref(5)  # For statistical results
const PROFILE_DIGITS = Ref(3)  # For reference grid values

"""
    set_display_digits(n::Int)

Set the number of significant digits for statistical results (dy/dx, std error, CI) in show methods.
Default is 5 digits.
"""
function set_display_digits(n::Int)
    if n < 1 || n > 15
        throw(ArgumentError("Display digits must be between 1 and 15"))
    end
    DISPLAY_DIGITS[] = n
    return nothing
end

"""
    get_display_digits() -> Int

Get the current number of significant digits for statistical results.
"""
get_display_digits() = DISPLAY_DIGITS[]

"""
    set_profile_digits(n::Int)

Set the number of significant digits for profile/reference grid values in show methods.
Default is 3 digits.
"""
function set_profile_digits(n::Int)
    if n < 1 || n > 15
        throw(ArgumentError("Profile digits must be between 1 and 15"))
    end
    PROFILE_DIGITS[] = n
    return nothing
end

"""
    get_profile_digits() -> Int

Get the current number of significant digits for profile/reference grid values.
"""
get_profile_digits() = PROFILE_DIGITS[]

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
    variables::Vector{String}  # The "x" in dy/dx - which variable each row represents
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
    n_obs = get(mr.metadata, :n_obs, missing)
    n_text = ismissing(n_obs) ? "" : " (N=$n_obs)"
    
    if analysis_type == :profile
        n_profiles = get(mr.metadata, :n_profiles, 1)
        println(io, "MarginsResult: $n_results results at $n_profiles profiles$n_text")
    else
        println(io, "MarginsResult: $n_results population results$n_text")  
    end
    
    # Stata-style table with horizontal lines
    _show_stata_table(io, mr)
end

function Base.show(io::IO, ::MIME"text/plain", mr::MarginsResult)
    show(io, mr)
    # unpleasant to show this every time?
    # println(io, "\nUse DataFrame(result; format=...) for different table formats")
end

# Helper function to generate clean display labels for show method
function _generate_clean_display_labels(mr::MarginsResult)
    # Combine variable name with contrast for clear identification
    display_labels = Vector{String}(undef, length(mr.estimates))
    
    for i in 1:length(mr.estimates)
        var_name = mr.variables[i]
        contrast = mr.terms[i]
        
        # For clean display, show "variable: contrast" format
        display_labels[i] = "$var_name: $contrast"
    end
    
    return display_labels
end

function _show_stata_table(io::IO, mr::MarginsResult)
    # Calculate confidence intervals
    alpha = get(mr.metadata, :alpha, 0.05)
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha)
    
    # Check if this is profile margins (has profile_values)
    has_profile = mr.profile_values !== nothing
    
    # Check if this is predictions (don't need Variable and Contrast columns)
    type = get(mr.metadata, :type, :effects)
    is_predictions = type === :predictions
    
    # Column widths for alignment (skip Variable/Contrast for predictions)
    var_width = if is_predictions
        0  # Skip Variable column for predictions
    else
        try
            max(8, maximum(length.(mr.variables)) + 1)
        catch e
            @warn "Display issue with variable names, using default width" exception=e
            12  # Safe default width
        end
    end
    
    contrast_width = if is_predictions
        0  # Skip Contrast column for predictions
    else
        try
            max(10, maximum(length.(mr.terms)) + 1)
        catch e
            @warn "Display issue with contrast terms, using default width" exception=e
            15  # Safe default width
        end
    end
    
    # Context column (for population margins with scenarios/groups)
    context_width = 0
    context_text = ""
    has_contexts = get(mr.metadata, :has_contexts, false) && get(mr.metadata, :analysis_type, :unknown) == :population
    if has_contexts && !is_predictions  # Only for effects, not predictions
        scenarios_vars = get(mr.metadata, :scenarios_vars, Symbol[])
        groups_vars = get(mr.metadata, :groups_vars, Symbol[])
        
        if !isempty(scenarios_vars) && !isempty(groups_vars)
            # Both scenarios and groups
            context_text = "Scenario→Group: $(join(scenarios_vars, ','))→$(join(groups_vars, ','))"
        elseif !isempty(scenarios_vars)
            # Scenarios only
            context_text = "Scenario: $(join(scenarios_vars, ','))"
        elseif !isempty(groups_vars)
            # Groups only  
            context_text = "Group: $(join(groups_vars, ','))"
        end
        context_width = max(12, length(context_text) + 1)
    end
    
    # Profile columns (for profile margins or population margins with contexts)
    profile_widths = Int[]  # Specify Int type to avoid sum() type issues
    profile_names = String[]  # Display names
    profile_raw_names = Symbol[]  # Raw names for data access
    if has_profile
        scenarios_vars = get(mr.metadata, :scenarios_vars, Symbol[])
        groups_vars = get(mr.metadata, :groups_vars, Symbol[])
        
        for (name, values) in pairs(mr.profile_values)
            # Store raw name for data access
            push!(profile_raw_names, name)
            
            # Determine proper column name based on scenarios vs groups
            display_name = if has_contexts
                # For population margins with contexts, use proper naming
                if Symbol(name) in scenarios_vars
                    "at_$(name)"  # Scenarios use at_ prefix
                else
                    string(name)  # Groups use no prefix  
                end
            else
                # For regular profile margins, use as-is
                string(name)
            end
            
            push!(profile_names, display_name)
            # Calculate width needed for this profile column
            max_val_width = try
                maximum(length.(string.(values)))
            catch e
                6  # fallback width
            end
            header_width = length(display_name)
            col_width = max(7, header_width + 1)  # Ensure header fits
            push!(profile_widths, col_width)
        end
    end
    
    num_width = 12
    
    # Calculate total width (account for skipped columns in predictions and Context column)
    var_contrast_width = if is_predictions 
        0  # Skip both Variable and Contrast columns
    else
        var_width + contrast_width + 1  # +1 for space between columns
    end
    context_width_with_space = has_contexts && !is_predictions ? context_width + 1 : 0
    total_width = var_contrast_width + context_width_with_space + sum(profile_widths) + 4*num_width + length(profile_widths) + 4
    println(io, "─"^total_width)
    
    # Column headers - dynamic based on type and measure
    measure = get(mr.metadata, :measure, :effect)
    
    # For predictions, always use "Prediction" regardless of measure
    if is_predictions
        header_text = "Prediction"
    else
        # For effects, use measure-specific headers
        header_text = measure === :effect ? "dy/dx" :
                      measure === :elasticity ? "eyex" :
                      measure === :semielasticity_dyex ? "dyex" :
                      measure === :semielasticity_eydx ? "eydx" : "dy/dx"
    end
    
    # Print headers (skip Variable/Contrast for predictions)
    if !is_predictions
        print(io, rpad("Variable", var_width))
        print(io, " ")
        print(io, rpad("Contrast", contrast_width))
        
        # Context column header
        if has_contexts
            print(io, " ")
            print(io, rpad("Context", context_width))
        end
    end
    
    # Profile column headers
    if has_profile
        for (i, (name, width)) in enumerate(zip(profile_names, profile_widths))
            print(io, " ")
            print(io, rpad(name, width))
        end
    end
    
    print(io, lpad(header_text, num_width))
    print(io, lpad("Std. Err.", num_width))
    print(io, lpad("[$(Int(100*(1-alpha)))% Conf.", num_width))
    print(io, lpad("Interval]", num_width))
    println(io)
    
    # Separator line  
    println(io, "─"^total_width)
    
    # Data rows
    for i in 1:length(mr.estimates)
        # Print Variable and Contrast columns only for effects (not predictions)
        if !is_predictions
            var_name = try
                mr.variables[i]
            catch e
                "undefined_var_$i"
            end
            
            contrast = try
                mr.terms[i]
            catch e
                "undefined_contrast_$i"
            end
            
            print(io, rpad(var_name, var_width))
            print(io, " ")
            print(io, rpad(contrast, contrast_width))
            
            # Context column data
            if has_contexts
                print(io, " ")
                print(io, rpad(context_text, context_width))
            end
        end
        
        # Profile values
        if has_profile
            for (j, (display_name, raw_name, width)) in enumerate(zip(profile_names, profile_raw_names, profile_widths))
                print(io, " ")
                profile_val = try
                    val = mr.profile_values[raw_name][i]
                    # Handle both numeric and string values
                    if val isa Number
                        format_number(Float64(val); profile_column=true)
                    else
                        # For categorical/string values, use as-is
                        string(val)
                    end
                catch e
                    "N/A"
                end
                print(io, rpad(profile_val, width))
            end
        end
        
        print(io, lpad(format_number(mr.estimates[i]), num_width))
        print(io, lpad(format_number(mr.standard_errors[i]), num_width))
        print(io, lpad(format_number(lower[i]), num_width))
        print(io, lpad(format_number(upper[i]), num_width))
        println(io)
    end
    
    # Bottom line
    println(io, "─"^total_width)
end

# Advanced numeric formatting for display
"""
    format_number(x::Float64; digits::Int=get_display_digits(), force_scientific::Bool=false, profile_column::Bool=false) -> String

Format a number for display with configurable precision and notation.

# Arguments
- `x`: Number to format
- `digits`: Number of significant digits (default from global setting)
- `force_scientific`: Force scientific notation
- `profile_column`: Use shorter format suitable for profile/reference values

# Formatting Rules
- Statistical results: Full precision with auto scientific notation for very small/large values
- Profile columns: Shorter format, avoid scientific notation when possible
- Auto scientific: |x| < 1e-4 or |x| >= 10^digits
"""
function format_number(x::Float64; digits::Int=get_display_digits(), 
                      force_scientific::Bool=false, profile_column::Bool=false)
    if isnan(x) || !isfinite(x)
        return string(x)
    end
    
    if x == 0.0
        return "0"
    end
    
    abs_x = abs(x)
    
    # Profile columns use consistent formatting across all values
    if profile_column
        if abs_x >= 1000 || abs_x < 0.00001
            return @sprintf("%.2g", x)  # Scientific for extreme values only
        else
            # Use consistent decimal places for all profile values in typical ranges
            # Most profile values are small (0.001-0.1 range), so use 4 decimal places consistently
            formatted = @sprintf("%.4f", x)
            
            # Remove trailing zeros for cleaner appearance
            if contains(formatted, '.')
                formatted = rstrip(formatted, '0')
                formatted = rstrip(formatted, '.')
            end
            
            return formatted
        end
    end
    
    # Auto scientific notation thresholds
    scientific_threshold_low = 10.0^(-4)
    scientific_threshold_high = 10.0^(digits)
    
    use_scientific = force_scientific || abs_x < scientific_threshold_low || abs_x >= scientific_threshold_high
    
    if use_scientific
        # Use a fixed format string based on digits
        if digits == 3
            return @sprintf("%.2e", x)
        elseif digits == 4
            return @sprintf("%.3e", x) 
        elseif digits == 5
            return @sprintf("%.4e", x)
        elseif digits == 6
            return @sprintf("%.5e", x)
        elseif digits == 8
            return @sprintf("%.7e", x)
        else
            # Fallback for other digit counts
            return @sprintf("%.4e", x)
        end
    else
        # Use fixed-point notation with appropriate precision
        if digits <= 3
            formatted = @sprintf("%.3f", x)
        elseif digits == 4
            formatted = @sprintf("%.4f", x)
        elseif digits == 5
            formatted = @sprintf("%.5f", x)
        elseif digits == 6
            formatted = @sprintf("%.6f", x)
        elseif digits >= 8
            formatted = @sprintf("%.8f", x)
        else
            formatted = @sprintf("%.5f", x)  # Default fallback
        end
        
        # Remove trailing zeros and unnecessary decimal point
        if contains(formatted, '.')
            formatted = rstrip(formatted, '0')
            formatted = rstrip(formatted, '.')
        end
        
        return formatted
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
        variable = mr.variables,  # The "x" in dy/dx
        contrast = mr.terms,      # Human-readable contrast/label
        estimate = mr.estimates,
        se = mr.standard_errors, 
        t_stat = t_stats,
        p_value = p_values
    )
    # Use new variable/contrast structure - no backward compatibility
    
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
        variable = mr.variables,  # The "x" in dy/dx
        contrast = mr.terms,
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
        variable = mr.variables,  # The "x" in dy/dx
        contrast = mr.terms,
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
    df[!, :variable] = mr.variables  # The "x" in dy/dx
    df[!, :contrast] = mr.terms  # Clean variable names
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
