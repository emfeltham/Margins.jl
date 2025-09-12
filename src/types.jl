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

Abstract type for marginal effects and predictions results.

See `EffectsResult` and `PredictionsResult` for concrete implementations.
"""
abstract type MarginsResult end

"""
    EffectsResult <: MarginsResult

Container for marginal effects results (AME, MEM, MER).

Effects represent "dy/dx for variable X" so they include variable/contrast information.

Fields:
- `estimates::Vector{Float64}`: Point estimates
- `standard_errors::Vector{Float64}`: Standard errors
- `variables::Vector{String}`: The "x" in dy/dx - which variable each row represents
- `terms::Vector{String}`: Contrast descriptions (e.g., "continuous", "treated vs control")
- `profile_values::Union{Nothing, NamedTuple}`: Reference grid values (for profile effects MEM/MER)
- `group_values::Union{Nothing, NamedTuple}`: Grouping variable values
- `gradients::Matrix{Float64}`: Parameter gradients (G matrix) for delta-method
- `metadata::Dict{Symbol, Any}`: Analysis metadata (model info, options used, etc.)

# Examples
```julia
result = population_margins(model, data; type=:effects, vars=[:x1, :x2])  # AME
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:x1])  # MEM

DataFrame(result)  # Includes variable/contrast columns
```
"""
struct EffectsResult <: MarginsResult
    estimates::Vector{Float64}
    standard_errors::Vector{Float64}
    variables::Vector{String}  # The "x" in dy/dx
    terms::Vector{String}      # Contrast descriptions
    profile_values::Union{Nothing, NamedTuple}  # For profile effects (MEM/MER)
    group_values::Union{Nothing, NamedTuple}
    gradients::Matrix{Float64}
    metadata::Dict{Symbol, Any}
end

"""
    PredictionsResult <: MarginsResult

Container for predictions results (AAP, APM, APR).

Predictions represent "fitted value at scenario" so they don't need variable/contrast information.

Fields:
- `estimates::Vector{Float64}`: Point estimates (predicted values)
- `standard_errors::Vector{Float64}`: Standard errors
- `profile_values::Union{Nothing, NamedTuple}`: Reference grid values (for profile predictions APM/APR)
- `group_values::Union{Nothing, NamedTuple}`: Grouping variable values
- `gradients::Matrix{Float64}`: Parameter gradients (G matrix) for delta-method
- `metadata::Dict{Symbol, Any}`: Analysis metadata (model info, options used, etc.)

# Examples
```julia
result = population_margins(model, data; type=:predictions)  # AAP
result = profile_margins(model, data, means_grid(data); type=:predictions)  # APM

DataFrame(result)  # No variable/contrast columns, just statistical results
```
"""
struct PredictionsResult <: MarginsResult
    estimates::Vector{Float64}
    standard_errors::Vector{Float64}
    profile_values::Union{Nothing, NamedTuple}  # For profile predictions (APM/APR)
    group_values::Union{Nothing, NamedTuple}
    gradients::Matrix{Float64}
    metadata::Dict{Symbol, Any}
end

import DataFrames: DataFrame # explicit import to extend method

# Type-specific DataFrame dispatch for effects
function DataFrame(mr::EffectsResult; format::Symbol=:auto)
    # Auto-detect natural format based on analysis type
    if format == :auto
        analysis_type = get(mr.metadata, :analysis_type, :unknown)
        # For effects, use profile format for profile analysis, standard otherwise
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
        error("Unknown format: $format. EffectsResult supports: :standard, :compact, :confidence, :profile, :stata")
    end
end

# Type-specific DataFrame dispatch for predictions
function DataFrame(mr::PredictionsResult; format::Symbol=:auto)
    # Auto-detect natural format - always use predictions format
    if format == :auto
        format = :predictions
    end
    
    # Predictions only support the predictions format
    if format == :predictions
        return _predictions_table(mr)
    else
        error("Unknown format: $format. PredictionsResult only supports: :predictions (or :auto)")
    end
end

# Tables.jl interface implementation - uses auto format
Tables.istable(::Type{<:MarginsResult}) = true
Tables.rowaccess(::Type{<:MarginsResult}) = true
Tables.rows(mr::MarginsResult) = Tables.rows(DataFrame(mr))
Tables.schema(mr::MarginsResult) = Tables.schema(DataFrame(mr))

# DataFrame conversion compatibility
Base.convert(::Type{DataFrame}, mr::MarginsResult) = DataFrame(mr)

# Stata-style display methods for effects
function Base.show(io::IO, mr::EffectsResult)
    n_results = length(mr.estimates)
    analysis_type = get(mr.metadata, :analysis_type, :unknown)
    
    # Header
    n_obs = get(mr.metadata, :n_obs, missing)
    n_text = ismissing(n_obs) ? "" : " (N=$n_obs)"
    
    if analysis_type == :profile
        n_profiles = get(mr.metadata, :n_profiles, 1)
        println(io, "EffectsResult: $n_results effects at $n_profiles profiles$n_text")
    else
        println(io, "EffectsResult: $n_results population effects$n_text")  
    end
    
    # Optional context header for population-with-contexts
    _show_context_header(io, mr)
    
    # Stata-style table with horizontal lines
    _show_stata_table(io, mr)
end

# Stata-style display methods for predictions
function Base.show(io::IO, mr::PredictionsResult)
    n_results = length(mr.estimates)
    analysis_type = get(mr.metadata, :analysis_type, :unknown)
    
    # Header
    n_obs = get(mr.metadata, :n_obs, missing)
    n_text = ismissing(n_obs) ? "" : " (N=$n_obs)"
    
    if analysis_type == :profile
        n_profiles = get(mr.metadata, :n_profiles, 1)
        println(io, "PredictionsResult: $n_results predictions at $n_profiles profiles$n_text")
    else
        println(io, "PredictionsResult: $n_results population predictions$n_text")  
    end
    
    # Optional context header for population-with-contexts
    _show_context_header(io, mr)
    
    # Stata-style table with horizontal lines
    _show_stata_table(io, mr)
end

function Base.show(io::IO, ::MIME"text/plain", mr::EffectsResult)
    show(io, mr)
    # unpleasant to show this every time?
    # println(io, "\nUse DataFrame(result; format=...) for different table formats")
end

function Base.show(io::IO, ::MIME"text/plain", mr::PredictionsResult)
    show(io, mr)
    # unpleasant to show this every time?
    # println(io, "\nUse DataFrame(result; format=...) for different table formats")
end

# Helper function to generate clean display labels for effects show method
function _generate_clean_display_labels(mr::EffectsResult)
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

# Predictions don't need display labels since they don't have variable/contrast concepts
_generate_clean_display_labels(mr::PredictionsResult) = String[]

"""
    _show_context_header(io::IO, mr::MarginsResult)

Show optional context header for population-with-contexts cases.
Displays groups and scenarios derived from metadata in format: "Groups: a, b; Scenarios: x, y"

Only shows when:
- `has_contexts == true` 
- `analysis_type == :population`
- There are actually groups or scenarios to display

Kept behind a guard to avoid noisy output for simple cases.
"""
function _show_context_header(io::IO, mr::MarginsResult)
    has_contexts = get(mr.metadata, :has_contexts, false)
    analysis_type = get(mr.metadata, :analysis_type, :unknown)
    
    # Only show for population-with-contexts
    if !(has_contexts && analysis_type == :population)
        return
    end
    
    groups, scenarios = context_columns(mr)
    
    # Only show if there are groups or scenarios to display
    if isempty(groups) && isempty(scenarios)
        return
    end
    
    parts = String[]
    
    if !isempty(groups)
        group_names = join(string.(groups), ", ")
        push!(parts, "Groups: $group_names")
    end
    
    if !isempty(scenarios)
        scenario_names = join(string.(scenarios), ", ")  
        push!(parts, "Scenarios: $scenario_names")
    end
    
    if !isempty(parts)
        println(io, join(parts, "; "))
    end
end

"""
    context_columns(mr::MarginsResult) -> (groups::Vector{Symbol}, scenarios::Vector{Symbol})

Extract group and scenario column names from a MarginsResult, identifying which columns in 
`profile_values` represent groups vs scenarios based on metadata.

Returns two vectors:
- `groups`: Variable names that represent grouping (e.g., from `over` parameter)
- `scenarios`: Variable names that represent scenarios (e.g., from `at` parameter)

Only includes variables that are actually present in `mr.profile_values`.

# Examples
```julia
result = population_margins(model, data; over=:region, at=Dict(:x => [0, 1]))
groups, scenarios = context_columns(result)
# groups = [:region], scenarios = [:x]

result = profile_margins(model, data, means_grid(data))
groups, scenarios = context_columns(result) 
# groups = Symbol[], scenarios = Symbol[] (profile analysis has no contexts)
```
"""
function context_columns(mr::MarginsResult)
    # Get metadata variables
    groups_vars = get(mr.metadata, :groups_vars, Symbol[])
    scenarios_vars = get(mr.metadata, :scenarios_vars, Symbol[])
    
    # Only include variables that are actually present in profile_values
    if mr.profile_values === nothing
        return Symbol[], Symbol[]
    end
    
    profile_keys = Set(keys(mr.profile_values))
    
    # Filter to only include variables present in the actual result
    groups = Symbol[var for var in groups_vars if var in profile_keys]
    scenarios = Symbol[var for var in scenarios_vars if var in profile_keys]
    
    return groups, scenarios
end

"""
    append_context_columns!(df::DataFrame, mr::MarginsResult; order=:front)

Add context columns (groups and scenarios) to a DataFrame with consistent naming and placement.

Implements the core naming convention:
- **Population contexts** (`mr.metadata[:has_contexts] == true`):
  - Group columns: unprefixed names (e.g., `region`)
  - Scenario columns: `at_<var>` prefix (e.g., `at_x`)
- **Profile analysis** (`mr.metadata[:analysis_type] == :profile`):
  - All profile/grid columns: bare names (e.g., `x`, `z`) with no prefixes

Dispatches on specific result types for type-safe behavior.

# Arguments
- `df::DataFrame`: DataFrame to modify in-place
- `mr::MarginsResult`: Result containing context information (dispatches on EffectsResult/PredictionsResult)
- `order=:front`: Place context columns at the front (`:front`) or back (`:back`)

# Examples
```julia
# Population with contexts: groups unprefixed, scenarios prefixed
result = population_margins(model, data; over=:region, at=Dict(:x => [0, 1]))
df = DataFrame()
append_context_columns!(df, result)
# Adds columns: region, at_x (groups first, then scenarios)

# Profile analysis: bare profile names
result = profile_margins(model, data, means_grid(data))
df = DataFrame()  
append_context_columns!(df, result)
# Adds columns: x, z (bare profile variable names)
```
"""
function append_context_columns!(df::DataFrame, mr::MarginsResult; order::Symbol=:front)
    if mr.profile_values === nothing
        return df  # No context columns to add
    end
    
    has_contexts = get(mr.metadata, :has_contexts, false)
    analysis_type = get(mr.metadata, :analysis_type, :unknown)
    
    if has_contexts && analysis_type == :population
        # Population contexts: distinguish groups from scenarios
        groups, scenarios = context_columns(mr)
        
        # Add group columns first (unprefixed)
        for group_var in groups
            values = mr.profile_values[group_var]
            col_name = group_var  # No prefix for groups
            
            if order == :front
                DataFrames.insertcols!(df, 1, col_name => values)
            else
                df[!, col_name] = values
            end
        end
        
        # Add scenario columns next (with at_ prefix)
        for scenario_var in scenarios
            values = mr.profile_values[scenario_var]
            col_name = Symbol("at_", scenario_var)  # at_ prefix for scenarios
            
            if order == :front
                # Insert after existing columns (groups are already added)
                insert_pos = length(groups) + 1
                DataFrames.insertcols!(df, insert_pos, col_name => values)
            else
                df[!, col_name] = values
            end
        end
        
    elseif analysis_type == :profile
        # Profile analysis: bare profile/grid column names (no prefixes)
        for (var_name, values) in pairs(mr.profile_values)
            col_name = var_name  # Bare variable name
            
            if order == :front
                DataFrames.insertcols!(df, 1, col_name => values)
            else
                df[!, col_name] = values
            end
        end
        
    else
        # Fallback: treat all as scenarios with at_ prefix
        for (var_name, values) in pairs(mr.profile_values)
            col_name = Symbol("at_", var_name)
            
            if order == :front
                DataFrames.insertcols!(df, 1, col_name => values)
            else
                df[!, col_name] = values
            end
        end
    end
    
    return df
end

function _show_stata_table(io::IO, mr::EffectsResult)
    # Calculate confidence intervals
    alpha = get(mr.metadata, :alpha, 0.05)
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha)
    
    # Check if this is profile margins (has profile_values)
    has_profile = mr.profile_values !== nothing
    
    # Column widths for alignment (effects always have Variable/Contrast columns)
    var_width = try
        max(8, maximum(length.(mr.variables)) + 1)
    catch e
        @warn "Display issue with variable names, using default width" exception=e
        12  # Safe default width
    end
    
    contrast_width = try
        max(10, maximum(length.(mr.terms)) + 1)
    catch e
        @warn "Display issue with contrast terms, using default width" exception=e
        15  # Safe default width
    end
    
    # Context information is already conveyed by at_* column headers, so no separate Context column needed
    has_contexts = get(mr.metadata, :has_contexts, false) && get(mr.metadata, :analysis_type, :unknown) == :population
    
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
            col_width = max(7, max(header_width, max_val_width) + 1)  # Ensure both header and values fit
            push!(profile_widths, col_width)
        end
    end
    
    num_width = 12
    
    # Calculate total width (effects always have Variable and Contrast columns, no Context column)
    var_contrast_width = var_width + contrast_width + 1  # +1 for space between columns
    total_width = var_contrast_width + sum(profile_widths) + 4*num_width + length(profile_widths) + 4
    println(io, "─"^total_width)
    
    # Column headers - dynamic based on measure (effects)
    measure = get(mr.metadata, :measure, :effect)
    
    # For effects, use measure-specific headers
    header_text = measure === :effect ? "dy/dx" :
                  measure === :elasticity ? "eyex" :
                  measure === :semielasticity_dyex ? "dyex" :
                  measure === :semielasticity_eydx ? "eydx" : "dy/dx"
    
    # Print headers (effects always have Variable/Contrast)
    print(io, rpad("Variable", var_width))
    print(io, " ")
    print(io, rpad("Contrast", contrast_width))
    
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
        # Print Variable and Contrast columns (effects always have these)
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

function _show_stata_table(io::IO, mr::PredictionsResult)
    # Calculate confidence intervals
    alpha = get(mr.metadata, :alpha, 0.05)
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha)
    
    # Check if this is profile margins (has profile_values)
    has_profile = mr.profile_values !== nothing
    
    # Profile columns (for profile margins or population margins with contexts)
    profile_widths = Int[]
    profile_names = String[]
    profile_raw_names = Symbol[]
    if has_profile
        scenarios_vars = get(mr.metadata, :scenarios_vars, Symbol[])
        groups_vars = get(mr.metadata, :groups_vars, Symbol[])
        
        for (name, values) in pairs(mr.profile_values)
            push!(profile_raw_names, name)
            
            # For predictions, use clean column names (no at_ prefix)
            display_name = string(name)
            push!(profile_names, display_name)
            
            # Calculate width needed for this profile column
            max_val_width = try
                maximum(length.(string.(values)))
            catch e
                6  # fallback width
            end
            header_width = length(display_name)
            col_width = max(7, max(header_width, max_val_width) + 1)
            push!(profile_widths, col_width)
        end
    end
    
    num_width = 12
    
    # Calculate total width (no Variable/Contrast columns for predictions)
    total_width = sum(profile_widths) + 4*num_width + length(profile_widths) + 4
    println(io, "─"^total_width)
    
    # Column headers - always "Prediction" for predictions
    header_text = "Prediction"
    
    # Profile column headers
    if has_profile
        for (name, width) in zip(profile_names, profile_widths)
            print(io, rpad(name, width))
            print(io, " ")
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
        # Profile values
        if has_profile
            for (display_name, raw_name, width) in zip(profile_names, profile_raw_names, profile_widths)
                profile_val = try
                    val = mr.profile_values[raw_name][i]
                    if val isa Number
                        format_number(Float64(val); profile_column=true)
                    else
                        string(val)
                    end
                catch e
                    "N/A"
                end
                print(io, rpad(profile_val, width))
                print(io, " ")
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

# Helper function to create self-describing type column
function _create_type_description(analysis_type::Symbol, measure::Symbol, result_category::Symbol)
    # Create concise but descriptive type labels
    if result_category == :effects
        if analysis_type == :population
            # Population effects (AME)
            if measure == :effect
                return "AME"  # Average Marginal Effect
            elseif measure == :elasticity
                return "AME (elasticity)"
            elseif measure == :semielasticity_dyex
                return "AME (semi-elasticity dyex)"
            elseif measure == :semielasticity_eydx
                return "AME (semi-elasticity eydx)"
            else
                return "AME"
            end
        else  # profile
            # Profile effects (MEM/MER)
            if measure == :effect
                return "MEM"  # Marginal Effect at the Mean / Marginal Effect at Reference
            elseif measure == :elasticity
                return "MEM (elasticity)"
            elseif measure == :semielasticity_dyex
                return "MEM (semi-elasticity dyex)"
            elseif measure == :semielasticity_eydx
                return "MEM (semi-elasticity eydx)"
            else
                return "MEM"
            end
        end
    else  # predictions
        if analysis_type == :population
            return "AAP"  # Average Adjusted Prediction
        else  # profile
            return "APM"  # Adjusted Prediction at the Mean / Adjusted Prediction at Reference
        end
    end
end

# Format-specific table builders

function _standard_table(mr::EffectsResult)
    t_stats = mr.estimates ./ mr.standard_errors
    p_values = 2 .* (1 .- cdf.(Normal(), abs.(t_stats)))
    
    # Create self-describing type column
    analysis_type = get(mr.metadata, :analysis_type, :population)
    measure = get(mr.metadata, :measure, :effect)
    type_description = _create_type_description(analysis_type, measure, :effects)
    
    # Build DataFrame with new column order: context columns first
    df = DataFrame()
    append_context_columns!(df, mr; order=:front)
    
    # Add type column
    df[!, :type] = fill(type_description, length(mr.estimates))
    
    # Add statistical columns
    df[!, :variable] = mr.variables  # The "x" in dy/dx
    df[!, :contrast] = mr.terms      # Human-readable contrast/label
    df[!, :estimate] = mr.estimates
    df[!, :se] = mr.standard_errors
    df[!, :t_stat] = t_stats
    df[!, :p_value] = p_values
    
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
    
    return df
end

function _compact_table(mr::EffectsResult)
    # Create self-describing type column
    analysis_type = get(mr.metadata, :analysis_type, :population)
    measure = get(mr.metadata, :measure, :effect)
    type_description = _create_type_description(analysis_type, measure, :effects)
    
    # Build DataFrame with new column order: context columns first
    df = DataFrame()
    append_context_columns!(df, mr; order=:front)
    
    # Add type column
    df[!, :type] = fill(type_description, length(mr.estimates))
    
    # Add statistical columns
    df[!, :variable] = mr.variables  # The "x" in dy/dx
    df[!, :contrast] = mr.terms
    df[!, :estimate] = mr.estimates
    df[!, :se] = mr.standard_errors
    
    return df
end

function _confidence_table(mr::EffectsResult)
    alpha = get(mr.metadata, :alpha, 0.05)
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha)
    
    # Create self-describing type column
    analysis_type = get(mr.metadata, :analysis_type, :population)
    measure = get(mr.metadata, :measure, :effect)
    type_description = _create_type_description(analysis_type, measure, :effects)
    
    # Build DataFrame with new column order: context columns first
    df = DataFrame()
    append_context_columns!(df, mr; order=:front)
    
    # Add type column
    df[!, :type] = fill(type_description, length(mr.estimates))
    
    # Add statistical columns
    df[!, :variable] = mr.variables  # The "x" in dy/dx
    df[!, :contrast] = mr.terms
    df[!, :estimate] = mr.estimates
    df[!, :lower] = lower
    df[!, :upper] = upper
    
    return df
end

function _profile_table(mr::EffectsResult)
    # Profile-first organization: reference grid with results attached
    alpha = get(mr.metadata, :alpha, 0.05)
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha)
    
    # Create self-describing type column
    analysis_type = get(mr.metadata, :analysis_type, :population)
    measure = get(mr.metadata, :measure, :effect)
    type_description = _create_type_description(analysis_type, measure, :effects)
    
    # Build DataFrame with new column order: context columns first (using consistent helper)
    df = DataFrames.DataFrame()
    append_context_columns!(df, mr; order=:front)
    
    # Add type column after profile columns but before result columns
    df[!, :type] = fill(type_description, length(mr.estimates))
    
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
    
    return df
end

function _predictions_table(mr::PredictionsResult)
    # Predictions-specific format: omits variable/contrast columns
    t_stats = mr.estimates ./ mr.standard_errors
    p_values = 2 .* (1 .- cdf.(Normal(), abs.(t_stats)))
    
    # Create self-describing type column
    analysis_type = get(mr.metadata, :analysis_type, :population)
    type_description = _create_type_description(analysis_type, :prediction, :predictions)
    
    # Build DataFrame with new column order: context columns first (using consistent helper)
    df = DataFrame()
    append_context_columns!(df, mr; order=:front)
    
    # Add type column after context columns but before statistical columns
    df[!, :type] = fill(type_description, length(mr.estimates))
    
    # Core statistical columns (no variable/contrast)
    df[!, :estimate] = mr.estimates
    df[!, :se] = mr.standard_errors
    df[!, :t_stat] = t_stats
    df[!, :p_value] = p_values
    
    # Add confidence intervals if alpha is specified in metadata
    if haskey(mr.metadata, :alpha)
        alpha = mr.metadata[:alpha]
        lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha)
        df[!, :ci_lower] = lower
        df[!, :ci_upper] = upper
    end
    
    # Add sample size
    if haskey(mr.metadata, :has_subgroup_n) && haskey(mr.metadata, :subgroup_n_values)
        # Grouped case: use the actual subgroup sizes from computation
        df[!, :n] = mr.metadata[:subgroup_n_values]
    else
        # Simple case: use overall sample size  
        n_obs = get(mr.metadata, :n_obs, missing)
        df[!, :n] = fill(n_obs, length(mr.estimates))
    end
    
    return df
end

function _stata_table(mr::EffectsResult)
    t_stats = mr.estimates ./ mr.standard_errors  
    p_values = 2 .* (1 .- cdf.(Normal(), abs.(t_stats)))
    
    # Create self-describing type column
    analysis_type = get(mr.metadata, :analysis_type, :population)
    measure = get(mr.metadata, :measure, :effect)
    type_description = _create_type_description(analysis_type, measure, :effects)
    
    # Get sample size from metadata
    n_obs = get(mr.metadata, :n_obs, missing)
    
    # Build DataFrame with new column order: context columns first (using consistent helper)
    df = DataFrame()
    append_context_columns!(df, mr; order=:front)
    
    # Add type column
    df[!, :type] = fill(type_description, length(mr.estimates))
    
    # Add Stata-style statistical columns
    df[!, :margin] = mr.estimates      # Stata uses "margin" not "estimate"
    df[!, :std_err] = mr.standard_errors # "std_err" not "se" 
    df[!, :t] = t_stats
    df[!, :P_t] = p_values            # "P>|t|" equivalent
    df[!, :N] = fill(n_obs, length(mr.estimates))  # Add sample size (Stata uses uppercase N)
    
    return df
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
