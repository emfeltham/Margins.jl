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
    Inference(vcov, df, alpha, dropped=Int[])

How the standard errors, tests and intervals of a result were computed. Every
`EffectsResult` and `PredictionsResult` carries one (`result.inference`), set from the call
that produced it, and everything computed from the result afterwards (`DataFrame(result)`,
`contrast`, `second_differences`) reads it, so they cannot disagree with the result.

Fields:
- `vcov::Matrix{Float64}`: the parameter covariance matrix Σ the delta method used (from the
  `vcov=` keyword of the call, `GLM.vcov(model)` by default). With `gᵢ = result.gradients[i, :]`,
  the standard error of row `i` is `√(gᵢ'Σgᵢ)` and the covariance of rows `i` and `j` is
  `gᵢ'Σgⱼ`; `contrast` and `second_differences` use it for the standard error of a difference.
- `df::Float64`: degrees of freedom of the reference distribution: t(`df`), or z when
  `df == Inf` (the `df=` keyword of the call, else the model class's rule; see
  `population_margins`)
- `alpha::Float64`: significance level of the intervals, in (0, 1) (the `alpha=` keyword)
- `dropped::Vector{Int}`: the coefficients the fit dropped as collinear (a rank-deficient fit;
  empty otherwise). They are treated as fixed at 0: their rows and columns of `vcov` are 0. A
  row (or a contrast or second difference of rows) whose gradient is nonzero only in dropped
  coefficients is not identified and has a NaN standard error (see `population_margins`,
  *Rank-deficient fits*).

To use a different covariance estimator, pass `vcov=` to `population_margins` /
`profile_margins` / `second_differences`; a result is not re-used with another Σ.
"""
struct Inference
    vcov::Matrix{Float64}
    df::Float64
    alpha::Float64
    dropped::Vector{Int}
    function Inference(vcov::Matrix{Float64}, df::Float64, alpha::Float64, dropped::Vector{Int}=Int[])
        size(vcov, 1) == size(vcov, 2) ||
            throw(ArgumentError("Inference: vcov must be square, got size $(size(vcov))"))
        df > 0 || throw(ArgumentError("Inference: df must be positive (Inf for z), got $df"))
        0 < alpha < 1 || throw(ArgumentError("Inference: alpha must be in (0, 1), got $alpha"))
        all(j -> 1 <= j <= size(vcov, 1), dropped) ||
            throw(ArgumentError("Inference: dropped coefficient indices must be in 1:$(size(vcov, 1)), got $dropped"))
        return new(vcov, df, alpha, dropped)
    end
end
Inference(vcov::AbstractMatrix{<:Real}, df::Real, alpha::Real, dropped::AbstractVector{<:Integer}=Int[]) =
    Inference(Matrix{Float64}(vcov), Float64(df), Float64(alpha), Vector{Int}(dropped))

"""
    ResultContext(columns::NamedTuple, kinds::Vector{Symbol})

The per-row context columns of a result: what each row was evaluated at or over
(`result.context`; `nothing` for a plain AME/AAP).

Fields:
- `columns::NamedTuple`: one vector per context variable, one entry per result row, in output
  order: for `profile_margins`, the reference grid's columns in the grid's order (the columns
  the user gave, without the response; typical values filled in are not shown); for
  `population_margins`, the `groups=` variables in the order given, then the `scenarios=`
  variables in the order given
- `kinds::Vector{Symbol}`: the kind of each column, in the same order: `:grid` (a reference-grid
  column; bare name in `DataFrame(result)`), `:group` (a `groups=` variable; `over_<var>`) or
  `:scenario` (a `scenarios=` variable; `at_<var>`)

`keys(context)`, `context[name]` and `pairs(context)` read the columns;
`Margins.context_kind(context, name)` gives a column's kind.
"""
struct ResultContext
    columns::NamedTuple
    kinds::Vector{Symbol}
    function ResultContext(columns::NamedTuple, kinds::Vector{Symbol})
        length(kinds) == length(columns) || throw(ArgumentError(
            "ResultContext: $(length(columns)) columns but $(length(kinds)) kinds"))
        for k in kinds
            k in (:grid, :scenario, :group) || throw(ArgumentError(
                "ResultContext: kind must be :grid, :scenario or :group, got $(repr(k))"))
        end
        isempty(columns) || allequal(map(length, values(columns))) || throw(ArgumentError(
            "ResultContext: columns have different lengths $(map(length, columns))"))
        return new(columns, kinds)
    end
end

Base.keys(ctx::ResultContext) = keys(ctx.columns)
Base.getindex(ctx::ResultContext, name::Symbol) = ctx.columns[name]
Base.haskey(ctx::ResultContext, name::Symbol) = haskey(ctx.columns, name)
Base.pairs(ctx::ResultContext) = pairs(ctx.columns)
Base.length(ctx::ResultContext) = length(ctx.columns)
Base.isempty(ctx::ResultContext) = isempty(ctx.columns)
Base.:(==)(a::ResultContext, b::ResultContext) = a.columns == b.columns && a.kinds == b.kinds

"""
    context_kind(ctx::ResultContext, name::Symbol) -> Symbol

The kind of context column `name`: `:grid`, `:group` or `:scenario`.
"""
function context_kind(ctx::ResultContext, name::Symbol)
    i = findfirst(==(name), keys(ctx.columns))
    isnothing(i) && throw(ArgumentError("No context column :$name (columns: $(join(keys(ctx), ", ")))"))
    return ctx.kinds[i]
end

"""
    EffectsResult <: MarginsResult

Container for marginal effects results (AME, MEM).

Effects represent "dy/dx for variable X" so they include variable/contrast information.

Fields:
- `estimates::Vector{Float64}`: Point estimates
- `standard_errors::Vector{Float64}`: Standard errors
- `variables::Vector{String}`: The "x" in dy/dx - which variable each row represents
- `terms::Vector{String}`: Contrast descriptions (e.g., "dy/dx", "treated - control")
- `context::Union{Nothing, ResultContext}`: Per-row context columns: the reference grid (profile effects MEM/MER) or the `groups=` and `scenarios=` variables (population effects), in output order, with the kind of each column; `nothing` for a plain AME (see [`ResultContext`](@ref))
- `gradients::Matrix{Float64}`: Parameter gradients (G matrix) for delta-method
- `inference::Inference`: the covariance matrix, degrees of freedom and `alpha` behind the standard errors and intervals (see [`Inference`](@ref))
- `metadata::Dict{Symbol, Any}`: Analysis metadata (model info, options used, …). The keys every result carries, with their types, are listed by `Margins.result_metadata_keys()`

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
    context::Union{Nothing, ResultContext}  # grid / groups / scenarios columns
    gradients::Matrix{Float64}
    inference::Inference
    metadata::Dict{Symbol, Any}
end

"""
    PredictionsResult <: MarginsResult

Container for predictions results (AAP, APM, APR).

Predictions represent "fitted value at scenario" so they don't need variable/contrast information.

Fields:
- `estimates::Vector{Float64}`: Point estimates (predicted values)
- `standard_errors::Vector{Float64}`: Standard errors
- `context::Union{Nothing, ResultContext}`: Per-row context columns: the reference grid (profile predictions APM/APR) or the `groups=` and `scenarios=` variables (population predictions), in output order, with the kind of each column; `nothing` for a plain AAP (see [`ResultContext`](@ref))
- `gradients::Matrix{Float64}`: Parameter gradients (G matrix) for delta-method
- `inference::Inference`: the covariance matrix, degrees of freedom and `alpha` behind the standard errors and intervals (see [`Inference`](@ref))
- `metadata::Dict{Symbol, Any}`: Analysis metadata (model info, options used, …). The keys every result carries, with their types, are listed by `Margins.result_metadata_keys()`

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
    context::Union{Nothing, ResultContext}  # grid / groups / scenarios columns
    gradients::Matrix{Float64}
    inference::Inference
    metadata::Dict{Symbol, Any}
end

"""
    ContrastResult

Container for contrast analysis results with delta-method inference.

Returned by `contrast(result, …)` when comparing two estimates (predictions or effects) of
a result. Provides complete statistical inference for the difference between two rows, with
the result's own covariance matrix, degrees of freedom and (by default) `alpha`
(`result.inference`).

# Fields
- `contrast::Float64`: Contrast estimate (difference between estimates)
- `se::Float64`: Standard error of the contrast
- `statistic::Float64`: Test statistic `contrast / se` (t when `df` is finite, z when `df == Inf`)
- `df::Float64`: Inference degrees of freedom (`Inf` for z; see `population_margins`)
- `p_value::Float64`: Two-tailed p-value
- `ci_lower::Float64`: Lower confidence interval bound
- `ci_upper::Float64`: Upper confidence interval bound
- `estimate1::Float64`: First estimate value
- `estimate2::Float64`: Second estimate value
- `gradient::Vector{Float64}`: Contrast gradient vector for delta-method
- `row1::Union{Int, Nothing}`: First row index (if applicable)
- `row2::Union{Int, Nothing}`: Second row index (if applicable)
- `metadata::Dict{Symbol, Any}`: Additional information (α, analysis description, etc.)

# Examples
```julia
# Profile predictions contrast
pred_result = profile_margins(model, data, cartesian_grid(x=[0, 1]); type=:predictions)
cr = contrast(pred_result, 1, 2)
println("Contrast: ", cr.contrast, " ± ", cr.se)

# Convert to DataFrame
df = DataFrame(cr)

# Access fields
println("p-value: ", cr.p_value)
println("95% CI: [", cr.ci_lower, ", ", cr.ci_upper, "]")
```

# See Also
- `contrast(::MarginsResult, ...)`: Compute contrasts from result objects
- `DataFrame(::ContrastResult)`: Convert to DataFrame
"""
struct ContrastResult
    contrast::Float64
    se::Float64
    statistic::Float64
    df::Float64
    p_value::Float64
    ci_lower::Float64
    ci_upper::Float64
    estimate1::Float64
    estimate2::Float64
    gradient::Vector{Float64}
    row1::Union{Int, Nothing}
    row2::Union{Int, Nothing}
    metadata::Dict{Symbol, Any}
end

import DataFrames: DataFrame # explicit import to extend method

# Type-specific DataFrame dispatch for effects
function DataFrame(mr::EffectsResult; format::Symbol=:auto, include_gradients::Bool=false)
    # Auto-detect natural format based on analysis type
    if format == :auto
        analysis_type = _result_analysis_type(mr, :unknown)
        # For effects, use profile format for profile analysis, standard otherwise
        format = analysis_type == :profile ? :profile : :standard
    end

    # Validate format compatibility
    if format == :profile && isnothing(mr.context)
        throw(ArgumentError("format=:profile requires profile-based results (from profile_margins)"))
    end

    if format == :standard
        return _standard_table(mr; include_gradients)
    elseif format == :compact
        return _compact_table(mr; include_gradients)
    elseif format == :confidence
        return _confidence_table(mr; include_gradients)
    elseif format == :profile
        return _profile_table(mr; include_gradients)
    elseif format == :stata
        return _stata_table(mr; include_gradients)
    else
        error("Unknown format: $format. EffectsResult supports: :standard, :compact, :confidence, :profile, :stata")
    end
end

# Type-specific DataFrame dispatch for predictions
function DataFrame(mr::PredictionsResult; format::Symbol=:auto, include_gradients::Bool=false)
    # Auto-detect natural format - always use predictions format
    if format == :auto
        format = :predictions
    end

    # Predictions only support the predictions format
    if format == :predictions
        return _predictions_table(mr; include_gradients)
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
    analysis_type = _result_analysis_type(mr, :unknown)
    
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
    _show_random_effects_header(io, mr)
    
    # Stata-style table with horizontal lines
    _show_stata_table(io, mr)
end

# Stata-style display methods for predictions
function Base.show(io::IO, mr::PredictionsResult)
    n_results = length(mr.estimates)
    analysis_type = _result_analysis_type(mr, :unknown)
    
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
    _show_random_effects_header(io, mr)
    
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

# Display method for ContrastResult
function Base.show(io::IO, cr::ContrastResult)
    # Header
    println(io, "ContrastResult:")

    # Get alpha from metadata if available
    alpha = _result_alpha(cr)
    ci_pct = _format_level_pct(alpha)

    # Display width for alignment
    num_width = 14

    # Top border
    total_width = 60
    println(io, "─"^total_width)

    # Main statistics
    println(io, rpad("  Contrast:", 25), lpad(format_number(cr.contrast), num_width))
    println(io, rpad("  Std. Error:", 25), lpad(format_number(cr.se), num_width))
    stat_label = isfinite(cr.df) ? "  t (df = $(_format_df(cr.df))):" : "  z:"
    println(io, rpad(stat_label, 25), lpad(format_number(cr.statistic), num_width))
    println(io, rpad("  p-value:", 25), lpad(format_number(cr.p_value), num_width))
    println(io, rpad("  $(ci_pct)% CI:", 25), "[", format_number(cr.ci_lower), ", ", format_number(cr.ci_upper), "]")

    # Separator
    println(io, "─"^total_width)

    # Component estimates
    println(io, rpad("  Estimate 1:", 25), lpad(format_number(cr.estimate1), num_width))
    println(io, rpad("  Estimate 2:", 25), lpad(format_number(cr.estimate2), num_width))

    # Row indices if available
    if !isnothing(cr.row1) && !isnothing(cr.row2)
        println(io, rpad("  Rows compared:", 25), "$(cr.row1) - $(cr.row2)")
    end

    # Bottom border
    println(io, "─"^total_width)
end

function Base.show(io::IO, ::MIME"text/plain", cr::ContrastResult)
    show(io, cr)
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
    _show_random_effects_header(io::IO, mr::MarginsResult)

On results for a model with random effects (`metadata[:re_factors]` is set), one line naming how
they entered the estimates, so that a typical-group estimate (random effects 0) is not read as a
population-averaged one or as one for the groups in the data.
"""
function _show_random_effects_header(io::IO, mr::MarginsResult)
    factors = get(mr.metadata, :re_factors, nothing)
    isnothing(factors) && return
    used = Symbol[get(mr.metadata, :re_terms, Symbol[])...]
    zeroed = setdiff(factors, used)
    parts = String[]
    if get(mr.metadata, :random_effects, :zero) === :marginal
        sd = get(mr.metadata, :re_sd, NaN)
        isempty(used) || push!(parts, "averaged over " * join(string.(used), ", ") *
                                      " (population-averaged; SD $(round(sd; sigdigits=3)))")
    else
        isempty(used) || push!(parts, "predicted for " * join(string.(used), ", "))
    end
    isempty(zeroed) || push!(parts, "0 for " * join(string.(zeroed), ", ") *
                                      (isempty(used) ? " (a typical group)" : ""))
    println(io, "Random effects: ", join(parts, "; "))
end

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
    has_contexts = _result_has_contexts(mr)
    analysis_type = _result_analysis_type(mr, :unknown)
    
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

The group and scenario columns of a result's context (`mr.context`), each in the order the
`groups=` / `scenarios=` variables were given.

These are the raw variable names; in `DataFrame(result)` the corresponding columns are
`over_<var>` (groups) and `at_<var>` (scenarios). Reference-grid columns (`profile_margins`) are
neither.

# Examples
```julia
result = population_margins(model, data; groups=:region, scenarios=(x=[0, 1],))
groups, scenarios = context_columns(result)
# groups = [:region], scenarios = [:x]

result = profile_margins(model, data, means_grid(data))
groups, scenarios = context_columns(result)
# groups = Symbol[], scenarios = Symbol[] (profile analysis has no contexts)
```
"""
function context_columns(mr::MarginsResult)
    ctx = mr.context
    isnothing(ctx) && return Symbol[], Symbol[]
    names_ = collect(Symbol, keys(ctx))
    groups = Symbol[n for (n, k) in zip(names_, ctx.kinds) if k === :group]
    scenarios = Symbol[n for (n, k) in zip(names_, ctx.kinds) if k === :scenario]
    return groups, scenarios
end

# Output column names of population context variables (DataFrame and `show`)
_group_column_name(var) = Symbol("over_", var)
_scenario_column_name(var) = Symbol("at_", var)

# Output name of a context column of the given kind: over_ for groups, at_ for scenarios,
# bare for reference-grid columns
_context_column_name(name::Symbol, kind::Symbol) =
    kind === :group ? _group_column_name(name) :
    kind === :scenario ? _scenario_column_name(name) : name

"""
    append_context_columns!(df::DataFrame, mr::MarginsResult; order=:front)

Add the context columns of `mr` (`mr.context`) to a DataFrame, in the context's order, with
the output naming convention:
- reference-grid columns (`profile_margins`): bare names (e.g. `x`, `z`)
- `groups=` columns: `over_<var>` (e.g. `over_region`), as in Stata's `over()` (this covers
  plain, binned and nested groups)
- `scenarios=` columns: `at_<var>` (e.g. `at_x`)

Population results list the group columns first, then the scenario columns, each in the order
given in the call; profile results list the grid columns in the grid's order.

# Arguments
- `df::DataFrame`: DataFrame to modify in-place
- `mr::MarginsResult`: Result whose context columns are added
- `order=:front`: Place context columns at the front (`:front`) or back (`:back`)

# Examples
```julia
result = population_margins(model, data; groups=:region, scenarios=(x=[0, 1],))
df = DataFrame()
append_context_columns!(df, result)
# Adds columns: over_region, at_x

result = profile_margins(model, data, means_grid(data))
df = DataFrame()
append_context_columns!(df, result)
# Adds columns: x, z (bare grid column names, in the grid's order)
```
"""
function append_context_columns!(df::DataFrame, mr::MarginsResult; order::Symbol=:front)
    ctx = mr.context
    isnothing(ctx) && return df  # No context columns to add
    for (j, (name, kind)) in enumerate(zip(keys(ctx), ctx.kinds))
        col_name = _context_column_name(name, kind)
        if order == :front
            DataFrames.insertcols!(df, j, col_name => ctx[name])
        else
            df[!, col_name] = ctx[name]
        end
    end
    return df
end

# Display columns of a result's context, in output order: (output name, values) pairs
function _context_display_columns(mr::MarginsResult)
    ctx = mr.context
    isnothing(ctx) && return Pair{String,AbstractVector}[]
    return Pair{String,AbstractVector}[string(_context_column_name(name, kind)) => ctx[name]
                                       for (name, kind) in zip(keys(ctx), ctx.kinds)]
end

function _show_stata_table(io::IO, mr::EffectsResult)
    # Calculate confidence intervals
    alpha = mr.inference.alpha
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha; df=mr.inference.df)
    
    # Context columns (reference grid, or population groups/scenarios)
    has_profile = !isnothing(mr.context)
    
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
    has_contexts = _result_has_contexts(mr) && _result_analysis_type(mr, :unknown) == :population
    
    # Profile columns (for profile margins or population margins with contexts)
    profile_widths = Int[]  # Specify Int type to avoid sum() type issues
    profile_names = String[]  # Display names
    profile_columns = AbstractVector[]  # Values, for data access
    if has_profile
        # Same names and order as the DataFrame: over_ for groups, at_ for scenarios, bare for grid columns
        for (display_name, values) in _context_display_columns(mr)
            push!(profile_columns, values)
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
    print(io, lpad("[$(_format_level_pct(alpha))% Conf.", num_width))
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
            for (j, (display_name, values, width)) in enumerate(zip(profile_names, profile_columns, profile_widths))
                print(io, " ")
                profile_val = try
                    val = values[i]
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
    # Calculate confidence (or prediction) intervals
    alpha = mr.inference.alpha
    is_prediction_interval = get(mr.metadata, :interval, :confidence) === :prediction
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha; var_inflation=_interval_var_inflation(mr), df=mr.inference.df)
    
    # Context columns (reference grid, or population groups/scenarios)
    has_profile = !isnothing(mr.context)
    
    # Profile columns (for profile margins or population margins with contexts)
    profile_widths = Int[]
    profile_names = String[]
    profile_columns = AbstractVector[]
    if has_profile
        # Same names and order as the DataFrame: over_ for groups, at_ for scenarios, bare for grid columns
        for (display_name, values) in _context_display_columns(mr)
            push!(profile_columns, values)
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
    interval_label = is_prediction_interval ? "% Pred." : "% Conf."
    print(io, lpad("[$(_format_level_pct(alpha))$(interval_label)", num_width))
    print(io, lpad("Interval]", num_width))
    println(io)
    
    # Separator line  
    println(io, "─"^total_width)
    
    # Data rows
    for i in 1:length(mr.estimates)
        # Profile values
        if has_profile
            for (display_name, values, width) in zip(profile_names, profile_columns, profile_widths)
                profile_val = try
                    val = values[i]
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
#
# `var_inflation` adds an observation-level (residual) variance term to the
# interval half-width, turning a confidence interval into a prediction interval:
#   CI:  est ± c·√(SE²)
#   PI:  est ± c·√(SE² + var_inflation)
# where c is the t(df) critical value (Normal when df == Inf; see `_inference_df`).
# It defaults to 0.0, so all existing confidence-interval callers are unaffected.
function _calculate_confidence_intervals(estimates::Vector{Float64}, standard_errors::Vector{Float64}, alpha::Float64; var_inflation::Float64=0.0, df::Real=Inf)
    crit = _critical_value(alpha, df)  # t(df) quantile, or Normal when df == Inf
    half = crit .* sqrt.(standard_errors .^ 2 .+ var_inflation)
    lower = estimates .- half
    upper = estimates .+ half
    return lower, upper
end

# Format a confidence/prediction level as a percentage string for display.
# e.g. alpha=0.05 -> "95", alpha=0.025 -> "97.5". Uses %g rather than Int() so
# fractional confidence levels do not raise InexactError.
_format_level_pct(alpha::Real) = @sprintf("%g", 100 * (1 - alpha))

# Degrees of freedom for display: integers without a trailing ".0".
_format_df(df::Real) = isinteger(df) ? string(Int(df)) : @sprintf("%g", df)

# Test statistics, their df column and two-sided p-values for a result's rows.
function _inference_columns(mr::MarginsResult)
    dof = mr.inference.df
    statistics = mr.estimates ./ mr.standard_errors
    p_values = compute_p_value.(statistics, dof)
    return statistics, fill(dof, length(statistics)), p_values
end

# Residual variance to add when a result requests prediction intervals.
# Returns the stored residual variance when `interval=:prediction`, else 0.0
# (the confidence-interval case). Population results never set `:interval`, so
# they transparently get 0.0 and behave exactly as before.
function _interval_var_inflation(mr)::Float64
    if get(mr.metadata, :interval, :confidence) === :prediction
        return Float64(get(mr.metadata, :resid_var, 0.0))
    end
    return 0.0
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
            # Profile effects (MEM)
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

function _standard_table(mr::EffectsResult; include_gradients::Bool=false)
    statistics, dfs, p_values = _inference_columns(mr)

    # Create self-describing type column
    analysis_type = _result_analysis_type(mr, :population)
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
    df[!, :statistic] = statistics
    df[!, :df] = dfs
    df[!, :p_value] = p_values

    # Confidence (or prediction) intervals at the result's alpha.
    # The reported `se`/`statistic`/`p_value` always describe the mean estimate;
    # only the interval bounds widen when `interval=:prediction` (see metadata).
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, mr.inference.alpha; var_inflation=_interval_var_inflation(mr), df=mr.inference.df)
    df[!, :ci_lower] = lower
    df[!, :ci_upper] = upper

    # CRITICAL: Use actual subgroup sizes, never fallback to incorrect quantities
    if haskey(mr.metadata, :has_subgroup_n) && haskey(mr.metadata, :subgroup_n_values)
        # Grouped case: use the actual subgroup sizes from computation
        df[!, :n] = mr.metadata[:subgroup_n_values]
    else
        # Simple case: use overall sample size
        n_obs = get(mr.metadata, :n_obs, missing)
        df[!, :n] = fill(n_obs, length(mr.estimates))
    end

    # Add gradient vectors if requested
    if include_gradients
        df[!, :gradient] = [mr.gradients[i, :] for i in 1:length(mr.estimates)]
    end

    return df
end

function _compact_table(mr::EffectsResult; include_gradients::Bool=false)
    # Create self-describing type column
    analysis_type = _result_analysis_type(mr, :population)
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

    # Add gradient vectors if requested
    if include_gradients
        df[!, :gradient] = [mr.gradients[i, :] for i in 1:length(mr.estimates)]
    end

    return df
end

function _confidence_table(mr::EffectsResult; include_gradients::Bool=false)
    alpha = mr.inference.alpha
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha; df=mr.inference.df)

    # Create self-describing type column
    analysis_type = _result_analysis_type(mr, :population)
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
    df[!, :ci_lower] = lower
    df[!, :ci_upper] = upper

    # Add gradient vectors if requested
    if include_gradients
        df[!, :gradient] = [mr.gradients[i, :] for i in 1:length(mr.estimates)]
    end

    return df
end

function _profile_table(mr::EffectsResult; include_gradients::Bool=false)
    # Profile-first organization: reference grid with results attached
    alpha = mr.inference.alpha
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, alpha; df=mr.inference.df)

    # Create self-describing type column
    analysis_type = _result_analysis_type(mr, :population)
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
    statistics, dfs, p_values = _inference_columns(mr)
    df[!, :statistic] = statistics
    df[!, :df] = dfs
    df[!, :p_value] = p_values
    df[!, :ci_lower] = lower
    df[!, :ci_upper] = upper

    # Add sample size from metadata
    n_obs = get(mr.metadata, :n_obs, missing)
    df[!, :n] = fill(n_obs, length(mr.estimates))

    # Add gradient vectors if requested
    if include_gradients
        df[!, :gradient] = [mr.gradients[i, :] for i in 1:length(mr.estimates)]
    end

    return df
end

function _predictions_table(mr::PredictionsResult; include_gradients::Bool=false)
    # Predictions-specific format: omits variable/contrast columns
    statistics, dfs, p_values = _inference_columns(mr)

    # Create self-describing type column
    analysis_type = _result_analysis_type(mr, :population)
    type_description = _create_type_description(analysis_type, :prediction, :predictions)

    # Build DataFrame with new column order: context columns first (using consistent helper)
    df = DataFrame()
    append_context_columns!(df, mr; order=:front)

    # Add type column after context columns but before statistical columns
    df[!, :type] = fill(type_description, length(mr.estimates))

    # Core statistical columns (no variable/contrast)
    df[!, :estimate] = mr.estimates
    df[!, :se] = mr.standard_errors
    df[!, :statistic] = statistics
    df[!, :df] = dfs
    df[!, :p_value] = p_values

    # Confidence (or prediction) intervals at the result's alpha.
    # The reported `se`/`statistic`/`p_value` always describe the mean estimate;
    # only the interval bounds widen when `interval=:prediction` (see metadata).
    lower, upper = _calculate_confidence_intervals(mr.estimates, mr.standard_errors, mr.inference.alpha; var_inflation=_interval_var_inflation(mr), df=mr.inference.df)
    df[!, :ci_lower] = lower
    df[!, :ci_upper] = upper

    # Add sample size
    if haskey(mr.metadata, :has_subgroup_n) && haskey(mr.metadata, :subgroup_n_values)
        # Grouped case: use the actual subgroup sizes from computation
        df[!, :n] = mr.metadata[:subgroup_n_values]
    else
        # Simple case: use overall sample size
        n_obs = get(mr.metadata, :n_obs, missing)
        df[!, :n] = fill(n_obs, length(mr.estimates))
    end

    # Add gradient vectors if requested
    if include_gradients
        df[!, :gradient] = [mr.gradients[i, :] for i in 1:length(mr.estimates)]
    end

    return df
end

function _stata_table(mr::EffectsResult; include_gradients::Bool=false)
    statistics, _, p_values = _inference_columns(mr)
    # Stata labels the statistic `t` (P>|t|) for t inference and `z` (P>|z|) for z inference
    t_based = isfinite(mr.inference.df)

    # Create self-describing type column
    analysis_type = _result_analysis_type(mr, :population)
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
    df[!, t_based ? :t : :z] = statistics
    df[!, t_based ? :P_t : :P_z] = p_values  # "P>|t|" / "P>|z|" equivalent
    df[!, :N] = fill(n_obs, length(mr.estimates))  # Add sample size (Stata uses uppercase N)

    # Add gradient vectors if requested
    if include_gradients
        df[!, :gradient] = [mr.gradients[i, :] for i in 1:length(mr.estimates)]
    end

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
