# computation/scenarios.jl - Categorical contrast utilities
#
# Utilities for generating and processing categorical contrasts in marginal effects.
# Uses ContrastEvaluator kernel system for zero-allocation computations.

using CategoricalArrays: CategoricalValue, levels

"""
    ContrastPair{T}

Type-stable representation of a categorical contrast pair.
Eliminates dynamic typing in contrast generation.

# Type Parameter
- `T`: Type of the categorical levels being contrasted

# Fields
- `level1::T`: First level in the contrast (typically baseline)
- `level2::T`: Second level in the contrast (typically treatment)

# Examples
```julia
# Boolean contrast
bool_pair = ContrastPair(false, true)

# String contrast
string_pair = ContrastPair("Control", "Treatment")

# Numeric contrast
numeric_pair = ContrastPair(1, 2)
```
"""
struct ContrastPair{T}
    level1::T
    level2::T
end

# Note: ContrastPair types must match the data type for proper type stability
# Converting between types (e.g., String ↔ CategoricalValue) is not supported

"""
    generate_contrast_pairs(var_col, rows, contrasts::Symbol, model, var::Symbol, data_nt) -> Vector{ContrastPair}

Generate contrast pairs for categorical variables using type-stable representation.
Supports both baseline and pairwise contrasts with proper type inference.

# Arguments
- `var_col`: Data column for the categorical variable
- `rows`: Row indices to consider for level detection
- `contrasts`: Contrast type (:baseline or :pairwise)
- `model`: Statistical model (for baseline level detection)
- `var`: Variable name (for baseline level detection)
- `data_nt`: Named tuple containing data (for baseline level detection)

# Returns
- `Vector{ContrastPair{T}}`: Type-stable contrast pairs where T = eltype(var_col)

# Contrast Types
- `:baseline`: Each non-baseline level vs. baseline level
- `:pairwise`: All unique pairs of levels (excluding self-pairs)

# Special Cases
- **Boolean variables**: Returns `[ContrastPair(false, true)]` for baseline contrasts
- **Single level**: Returns `[ContrastPair(level, level)]` to indicate no contrasts
- **Empty levels**: Handled gracefully with appropriate error messages

# Performance
- Type-stable contrast pairs eliminate dynamic typing downstream
- Single pass through data for level detection
- Efficient Set operations for unique level identification
"""
function generate_contrast_pairs(var_col, rows, contrasts::Symbol, model, var::Symbol, data_nt)
    T = eltype(var_col)

    if contrasts === :baseline
        if T <: Bool
            return ContrastPair{Bool}[ContrastPair(false, true)]
        elseif T <: CategoricalArrays.CategoricalValue
            # Levels come from the FITTED MODEL (FormulaCompiler's `categorical_levels`), in
            # model order, not from the data's pool: the pool may be reordered, or lack
            # levels the model knows — e.g. a subset without the reference level. Pairs hold
            # the raw level values; the ContrastEvaluator resolves them by value.
            (model_levels, observed_strs) = _model_contrast_levels(var_col, model, var, data_nt)
            baseline_level_str = string(_get_baseline_level(model, var, data_nt))
            bidx = findfirst(l -> string(l) == baseline_level_str, model_levels)
            isnothing(bidx) && error("Baseline level '$baseline_level_str' is not a level of :$var in the fitted model")
            baseline_level = model_levels[bidx]

            # Unused levels (no observations in the data) are skipped: StatsModels drops
            # zero-count pool levels when it builds the contrasts, and a contrast to a level
            # with no rows would be a spurious zero effect. The baseline is always the
            # reference, observed or not. Filter on the full data column, not `rows`, so that
            # levels present in the data but absent from a subgroup are still included.
            L = eltype(model_levels)
            contrast_pairs = ContrastPair{L}[]
            for level in model_levels
                level_str = string(level)
                if level_str != baseline_level_str && level_str in observed_strs
                    push!(contrast_pairs, ContrastPair{L}(baseline_level, level))
                end
            end

            if isempty(contrast_pairs)
                return ContrastPair{L}[ContrastPair{L}(baseline_level, baseline_level)]
            end

            return contrast_pairs
        else
            # For other types (raw strings, etc.), use the data type directly
            baseline_level = _get_baseline_level(model, var, data_nt)
            all_levels = Set{T}()
            for row in rows
                push!(all_levels, var_col[row])
            end

            contrast_pairs = ContrastPair{T}[]
            for level in all_levels
                if level != baseline_level
                    push!(contrast_pairs, ContrastPair(baseline_level, level))
                end
            end

            # Handle case where only baseline level exists
            if isempty(contrast_pairs)
                return ContrastPair{T}[ContrastPair(baseline_level, baseline_level)]
            end

            return contrast_pairs
        end
    elseif contrasts === :pairwise
        if T <: CategoricalArrays.CategoricalValue
            # Model level order (see the baseline branch), observed levels only
            (model_levels, observed_strs) = _model_contrast_levels(var_col, model, var, data_nt)
            unique_levels = [l for l in model_levels if string(l) in observed_strs]
        else
            # For other types, unique works fine
            unique_levels = unique(var_col[rows])
        end

        contrast_pairs = ContrastPair{eltype(unique_levels)}[]
        for (i, level1) in enumerate(unique_levels)
            for (j, level2) in enumerate(unique_levels)
                if i < j
                    push!(contrast_pairs, ContrastPair(level1, level2))
                end
            end
        end
        return contrast_pairs
    else
        throw(ArgumentError("Unsupported contrast type: $contrasts. Use :baseline or :pairwise"))
    end
end

"""
    format_categorical_results(contrast_results::Vector) -> Vector

Format categorical contrast results with proper type stability.
Separates result formatting from computation logic.

# Arguments
- `contrast_results`: Vector of contrast computation results

# Returns
- `Vector`: Formatted results (currently pass-through)

# Implementation Notes
Results are already in the correct format from the ContrastEvaluator kernel system.
This function can be extended for additional formatting needs without changing
computation kernels.

# Future Extensions
Could include:
- Result validation and error checking
- Unit conversion or scaling
- Statistical summary computations
- Alternative result representations
"""
function format_categorical_results(contrast_results::Vector)
    # Results are already in the correct format from the ContrastEvaluator kernel
    # This function can be extended for additional formatting needs
    return contrast_results
end
"""
    _model_contrast_levels(var_col, model, var, data_nt) -> (levels, observed_strings)

Levels of categorical `var` as the fitted model knows them (raw values, model order, from
FormulaCompiler's `categorical_levels`), and the set of level strings observed in `var_col`.
Falls back to the column's own levels when the compiled formula does not expose them.
"""
function _model_contrast_levels(var_col, model, var::Symbol, data_nt)
    cats = _model_categorical_levels(model, data_nt)
    model_levels = haskey(cats, var) ? collect(cats[var]) :
        CategoricalArrays.unwrap.(levels(var_col))
    observed = Set{String}(string(var_col[i]) for i in eachindex(var_col))
    return (model_levels, observed)
end
