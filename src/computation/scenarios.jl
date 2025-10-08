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
            # For categorical variables, use ALL levels from the categorical variable,
            # not just levels in the subset of rows (important for AME computation)
            baseline_level_str = _get_baseline_level(model, var, data_nt)  # Returns String

            # Build a map from level strings to CategoricalValue instances using pool
            # This ensures all schema-defined levels are included, not just observed values
            level_map = Dict{String, T}()
            pool = var_col.pool
            for (i, level) in enumerate(levels(var_col))
                level_str = string(level)  # Convert actual type (Bool, String, Int, etc.) to String
                level_map[level_str] = pool[i]  # Direct pool indexing (1-based)
            end

            # Get baseline as CategoricalValue
            if !haskey(level_map, baseline_level_str)
                error("Baseline level '$baseline_level_str' not found in categorical variable :$var")
            end
            baseline_level = level_map[baseline_level_str]

            # Create contrast pairs using CategoricalValue type
            all_levels = levels(var_col)  # Returns actual types (Bool, String, Int, etc.)
            contrast_pairs = ContrastPair{T}[]
            for level in all_levels
                level_str = string(level)  # Convert to string for map lookup
                if level_str != baseline_level_str
                    push!(contrast_pairs, ContrastPair(baseline_level, level_map[level_str]))
                end
            end

            # Handle case where only baseline level exists
            if isempty(contrast_pairs)
                return ContrastPair{T}[ContrastPair(baseline_level, baseline_level)]
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
            # For categorical variables, ensure we use CategoricalValue type
            # Build level map using pool to preserve type consistency
            level_map = Dict{String, T}()
            pool = var_col.pool
            for (i, level) in enumerate(levels(var_col))
                level_str = string(level)  # Convert actual type to String
                level_map[level_str] = pool[i]  # Direct pool indexing (1-based)
            end

            # Get unique level strings from the subset
            unique_level_strs = unique(string.(var_col[rows]))
            unique_levels = [level_map[s] for s in unique_level_strs]
        else
            # For other types, unique works fine
            unique_levels = unique(var_col[rows])
        end

        contrast_pairs = ContrastPair{T}[]
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