# core/variable_detection.jl - Variable type detection and validation utilities

using CategoricalArrays: CategoricalValue
using FormulaCompiler

"""
    _normalize_override_value(v)

Normalize user/profile override values before constructing scenarios.
Converts CategoricalValue to String for proper scenario handling.
"""
_normalize_override_value(v) = v
_normalize_override_value(v::CategoricalValue) = String(v)

"""
    _get_baseline_level(model, var::Symbol, data_nt::NamedTuple)

Extended version of _get_baseline_level with Boolean variable support.
Handles Boolean variables by returning `false` as the baseline level.

# Arguments
- `model`: Statistical model (GLM, etc.)
- `var`: Variable name
- `data_nt`: Named tuple containing data columns

# Returns
- Baseline level for the variable (specific to variable type)
"""
function _get_baseline_level(model, var::Symbol, data_nt::NamedTuple)
    col = getproperty(data_nt, var)
    if eltype(col) <: Bool
        return false
    elseif eltype(col) <: CategoricalArrays.CategoricalValue
        # For categorical variables, return String for consistent level_map lookup
        # FormulaCompiler returns the actual type (Bool, String, Int, etc.)
        baseline = _get_baseline_level(model, var)
        return string(baseline)
    else
        return _get_baseline_level(model, var)
    end
end

# _validate_variables function is defined in src/engine/core.jl

"""
    _detect_variable_type(data_nt::NamedTuple, var::Symbol) -> Symbol

Detect variable type for unified processing.
Classifies variables into :boolean, :continuous, or :categorical.

# Arguments
- `data_nt`: Named tuple containing data columns
- `var`: Variable name to analyze

# Returns
- `:boolean`: Boolean variables (true/false)
- `:continuous`: Numeric variables (excluding Boolean)
- `:categorical`: All other variables (strings, factors, etc.)

# Examples
```julia
data_nt = (x = [1.0, 2.0, 3.0], flag = [true, false, true], group = ["A", "B", "A"])
_detect_variable_type(data_nt, :x)      # returns :continuous
_detect_variable_type(data_nt, :flag)   # returns :boolean
_detect_variable_type(data_nt, :group)  # returns :categorical
```
"""
function _detect_variable_type(data_nt::NamedTuple, var::Symbol)
    col = getproperty(data_nt, var)

    if eltype(col) <: Bool
        return :boolean
    elseif eltype(col) <: Real
        return :continuous
    else
        return :categorical
    end
end

"""
    _is_continuous_variable(col) -> Bool

Determine if a data column represents a continuous variable.
Returns true for numeric types excluding Boolean.

# Arguments
- `col`: Data column to analyze

# Returns
- `Bool`: True if continuous, false otherwise
"""
function _is_continuous_variable(col)
    return eltype(col) <: Real && !(eltype(col) <: Bool)
end

"""
    count_categorical_rows(categorical_requested, engine::MarginsEngine, contrasts::Symbol=:baseline) -> Int

Count the number of result rows needed for categorical variables.
Used for pre-allocating result DataFrames with correct capacity.

# Arguments
- `categorical_requested`: Vector of categorical variable names
- `engine`: MarginsEngine containing model and data
- `contrasts`: Contrast type (:baseline or :pairwise)

# Returns
- `Int`: Total number of contrast rows needed

# Implementation Notes
For baseline contrasts, each categorical variable with n levels produces (n-1) contrasts.
For pairwise contrasts, each categorical variable with n levels produces n*(n-1)/2 contrasts.
Boolean variables produce exactly 1 contrast row.
"""
function count_categorical_rows(categorical_requested, engine::MarginsEngine, contrasts::Symbol=:baseline)
    additional_rows = 0
    # Handle case where categorical_requested is a single Symbol
    vars_to_process = if categorical_requested isa Symbol
        [categorical_requested]
    else
        categorical_requested
    end

    for var in vars_to_process
        var_col = getproperty(engine.data_nt, var)
        if _detect_variable_type(engine.data_nt, var) == :categorical
            unique_levels = unique(var_col)
            n_levels = length(unique_levels)

            if contrasts == :baseline
                baseline_level = _get_baseline_level(engine.model, var, engine.data_nt)
                non_baseline_count = sum(level != baseline_level for level in unique_levels)
                additional_rows += max(non_baseline_count, 1)
            elseif contrasts == :pairwise
                # All unique pairs: n*(n-1)/2
                additional_rows += div(n_levels * (n_levels - 1), 2)
            else
                throw(ArgumentError("Unsupported contrast type: $contrasts"))
            end
        else
            # Boolean variables always produce 1 contrast
            additional_rows += 1
        end
    end
    return additional_rows
end

"""
    count_categorical_rows(categorical_vars::Vector{Symbol}, engine::MarginsEngine, contrasts::Symbol=:baseline) -> Int

Alternative interface for counting categorical contrast rows.
Handles Vector{Symbol} input for categorical variables.

# Arguments
- `categorical_vars`: Vector of categorical variable names
- `engine`: MarginsEngine containing model and data
- `contrasts`: Contrast type (:baseline or :pairwise)

# Returns
- `Int`: Total number of contrast rows needed

# Implementation Notes
For baseline contrasts, returns (n_levels - 1) contrasts per categorical variable.
For pairwise contrasts, returns n_levels*(n_levels-1)/2 contrasts per categorical variable.
Missing variables are handled gracefully by returning 0 rows.
"""
function count_categorical_rows(categorical_vars::Vector{Symbol}, engine::MarginsEngine, contrasts::Symbol=:baseline)
    total = 0
    for var in categorical_vars
        if haskey(engine.data_nt, var)
            levels = unique(getproperty(engine.data_nt, var))
            n_levels = length(levels)

            if contrasts == :baseline
                # For baseline contrasts, we get (n_levels - 1) contrasts
                total += max(0, n_levels - 1)
            elseif contrasts == :pairwise
                # All unique pairs: n*(n-1)/2
                total += div(n_levels * (n_levels - 1), 2)
            else
                throw(ArgumentError("Unsupported contrast type: $contrasts"))
            end
        end
    end
    return total
end