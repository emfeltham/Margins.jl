# core/data_conversion.jl
# Data type conversion utilities for type stability

"""
    _convert_numeric_to_float64(data_nt::NamedTuple) -> NamedTuple

Convert all numeric columns in a NamedTuple to Float64 for type stability.

This function ensures homogeneous numeric types in the NamedTuple, preventing
type instability that causes allocations in downstream code. Categorical and
other non-numeric columns are preserved as-is.

# Type Stability Issue
Heterogeneous NamedTuples like:
```julia
@NamedTuple{x1::Vector{Float64}, x2::Vector{Int64}, cat::CategoricalVector{...}}
```
cause type instability when accessing columns generically, leading to allocations.

# Solution
Convert all numeric types to Float64:
```julia
@NamedTuple{x1::Vector{Float64}, x2::Vector{Float64}, cat::CategoricalVector{...}}
```
This allows type-stable column access for numeric operations.

# Arguments
- `data_nt::NamedTuple`: Input data as NamedTuple from Tables.columntable()

# Returns
- `NamedTuple`: Data with all numeric columns converted to Float64

# Performance
- Converts Int64, Int32, Float32, etc. to Float64
- Preserves CategoricalArray, String, Bool columns
- One-time conversion cost, eliminates per-row allocations

# Examples
```julia
# Before: heterogeneous types
data = (x1=[1.0, 2.0], x2=[1, 2], cat=categorical(["A", "B"]))
# Type: @NamedTuple{x1::Vector{Float64}, x2::Vector{Int64}, cat::CategoricalVector}

# After: homogeneous numeric types
data_converted = _convert_numeric_to_float64(data)
# Type: @NamedTuple{x1::Vector{Float64}, x2::Vector{Float64}, cat::CategoricalVector}
```
"""
function _convert_numeric_to_float64(data_nt::NamedTuple{names}) where {names}
    # Build new NamedTuple with Float64 numeric columns
    converted_cols = map(names) do name
        col = getproperty(data_nt, name)
        # Convert numeric columns (excluding Bool and CategoricalArray)
        if col isa AbstractVector{<:Real} && !(eltype(col) <: Bool) && !(eltype(col) === Float64)
            # Need to convert: Int64, Int32, Float32, etc.
            return Float64.(col)
        else
            # Keep as-is: Float64 (already correct), Bool, CategoricalArray, String, etc.
            return col
        end
    end
    # Reconstruct NamedTuple with same names
    return NamedTuple{names}(converted_cols)
end
