# Categorical Profile Mixtures

This document describes the categorical mixture feature for specifying population compositions in profile margins analysis.

## Overview

Margins.jl supports categorical mixtures for specifying population composition scenarios. This extends the existing fractional Bool support (e.g., `treated => 0.7`) to multi-level categorical variables using a simple `mix()` syntax.

## API Usage

### Basic Syntax

```julia
# Categorical mixture specification
at = Dict(
    :education => mix("high_school" => 0.3, "college" => 0.5, "graduate" => 0.2),
    :region => mix("urban" => 0.65, "rural" => 0.35),
    :age => [25, 50, 65]  # Regular numeric values still work
)

result = profile_margins(model, data; at=at, type=:predictions)
```

### Key Features

- **Sum-to-1 validation**: Mixture weights must sum to 1.0 (throws `ArgumentError` if not)
- **Level validation**: All mixture levels must exist in the original categorical variable
- **Integration**: Works seamlessly with existing numeric profile specifications

## Implementation

### FormulaCompiler Integration

The key implementation is extending FormulaCompiler's `ContrastOp` to handle weighted combinations of contrast matrix rows instead of selecting single rows.

**Current behavior**: For discrete categorical levels
```julia
# Single level: uses one row of contrast matrix
level = 2  # "college" 
scratch[pos] = contrast_matrix[2, i]  # Single row lookup
```

**Extended behavior**: For categorical mixtures  
```julia
# Mixture: weighted combination of contrast matrix rows
mixture = mix("high_school" => 0.3, "college" => 0.5, "graduate" => 0.2)
scratch[pos] = 0.3 * contrast_matrix[3, i] +    # high_school row
               0.5 * contrast_matrix[1, i] +    # college row  
               0.2 * contrast_matrix[2, i]      # graduate row
```

### Required Changes

**File**: `FormulaCompiler/src/compilation/execution.jl`

Extend the `ContrastOp` execution method to detect `CategoricalMixture` objects and compute weighted combinations:

```julia
@inline function execute_op(
    op::ContrastOp{Col, Positions}, 
    scratch, 
    data, 
    row_idx
) where {Col, Positions}
    column_data = getproperty(data, Col)
    
    if column_data isa OverrideVector{CategoricalMixture{T}} where T
        # Categorical mixture: weighted combination of contrast rows
        mixture = column_data.override_value
        
        # Map mixture level names to contrast matrix row indices
        original_levels = get_original_categorical_levels(column_data)
        
        for (i, pos) in enumerate(Positions)
            scratch[pos] = 0.0
            for (level_name, weight) in zip(mixture.levels, mixture.weights)
                level_idx = findfirst(==(string(level_name)), string.(original_levels))
                if level_idx !== nothing
                    scratch[pos] += weight * op.contrast_matrix[level_idx, i]
                end
            end
        end
    else
        # Standard categorical handling (unchanged)
        level = extract_level_code_zero_alloc(column_data, row_idx)
        level = clamp(level, 1, size(op.contrast_matrix, 1))
        
        for (i, pos) in enumerate(Positions)
            scratch[pos] = convert(eltype(scratch), op.contrast_matrix[level, i])
        end
    end
end
```

## Status (Updated 2025)

✅ **API Complete**: The `mix()` function and basic integration is working  
✅ **FormulaCompiler Foundation Ready**: All required primitives now available
- ✅ Type-flexible override system with fractional specification support
- ✅ Robust categorical handling with proper CategoricalArray support
- ✅ Zero-allocation scenario evaluation ready for production
🔄 **FormulaCompiler Extension**: Ready to implement weighted contrast support
📋 **Testing**: Need to verify proper weighted combinations vs. current approximation

## FormulaCompiler.jl Readiness

The recent FormulaCompiler.jl improvements provide the necessary foundation:

### ✅ **Fractional Specification Support**
FormulaCompiler now handles fractional values for both:
- **Boolean variables**: `treated => 0.7` (70% treated, 30% control)  
- **Integer variables**: `age => 25.5` (fractional age specification)

### ✅ **Robust Override System**  
The `create_scenario()` system now supports:
- **Type-flexible conversions**: Automatic handling of mixed data types
- **Error handling**: Proper validation for incompatible overrides
- **Memory efficiency**: OverrideVector optimizations maintain O(1) memory usage

### 🔄 **Next Implementation Step**
The categorical mixture feature can now be implemented by extending the existing override system to support `CategoricalMixture` objects. The weighted contrast matrix computation described above is ready to be implemented in FormulaCompiler.jl's execution engine.

### **Implementation Priority**
This feature should be implemented **after** the core Margins.jl API is stable, as categorical mixtures are an advanced feature for specialized use cases. The existing fractional Boolean support covers most practical scenarios for population composition analysis.