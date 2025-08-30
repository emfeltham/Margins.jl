# ~~FormulaCompiler OverrideVector Bug Report~~ → **RESOLVED: Margins.jl Usage Error**

**Date**: August 2025  
**Issue**: ~~`contrast_modelrow!` fails with `OverrideVector{CategoricalValue}` data~~ **INCORRECT USAGE**  
**Error**: `"Cannot extract level code from String"`  
**Impact**: ~~Blocks Issue #5 - Table-based categorical effects in Margins.jl~~ **RESOLVED**

## ✅ **RESOLUTION: Not a FormulaCompiler Bug**

**Root Cause**: Margins.jl was **incorrectly using FormulaCompiler's API**. The issue was not a bug in FormulaCompiler, but a misunderstanding of the proper usage pattern for categorical contrasts in profile scenarios.

## ❌ **Incorrect Approach (What We Were Doing)**

```julia
# WRONG: Trying to call contrast_modelrow! with scenario data containing overrides
scen = FormulaCompiler.create_scenario("profile", data_nt, profile_overrides)
FormulaCompiler.contrast_modelrow!(Δ, compiled, scen.data, 1; var=:cat, from="A", to="B")
# ERROR: Cannot extract level code from String
```

## ✅ **Correct Approach (Proper FormulaCompiler Usage)**

```julia
# RIGHT: Create separate scenarios for each contrast level, then evaluate and subtract
prof_from = copy(profile)
prof_to = copy(profile) 
prof_from[:cat] = "A"  # from level
prof_to[:cat] = "B"    # to level

scen_from = FormulaCompiler.create_scenario("from", data_nt, prof_from)
scen_to = FormulaCompiler.create_scenario("to", data_nt, prof_to)

# Evaluate at both scenarios
compiled(X_from, scen_from.data, 1)
compiled(X_to, scen_to.data, 1)

# Compute contrast manually
Δ .= X_to .- X_from  # This gives the proper contrast
```

## Reproduction Case

### Working Case (Regular Data)
```julia
using DataFrames, GLM, CategoricalArrays
using FormulaCompiler

df = DataFrame(
    cat = categorical(["A", "B", "A"]),
    y = [1.0, 2.0, 1.5]
)
model = lm(@formula(y ~ cat), df)
data_nt = Tables.columntable(df)

compiled = FormulaCompiler.compile_formula(model, data_nt)
Δ = Vector{Float64}(undef, length(compiled))
FormulaCompiler.contrast_modelrow!(Δ, compiled, data_nt, 1; var=:cat, from="A", to="B")
# ✅ Works: [0.0, 1.0]
```

### Failing Case (OverrideVector Data)
```julia
# Same setup as above, but with scenario override
processed_prof = Dict{Symbol,Any}(:cat => "A")
scenario = FormulaCompiler.create_scenario("test", data_nt, processed_prof)

# Scenario data contains: OverrideVector{CategoricalValue{String, UInt32}}
FormulaCompiler.contrast_modelrow!(Δ, compiled, scenario.data, 1; var=:cat, from="A", to="B")
# ❌ Fails: "Cannot extract level code from String"
```

## Data Type Analysis

### Original Data
- Type: `CategoricalVector{String, UInt32, String, CategoricalValue{String, UInt32}, Union{}}`
- Elements: `CategoricalValue{String, UInt32}`

### Scenario Override Data  
- Type: `FormulaCompiler.OverrideVector{CategoricalValue{String, UInt32}}`
- Elements: `CategoricalValue{String, UInt32}` (same as original)

**Observation**: The element types are identical, but the container types differ.

## Stack Trace
```
ERROR: Cannot extract level code from String
Stacktrace:
 [1] error(s::String) @ Base ./error.jl:35
 [2] extract_level_code_zero_alloc @ FormulaCompiler/src/compilation/execution.jl:338 [inlined]
 [3] execute_op @ FormulaCompiler/src/compilation/execution.jl:379 [inlined]
 [4] execute_ops_recursive @ FormulaCompiler/src/compilation/execution.jl:181 [inlined]
 [5] (::FormulaCompiler.UnifiedCompiled)(output::Vector{Float64}, data::NamedTuple, row_idx::Int64) @ FormulaCompiler/src/compilation/execution.jl:92
 [6] contrast_modelrow! @ FormulaCompiler/src/evaluation/derivatives/contrasts.jl:43
```

## FormulaCompiler Code Analysis

The error originates in `extract_level_code_zero_alloc`:
```julia
@inline function extract_level_code_zero_alloc(column_data::AbstractVector, row_idx::Int)
    cat_value = column_data[row_idx]
    if isa(cat_value, CategoricalValue)
        return Int(levelcode(cat_value))
    elseif isa(cat_value, Integer)
        return Int(cat_value)
    elseif hasproperty(cat_value, :level)
        return Int(cat_value.level)
    else
        error("Cannot extract level code from $(typeof(cat_value))")  # <- This line
    end
end
```

**Hypothesis**: When `column_data` is an `OverrideVector{CategoricalValue}`, the function `column_data[row_idx]` is somehow returning a `String` instead of the expected `CategoricalValue`.

## Impact on Margins.jl

This bug blocks:
- ✅ **Population categorical effects**: Work (use regular data, no overrides)
- ❌ **Profile-based categorical effects**: Fail (use scenario overrides)
- ❌ **Table-based categorical effects**: Fail (use scenario overrides)
- ✅ **Categorical predictions**: Work (different code path)

## Workarounds

### For Margins.jl Users
Use population-level categorical effects instead of profile-based:
```julia
# ✅ Works
result = population_margins(model, data; type=:effects, vars=[:cat])

# ❌ Fails  
result = profile_margins(model, data; at=Dict(:cat => "A"), type=:effects, vars=[:cat])
```

### For Developers
The issue is in FormulaCompiler, not Margins.jl. Any categorical contrast operation using scenario overrides will fail.

## Next Steps

1. **Investigate `OverrideVector` indexing behavior** in FormulaCompiler
2. **Check if `extract_level_code_zero_alloc` handles `OverrideVector` types correctly**
3. **Fix the categorical level extraction logic** to handle override vectors
4. **Test fix with Margins.jl categorical effects**

## Files Involved

### FormulaCompiler
- `src/compilation/execution.jl` - `extract_level_code_zero_alloc` function
- `src/evaluation/derivatives/contrasts.jl` - `contrast_modelrow!` function  
- `src/scenarios/overrides.jl` - `OverrideVector` type and creation

### Margins.jl
- `src/computation/categorical.jl` - Profile-based categorical effects
- `src/features/categorical_mixtures.jl` - Scenario processing
- `src/api/population.jl` - Population effects (fixed `row_idxs` bug)

## ✅ **Resolution Status: COMPLETE**

- **Root Cause**: ✅ **IDENTIFIED** - Margins.jl was using FormulaCompiler API incorrectly
- **Margins.jl Fix**: ✅ **COMPLETED** - Implemented proper scenario-based contrast approach
- **FormulaCompiler Fix**: ✅ **NOT NEEDED** - FormulaCompiler was working correctly all along
- **Issue #5 Status**: ✅ **RESOLVED** - Table-based categorical effects now working

## Key Learnings

1. **`contrast_modelrow!` is designed for regular data**, not scenario data with overrides
2. **Profile-based contrasts** should be computed by evaluating at multiple scenarios and taking differences
3. **FormulaCompiler's architecture was correct** - the issue was in how Margins.jl was using it
4. **RTFM principle**: The FormulaCompiler documentation and tests showed the correct usage patterns

## Impact on Margins.jl

✅ **All categorical effects now work**:
- ✅ Population categorical effects  
- ✅ Dict-based profile categorical effects
- ✅ Table-based profile categorical effects
- ✅ Categorical mixtures (were already working)