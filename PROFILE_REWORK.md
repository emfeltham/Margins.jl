# Profile System Architectural Rework

**COMPLETE**

## Problem Statement

The current `profile_margins` implementation has a fundamental architectural flaw that causes both performance issues and bugs (including the CategoricalMixture routing failure). The system incorrectly treats each profile as requiring its own FormulaCompiler compilation instead of treating profiles as rows in a single reference dataset.

## Current (Broken) Architecture

### What Happens Now
```julia
# profile/core.jl:149-150
profiles = [Dict(pairs(row)) for row in eachrow(reference_grid)]
df, G = _mem_continuous_and_categorical(engine, profiles; ...)

# engine/utilities.jl:346-350  
for profile in profiles
    refgrid_data = _build_refgrid_data(profile, engine.data_nt)  # Per-profile data
    refgrid_compiled = FormulaCompiler.compile_formula(engine.model, refgrid_data)  # Per-profile compilation!
    # ... evaluate single profile
end
```

### Problems with Current Design

1. **Performance**: O(n) compilations instead of O(1)
   - Each profile triggers expensive `FormulaCompiler.compile_formula()` call
   - For 10 profiles = 10 compilations of the same model formula
   - Should compile once, evaluate many times

2. **Bug Source**: Per-profile compilation breaks mixture routing
   - Each compilation uses slightly different data structures
   - Mixture detection may fail inconsistently across profiles
   - This is the root cause of the `extract_level_code` error with `CategoricalMixture`

3. **Conceptual Error**: Profiles are data, not different models
   - A profile is just a specific combination of covariate values
   - The model formula is identical across all profiles
   - Only the input data values change

4. **Mixed Abstractions**: Confuses reference grid vs scenario approaches
   - Reference grids should be single datasets with multiple rows
   - Scenarios/overrides are for counterfactual analysis on existing data
   - Current code mixes both approaches inefficiently

## Correct Architecture

### Core Principle
**Profiles are rows in a single reference dataset, not separate compilations**

### Proposed Solution
```julia
# Step 1: Build complete reference grid (already working)
reference_grid = _build_reference_grid(at, data_nt)  # DataFrame with all profiles

# Step 2: Compile once with complete reference grid
reference_data = Tables.columntable(reference_grid)
compiled = FormulaCompiler.compile_formula(model, reference_data)

# Step 3: Evaluate all profiles in batch
results = []
for row_idx in 1:nrow(reference_grid)
    # Single evaluation call per profile
    result = evaluate_profile(compiled, reference_data, row_idx, vars, target, backend)
    push!(results, result)
end
```

### Benefits of Correct Architecture

1. **Performance**: Single compilation for all profiles
   - O(1) compilation cost regardless of number of profiles  
   - Massive speedup for multi-profile analysis

2. **Bug Fix**: Consistent mixture routing
   - Single compilation ensures consistent mixture detection
   - Fixes `CategoricalMixture` → `extract_level_code` routing issue
   - All profiles handled by same compiled operations

3. **Conceptual Clarity**: Profiles as data rows
   - Clear separation: model compilation vs data evaluation
   - Aligns with FormulaCompiler's design principles
   - Easier to understand and maintain

4. **Memory Efficiency**: Single compiled object
   - No duplicate compilation objects in memory
   - Better cache locality for evaluation

## Implementation Plan

### Phase 1: Refactor `_mem_continuous_and_categorical`
- Replace per-profile compilation loop with single compilation
- Modify evaluation logic to work with reference grid rows
- Maintain same API but fix internal implementation

### Phase 2: Optimize Reference Grid Creation  
- Ensure `_build_reference_grid` creates efficient single DataFrame
- Verify mixture objects are preserved correctly in grid format
- Remove unnecessary `_build_refgrid_data` per-profile data creation

### Phase 3: Performance Testing
- Benchmark new vs old approach  
- Verify O(1) vs O(n) compilation scaling
- Test with various profile counts and mixture specifications

## Code Changes Required

### Primary File: `/src/engine/utilities.jl`
**Function**: `_mem_continuous_and_categorical`
- Replace lines 346-350 per-profile compilation loop
- Implement single compilation with batch evaluation
- Remove `_build_refgrid_data` calls in loop

### Secondary File: `/src/profile/core.jl`  
**Function**: `_profile_margins_impl`
- Potentially eliminate profile Dict conversion (line 149)
- Work directly with reference_grid DataFrame if possible
- Simplify data flow

## Expected Impact

### Bug Fixes
- ✅ Resolves `CategoricalMixture` → `extract_level_code` error
- ✅ Consistent mixture routing across all profiles  
- ✅ Eliminates compilation-dependent behavior differences

### Performance Improvements
- ✅ ~10-100x faster compilation for multi-profile analysis
- ✅ Lower memory usage (single compiled object)
- ✅ Better scalability with profile count

### Code Quality
- ✅ Clearer architectural separation
- ✅ Aligns with FormulaCompiler design principles
- ✅ Easier to debug and maintain
- ✅ Reduces code complexity

## Risk Assessment

### Low Risk Changes
- Internal implementation changes only
- Same external API maintained
- Existing tests should pass unchanged

### Validation Strategy  
- Compare outputs with current implementation before/after
- Run full test suite to ensure no regressions
- Benchmark performance improvements
- Test specifically with mixture objects to confirm bug fix

## Conclusion

This architectural rework addresses the root cause of both the performance issues and the `CategoricalMixture` bug. By treating profiles correctly as rows in a single reference dataset rather than separate compilation targets, we align with both statistical best practices and FormulaCompiler's design philosophy.

The fix is conceptually simple but requires careful implementation to maintain API compatibility while fundamentally changing the internal data flow.