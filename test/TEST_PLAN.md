# Margins.jl Test Plan & Current Status Analysis

## N.B.

- run individual test files with --project="test"
- run --project="." for Pkg.test()
- No print statements, icons, or distracting @info. Rely on the formal @test/@testset structure, and use `@debug` for extended output instead of `println` or `@info`
- All allocations and timings and performance should be measured with BenchmarkTools.jl -- no @allocated.
- Do not include skip logic. Tests should error or fail!

This document serves two purposes:
1. **Comprehensive Test Plan**: Specifies a correctness-focused test suite for Margins.jl
2. **Current Status Analysis**: Analysis of the ~80 existing test files and integration recommendations

## Explicit errors

**PRIORITY: CRITICAL** - The codebase contains multiple violations of the package's zero-tolerance error policy. These patterns produce potentially invalid statistical results without clear user notification, violating the core principle that "wrong standard errors are worse than no standard errors."

### 🚨 Statistical Correctness Violations Found

- [x] #### 1. **Silent Fallbacks in Link Function Detection** (`src/engine/core.jl:147, 160-163`)

**VIOLATION**: Link function extraction uses try-catch with fallback that could silently fail:
```julia
try
    # Try to extract link from GLM.jl...
catch e
    error("Failed to extract link function from model: $e. " *
          "Statistical correctness cannot be guaranteed without proper link function.")
end
```

**GOOD**: This actually **does error explicitly** - this is correct behavior that should be maintained.

**Documentation claims**: "Returns IdentityLink() fallback" but code actually errors - documentation is misleading.

- [x] #### 2. **Silent First-Value Fallbacks** (`src/engine/utilities.jl:597, src/profile/refgrids.jl:341-342`)

**CRITICAL VIOLATION**: Multiple locations use silent fallbacks to `first(col)`:
```julia
# src/engine/utilities.jl:597
else
    return first(col)  # Fallback to first value
end

# src/profile/refgrids.jl:341-342  
else
    # Fallback to first value (O(1))
    return first(col)
end
```

**STATISTICAL RISK**: Using first value for unknown data types could produce statistically meaningless reference grids without user awareness.

**REQUIRED FIX**: Should error with explicit message about unsupported data types rather than silent fallback.

- [-] #### 3. **Warning Instead of Error for Large Combinations** (`src/population/contexts.jl:29-31`)

NOTE: THIS ONE IS FINE

**VIOLATION**: Performance warnings instead of hard limits:
```julia
@warn "Large number of combinations detected ($total_combinations). " *
      "This may result in slow computation and large output. " *
      "Consider reducing grouping complexity..."
```

**STATISTICAL RISK**: Allows computations that may not complete or exhaust memory, leading to incomplete/invalid results.

**REQUIRED FIX**: Should have configurable hard limits with clear error messages.

- [x] #### 4. **Silent DataFrame Concatenation Fallback** (`src/population/contexts.jl:623-634`)

**CRITICAL VIOLATION**: DataFrame concatenation with silent fallback:
```julia
try
    return vcat(results, new_result; cols=:union)
catch e
    # Fallback: manual column alignment with string-based missing values
    all_cols = union(names(results), names(new_result))
    # ... fills with "missing" strings
end
```

**STATISTICAL RISK**: Silent data structure failures could corrupt results or misalign statistical estimates with their metadata.

**REQUIRED FIX**: Should error explicitly when DataFrame structures are incompatible rather than silent string-based missing value injection.

- [x] #### 5. **Gradient Format Fallback** (`src/features/averaging.jl:225-231`)

**VIOLATION**: Silent fallback to old gradient format:
```julia
elseif isa(grad_key, Tuple) && length(grad_key) == 2
    # Fallback to old format if available
    g_term, g_prof_idx = grad_key
```

**STATISTICAL RISK**: Format mismatches could lead to incorrect gradient associations and invalid standard errors.

NOTE: did we remove the old Gradient format?

**REQUIRED FIX**: Should error when gradient format is unexpected rather than attempting compatibility.

### 🎯 Required Test Coverage for Error-First Policy

#### Tests Needed:

1. **Link Function Failure Tests**:
   - Verify error (not silent fallback) for unsupported model types
   - Test with models lacking proper link function methods
   - Validate error message clarity and statistical rationale

2. **Data Type Validation Tests**:
   - Test unsupported data types cause explicit errors
   - Verify no silent `first(col)` fallbacks occur
   - Test mixed-type columns error appropriately

3. **Resource Limit Tests**:
   - Test combinations above memory/time limits cause errors
   - Verify user gets actionable error messages
   - Test graceful degradation boundaries

4. **DataFrame Structure Tests**:
   - Test incompatible result structures cause explicit errors
   - Verify no silent data corruption via string "missing" injection
   - Test metadata alignment validation

5. **Gradient Format Tests**:
   - Test gradient format mismatches cause explicit errors
   - Verify no silent fallbacks to old formats
   - Test standard error validity when gradients are malformed

#### Implementation Principle:
**EVERY statistical computation failure MUST produce explicit error with clear explanation rather than silent approximation or fallback.**

## More testing?

