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

## Reorganization plan

The current flat test directory structure should be reorganized to mirror the logical structure of `runtests.jl`, creating subdirectories that group related functionality:

### Proposed Directory Structure

```
test/
├── runtests.jl                     # Main test runner
├── core/                          # Core Functionality testset
│   ├── test_glm_basic.jl
│   ├── test_profiles.jl
│   ├── test_grouping.jl
│   ├── test_contrasts.jl
│   ├── test_vcov.jl
│   ├── test_errors.jl
│   ├── test_automatic_variable_detection.jl
│   └── test_mixedmodels.jl
├── features/                      # Advanced Features testset
│   ├── test_elasticities.jl
│   ├── test_categorical_mixtures.jl
│   ├── test_bool_profiles.jl
│   ├── test_table_profiles.jl
│   └── test_prediction_scales.jl
├── performance/                   # Performance testset
│   ├── test_performance.jl
│   └── test_zero_allocations.jl
└── statistical_validation/        # Statistical Correctness testset (already organized)
    ├── backend_consistency.jl
    ├── statistical_validation.jl
    └── [other validation files...]
```

### Implementation Plan

1. **Create subdirectories**: `mkdir -p test/{core,features,performance}`

2. **Move files to appropriate subdirectories**:
   ```bash
   # Core functionality
   mv test/test_glm_basic.jl test/core/
   mv test/test_profiles.jl test/core/
   mv test/test_grouping.jl test/core/
   mv test/test_contrasts.jl test/core/
   mv test/test_vcov.jl test/core/
   mv test/test_errors.jl test/core/
   mv test/test_automatic_variable_detection.jl test/core/
   mv test/test_mixedmodels.jl test/core/
   
   # Advanced features
   mv test/test_elasticities.jl test/features/
   mv test/test_categorical_mixtures.jl test/features/
   mv test/test_bool_profiles.jl test/features/
   mv test/test_table_profiles.jl test/features/
   mv test/test_prediction_scales.jl test/features/
   
   # Performance
   mv test/test_performance.jl test/performance/
   mv test/test_zero_allocations.jl test/performance/
   ```

3. **Update `runtests.jl`** to use new paths:
   ```julia
   # Core functionality tests
   @testset "Core Functionality" begin
       include("core/test_glm_basic.jl")
       include("core/test_profiles.jl")
       # ... etc
   end
   
   # Advanced Features 
   @testset "Advanced Features" begin
       include("features/test_elasticities.jl")
       include("features/test_categorical_mixtures.jl")
       # ... etc
   end
   
   # Performance
   @testset "Performance" begin
       include("performance/test_performance.jl")
       include("performance/test_zero_allocations.jl")
   end
   ```

### Benefits

- **Logical organization**: Test structure mirrors conceptual framework
- **Easy navigation**: Related tests grouped together
- **Scalability**: Easy to add new tests to appropriate categories
- **Maintainability**: Clear separation of concerns
- **Consistency**: Follows existing `statistical_validation/` pattern

### Precedent

The `statistical_validation/` directory already demonstrates this organizational approach with 16+ specialized validation files grouped under the "Statistical Correctness" testset.

## WARNING: Method definition - Minimal Solution Plan

The test suite generates 100+ "Method definition overwritten" warnings due to utility functions being included multiple times across different test files. These warnings don't affect functionality but create noise that obscures real issues.

### Minimal Solution: Centralized Include Strategy

The cleanest, minimal fix is to **centralize utility includes** in `runtests.jl` while keeping all existing test file structure intact.

**Current Problem:**
- `testing_utilities.jl` included by 16+ files → functions redefined 16+ times
- `bootstrap_se_validation.jl` included by 5+ files → functions redefined 5+ times  
- `analytical_se_validation.jl` included by multiple files → more redefinitions

**Proposed Solution:**

- [x] 1. **Add utility includes to runtests.jl** (after existing using statements):
   ```julia
   include("statistical_validation/testing_utilities.jl")
   include("statistical_validation/bootstrap_se_validation.jl") 
   include("statistical_validation/analytical_se_validation.jl")
   ```

- [ ] 2. **Remove utility includes from test files** - Remove these lines from files in `statistical_validation/`:
   - `include("testing_utilities.jl")`
   - `include("bootstrap_se_validation.jl")`
   - `include("analytical_se_validation.jl")`

- [ ] 3. **Keep legitimate test includes** - Preserve includes that load actual test files:
   - `include("multi_model_bootstrap_tests.jl")` ✓ Keep
   - `include("bootstrap_validation_tests.jl")` ✓ Keep
   - `include("robust_se_validation.jl")` ✓ Keep

**Benefits:**
- Zero method definition warnings (functions defined once, used everywhere)
- Minimal changes (no file reorganization or architectural changes)
- Preserved functionality (all existing tests intact)
- Easy to revert (simple to undo if needed)

**Expected result:** 100+ warnings → 0 warnings, 714/714 tests passing

## More testing?

