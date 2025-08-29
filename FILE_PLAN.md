# File Reorganization Plan for Margins.jl

This document outlines a comprehensive reorganization of the Margins.jl codebase to improve maintainability, clarity, and logical structure now that the package has reached production readiness.

## Current Structure Analysis

### Current Source Files (src/):
- `Margins.jl` (28 lines) - Main module file
- `api.jl` (706 lines) - **LARGE** - Contains both API functions and helper utilities
- `compute_continuous.jl` (234 lines) - Continuous marginal effects computation  
- `compute_categorical.jl` (102 lines) - Categorical contrasts computation
- `predictions.jl` (133 lines) - Prediction computations (APE/APM/APR)
- `engine_fc.jl` (166 lines) - FormulaCompiler integration and model info
- `profiles.jl` (119 lines) - Profile grid building and processing
- `results.jl` (114 lines) - Result types and formatting
- `categorical_mixtures.jl` (199 lines) - Categorical mixture support
- `link.jl` (42 lines) - Link function utilities

### Issues with Current Structure:
1. **`api.jl` is too large** (706 lines) - contains API, utilities, and helper functions
2. **Flat structure** - no logical grouping of related functionality
3. **Mixed concerns** - utilities scattered throughout files
4. **No clear separation** between public API and internal implementation
5. **Large test directory** with many individual test files

## Proposed New Structure

### 1. Core API Layer
```
src/
├── api/
│   ├── population.jl         # population_margins() function and dispatch  
│   ├── profile.jl            # profile_margins() functions and dispatch
│   └── common.jl             # Shared API utilities and helpers
```

### 2. Computation Engine Layer  
```
src/
├── computation/
│   ├── continuous.jl         # Continuous marginal effects (AME/MEM/MER)
│   ├── categorical.jl        # Categorical contrasts and discrete changes  
│   ├── predictions.jl        # Adjusted predictions (APE/APM/APR)
│   └── engine.jl             # FormulaCompiler integration (renamed from engine_fc.jl)
```

### 3. Core Infrastructure Layer
```
src/
├── core/
│   ├── results.jl           # MarginsResult type and builders
│   ├── profiles.jl          # Profile grid building and at parameter processing
│   ├── grouping.jl          # Grouping utilities (_build_groups, _split_by, etc.)
│   ├── utilities.jl         # General utilities (_subset_data, _resolve_weights, etc.)
│   └── link.jl              # Link function utilities
```

### 4. Advanced Features Layer
```  
src/
├── features/
│   ├── categorical_mixtures.jl   # Categorical mixture support
│   ├── averaging.jl             # Proper delta method averaging for profiles
│   └── covariance.jl            # Robust/clustered standard error handling
```

### 5. Updated Main Module
```
src/
├── Margins.jl               # Main module with clean includes
```

## Detailed File Splits and Moves

### Split `api.jl` (706 lines → ~200 lines each):

#### `src/api/population.jl` (~180 lines):
- `population_margins()` function
- Population-specific dispatch logic  
- Population grouping implementation

#### `src/api/profile.jl` (~200 lines):  
- `profile_margins()` functions (both dispatch methods)
- Profile-specific dispatch logic
- Profile grouping implementation

#### `src/api/common.jl` (~100 lines):
- `_try_dof_residual()`
- `_add_ci!()` function
- Common parameter validation
- Shared API utilities

#### `src/core/grouping.jl` (~120 lines):
- `_build_groups()`
- `_split_by()`
- `_subset_data()`
- All grouping-related utilities

#### `src/core/utilities.jl` (~100 lines):
- `_resolve_weights()`, `_merge_weights()`, `_balanced_weights()`
- `_resolve_vcov()`
- `_nrows()` and other general utilities

#### `src/features/averaging.jl` (~100 lines):
- `_average_profiles_with_proper_se()`
- Gradient averaging logic
- Delta method utilities for profile averaging

### Rename Files:
- `engine_fc.jl` → `computation/engine.jl` (clearer name)
- `compute_continuous.jl` → `computation/continuous.jl`  
- `compute_categorical.jl` → `computation/categorical.jl`

### Move Files:
- `predictions.jl` → `computation/predictions.jl`
- `profiles.jl` → `core/profiles.jl`
- `results.jl` → `core/results.jl` 
- `link.jl` → `core/link.jl`
- `categorical_mixtures.jl` → `features/categorical_mixtures.jl`

## Test Directory Reorganization

### Current Test Issues:
- 20+ individual test files in flat structure
- Some redundant/outdated test files
- Difficult to navigate and maintain

### Proposed Test Structure:
```
test/
├── runtests.jl              # Main test runner
├── api/
│   ├── test_population_api.jl     # population_margins() API tests
│   ├── test_profile_api.jl        # profile_margins() API tests  
│   └── test_grouping.jl           # Grouping functionality tests
├── computation/
│   ├── test_continuous_effects.jl # Continuous marginal effects
│   ├── test_categorical_effects.jl # Categorical contrasts
│   └── test_predictions.jl        # Prediction computations
├── integration/
│   ├── test_glm_integration.jl    # GLM compatibility
│   ├── test_mixed_datatypes.jl    # Int64/Bool/Float64 handling
│   └── test_link_scales.jl        # Link scale computations
├── features/
│   ├── test_categorical_mixtures.jl # Categorical mixture functionality
│   ├── test_averaging.jl          # Profile averaging with proper SEs
│   └── test_robust_vcov.jl        # Robust standard errors
└── performance/
    ├── test_allocations.jl        # Allocation benchmarks
    └── benchmarks.jl              # Performance benchmarks
```

### Test Files to Consolidate/Remove:
- `test_bool_profiles.jl` → merge into `test_mixed_datatypes.jl`
- `test_core_function.jl` → merge into appropriate computation tests
- `scratch.jl`, `additional_tests.jl` → remove or consolidate
- `*_allocations.jl` files → move to `performance/`

## Documentation Reorganization

### Current Docs Issues:
- Mixed documentation and planning files in root
- Some outdated planning documents

### Proposed Structure:
```
docs/
├── src/
│   ├── api.md               # API documentation (existing)
│   ├── reference_grids.md   # Reference grid documentation (existing)  
│   ├── index.md             # Main documentation (existing)
│   ├── examples.md          # Usage examples
│   └── internals.md         # Internal architecture documentation
├── planning/                # Move planning docs here
│   ├── MARGINS_PLAN.md      # Historical planning
│   ├── API_PLAN.md          # API design history  
│   ├── GRADIENT_STORAGE_PLAN.md # Technical planning
│   └── statistical_framework.md # Statistical background
└── archive/                 # Archive completed/outdated docs
    ├── CURRENT_ISSUES.md    # Historical issue tracking
    └── GLMMs.md             # Future extension plans
```

## Benefits of Proposed Reorganization

### 1. **Logical Separation of Concerns**:
- Clear separation between API, computation, core infrastructure, and features
- Each directory has a focused responsibility

### 2. **Improved Maintainability**:
- Smaller, focused files easier to understand and modify
- Related functionality grouped together
- Clear dependency hierarchy

### 3. **Better Navigation**:
- Developers can quickly find relevant code
- Logical structure matches mental model of package architecture
- Clear public vs private API separation  

### 4. **Easier Testing**:
- Tests organized to match source structure
- Integration vs unit tests clearly separated
- Performance tests isolated

### 5. **Enhanced Documentation**:
- Planning documents separated from user documentation
- Clear archival of historical development
- Better organization for future contributors

## Migration Steps

### Phase 1: Create New Directory Structure
1. Create new subdirectories in `src/`
2. Create new test organization

### Phase 2: Split and Move Files  
1. Split `api.jl` into focused components
2. Move files to appropriate directories
3. Update include statements in `Margins.jl`

### Phase 3: Update Tests
1. Reorganize test files to match new structure
2. Update `runtests.jl` to use new test organization
3. Remove redundant test files

### Phase 4: Update Documentation
1. Move planning documents to appropriate locations
2. Update documentation to reflect new structure
3. Archive resolved historical documents

### Phase 5: Validation
1. Run full test suite to ensure no regressions
2. Update CI/CD if needed
3. Update development documentation

## Implementation Priority

### High Priority (Essential):
- Split `api.jl` - too large and mixing concerns
- Create logical `src/` subdirectory structure  
- Organize test files for better maintainability

### Medium Priority (Beneficial):
- Move planning documents out of root
- Create clear documentation structure
- Archive resolved historical files

### Low Priority (Nice to have):
- Performance test isolation
- Advanced feature separation
- Detailed internal documentation

This reorganization will position Margins.jl as a well-structured, maintainable, and professional Julia package ready for long-term development and community contribution.