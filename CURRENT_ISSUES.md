# Current Implementation Issues in Margins.jl

**⚠️ STATISTICAL CORRECTNESS IS PARAMOUNT ⚠️**

This document tracks known limitations, approximations, and missing functionality in the current Margins.jl implementation. While the package is production-ready for most use cases, these issues represent **statistical validity gaps** that must be addressed to ensure rigorous econometric inference.

**Critical Principle**: Any approximation or fallback that affects standard error computation compromises statistical inference validity. Users relying on confidence intervals, hypothesis tests, or p-values need mathematically correct uncertainty estimates.

---

## **RESOLVED ISSUES** (Previously Listed as Active)

### **Categorical Mixture Handling** **WORKING CORRECTLY**

#### Former Issue #3: **Categorical Mixture Standard Errors** - **RESOLVED**
**Status**: **Working as Designed** - FormulaCompiler's `CategoricalMixtureOverride` system handles mixtures correctly  
**Resolution Date**: August 2025  

**What Works**:
- [x] **Predictions at mixtures**: `profile_margins(..., type=:predictions)` with mixture profiles work correctly
- [x] **Continuous effects at mixtures**: `profile_margins(..., type=:effects, vars=[:continuous_var])` at mixture profiles work correctly  
- [x] **Proper statistical foundation**: FormulaCompiler's `CategoricalMixtureOverride` uses weighted contrast computation, not arbitrary encoding

**What Correctly Fails**:
- ❌ **Categorical effects at mixtures**: `profile_margins(..., type=:effects, vars=[:categorical_var])` correctly fails because categorical effects are contrasts between specific levels - mixtures don't make statistical sense for effects

**Technical Implementation**:
FormulaCompiler automatically detects `MixtureWithLevels` objects and creates `CategoricalMixtureOverride` vectors that enable proper weighted contrast computation in the execution engine. The arbitrary encoding in `_mixture_to_scenario_value()` is bypassed.

#### Former Issue #4: **FormulaCompiler Best Practices** - **ALREADY IMPLEMENTED**
**Status**: **Architecture Correctly Implemented**  
**Resolution**: Margins.jl already follows FormulaCompiler's best practices:

- [x] **Continuous filtering**: `src/computation/engine.jl` properly uses `continuous_variables()` filtering
- [x] **Derivative evaluators**: Only built for continuous variables (lines 13-23 in engine.jl)
- [x] **Scenario system**: `create_scenario()` properly used throughout prediction and effect computation
- [x] **Error handling**: FormulaCompiler provides appropriate error messages for invalid operations

#### Former Issue #5: **Table-Based Categorical Effects** - **RESOLVED**
**Status**: **Fixed Through Proper FormulaCompiler Usage**  
**Resolution Date**: August 2025  
**Location**: Fixed in `src/computation/categorical.jl` (both `_categorical_effects` and `_categorical_effects_from_profiles`)

**Root Cause**: Margins.jl was incorrectly calling `contrast_modelrow!` with scenario data containing overrides. FormulaCompiler's `contrast_modelrow!` is designed to work with regular data, not scenario overrides.

**The Fix**: Implemented proper scenario-based contrast approach:
```julia
# OLD (incorrect): Try to use contrast_modelrow! with scenario data
FormulaCompiler.contrast_modelrow!(Δ, compiled, scen.data, 1; var=:cat, from="A", to="B")

# NEW (correct): Create separate scenarios and evaluate difference
prof_from[:cat] = "A"; prof_to[:cat] = "B"
scen_from = create_scenario("from", data, prof_from)
scen_to = create_scenario("to", data, prof_to)
compiled(X_from, scen_from.data, 1); compiled(X_to, scen_to.data, 1)
Δ .= X_to .- X_from  # Proper contrast computation
```

**Impact**: All categorical effects now work correctly:
- [x] **Population categorical effects**: Average marginal effects across sample
- [x] **Dict-based profile effects**: `profile_margins(model, data; at=Dict(:cat => "A"))`
- [x] **Table-based profile effects**: `profile_margins(model, data, reference_grid)`
- [x] **Both η and μ targets**: Link-scale and response-scale effects

## [x] **ALL ISSUES RESOLVED** (August 2025)

All previously identified statistical validity concerns have been successfully resolved.

---

## 📊 **IMPACT ASSESSMENT**

### **Statistical Validity Impact**
- **[x] RESOLVED**: Issues #1, #2 (grouped averaging, vcov fallback) **BOTH COMPLETED**
- **[x] RESOLVED**: Issue #3 - Categorical mixtures **WORKING CORRECTLY** (FormulaCompiler integration verified)
- **[x] RESOLVED**: Issue #4 - Architecture **ALREADY FOLLOWING BEST PRACTICES** (verification completed)
- **[x] RESOLVED**: Issue #5 - Table-based categorical effects **FIXED** (proper scenario-based contrast approach implemented)
- **[x] RESOLVED**: Issue #6 - Elasticity method **VALIDATED** as standard practice (no changes needed)

### **User Experience Impact**
- **[x] RESOLVED**: Silent statistical failures **ELIMINATED**
- **[x] RESOLVED**: Issue #3 - Categorical mixtures **STATISTICALLY SOUND** (proper FormulaCompiler integration)
- **[x] RESOLVED**: Issue #4 - Architecture **FOLLOWING ESTABLISHED PATTERNS** (zero-allocation preserved)
- **[x] RESOLVED**: Issue #5 - Table-based functionality **FULLY FUNCTIONAL** (all categorical effects working)
- **[x] RESOLVED**: Issue #6 - Elasticity computation **NO CONCERNS** (standard econometric practice)

---

## 🎯 **RESOLUTION TIMELINE**

### **COMPLETED (August 2025):**
- [x] **Issues #1, #2**: Grouped averaging and vcov fallback (completed earlier)
- [x] **Issue #3**: Categorical mixtures verified as working correctly through FormulaCompiler integration
- [x] **Issue #4**: Architecture verified as already following FormulaCompiler best practices  
- [x] **Issue #5**: Table-based categorical effects fixed by implementing proper scenario-based contrast approach
- [x] **Issue #6**: Elasticity method validated as standard econometric practice

### **🎯 CURRENT STATUS: ALL ISSUES RESOLVED**
All statistical validity concerns successfully resolved. Package ready for production use with full categorical effects support.

---

## 🔒 **ZERO TOLERANCE POLICY**

**FUNDAMENTAL PRINCIPLE**: Margins.jl maintains the highest standards of statistical rigor. Any approximation or fallback that affects standard error computation **must error out** rather than provide invalid results.

**🎯 ONGOING COMMITMENT**: All future development must adhere to this principle. **Wrong standard errors are worse than no standard errors.**
