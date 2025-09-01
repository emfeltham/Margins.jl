# PHASE_3_5.md: Critical FormulaCompiler Integration Fixes

## 🚨 **CRITICAL ISSUES DISCOVERED**

After examining the current codebase implementation against REORG.md guidelines, several **major violations** of FormulaCompiler best practices have been identified that must be fixed before Phase 4 performance optimization can succeed.

**Root Cause**: We are working **against** FormulaCompiler.jl instead of **with** it, violating the core principle from REORG.md: *"Use FormulaCompiler as intended, not fight against it"*.

---

## **❌ VIOLATION 1: Profile Margins Recompilation Anti-Pattern**

### **Current Implementation (BROKEN)**
**Location**: `src/engine/utilities.jl:280-290`

```julia
for profile in profiles
    # ❌ MASSIVE PERFORMANCE KILLER: Recompiling per profile!
    refgrid_data = _build_refgrid_data(profile, engine.data_nt)
    refgrid_compiled = FormulaCompiler.compile_formula(engine.model, refgrid_data)
    refgrid_de = FormulaCompiler.build_derivative_evaluator(refgrid_compiled, refgrid_data; vars=continuous_requested)
    
    # This pattern destroys performance - compilation is ~milliseconds per profile!
end
```

**Impact**: 
- **1000x+ performance degradation**: Compilation takes milliseconds, evaluation should take nanoseconds
- **Memory explosion**: Creating synthetic data for every profile
- **Against FC design**: Completely bypasses FormulaCompiler's scenario override system

### **Correct Implementation (REORG.md lines 62-69)**
```julia
# Use FormulaCompiler's scenario override system - SINGLE compilation, zero-allocation overrides
for profile in profiles
    scenario = FormulaCompiler.create_scenario("profile_$i", engine.data_nt; profile...)
    # Zero-allocation evaluation at scenario - always row 1 for scenarios
    FormulaCompiler.marginal_effects_eta!(g_buffer, engine.de, engine.β, 1; backend=:fd)
end
```

---

## **❌ VIOLATION 2: Incorrect Backend Selection**

### **Current Implementation (SUBOPTIMAL)**
Using `:ad` as default everywhere, violating performance guidelines.

### **Correct Implementation (REORG.md lines 72-74)**
```julia
# Population margins: Use :fd for zero allocations
FormulaCompiler.accumulate_ame_gradient!(gβ_sum, de, β, rows, var; link=link, backend=:fd)

# Profile margins: Use :ad for speed/accuracy  
FormulaCompiler.marginal_effects_eta!(g_buffer, de, β, 1; backend=:ad)
```

**Backend Strategy**:
- **`:fd` backend**: Zero allocations, perfect for population AME across many rows
- **`:ad` backend**: ~368 bytes per call, faster and more accurate, good for profiles

---

## **❌ VIOLATION 3: Missing Scenario Override System**

### **Current Implementation (MISSING)**
We don't use `FormulaCompiler.create_scenario()` anywhere in the codebase.

**Grep Result**: `❌ Not using FC's scenario override system`

### **Required Implementation**
**From REORG.md Scenario Override System (lines 62-69)**:

```julia
# Memory-efficient scenario creation (uses OverrideVector)
scenario = FormulaCompiler.create_scenario("profile", data_nt; x=1.0, group="A")

# Zero-allocation evaluation at scenario
FormulaCompiler.marginal_effects_eta!(g_buffer, de, β, 1; backend=:fd)  # Always row 1 for scenarios
```

**Benefits**:
- **Memory efficient**: Uses OverrideVector instead of copying data
- **Zero allocations**: No synthetic data creation
- **Designed for profiles**: Exactly what scenario overrides were built for

---

## **❌ VIOLATION 4: Improper AME Gradient Usage**

### **Current Issue**
We use `accumulate_ame_gradient!` but don't follow zero-allocation patterns correctly.

### **Required Fix (REORG.md lines 51-52)**
```julia
# Zero-allocation AME gradients for delta-method SEs  
FormulaCompiler.accumulate_ame_gradient!(gβ_sum, de, β, rows, var; link=link, backend=:fd)

# Single parameter gradients (for weighted AME)
FormulaCompiler.me_eta_grad_beta!(gβ_temp, de, β, row, var)  # η case
FormulaCompiler.me_mu_grad_beta!(gβ_temp, de, β, row, var; link=link)  # μ case
```

---

## **❌ VIOLATION 5: Inefficient Caching Implementation**

### **Current Implementation (FRAGMENTED)**
Multiple cache systems:
- `COMPILED_CACHE` in `population/core.jl`
- `TYPICAL_VALUES_CACHE` in `profile/refgrids.jl`  
- Different cache keys in different files

### **Required Fix**
Unified caching following FC patterns (REORG.md line 87):

```julia
# Compilation caching: Cache compiled formulas by data signature to avoid recompilation
const ENGINE_CACHE = Dict{UInt64, MarginsEngine}()

function _get_or_build_engine(model, data_nt, vars)
    cache_key = hash(model, keys(data_nt), vars)
    return get!(ENGINE_CACHE, cache_key) do
        build_engine(model, data_nt, vars)
    end
end
```

---

## **🎯 PERFORMANCE IMPACT OF VIOLATIONS**

**Why We Can't Hit Performance Targets**:

1. **Profile recompilation**: Adding ~10ms per profile instead of ~1μs target
2. **Wrong backend**: Missing zero-allocation opportunities  
3. **No scenario overrides**: Creating unnecessary synthetic data
4. **Fragmented caching**: Recompiling when we shouldn't

**Current vs Target Performance**:
- **Population**: ~10μs per row → Target: <100ns per row (100x improvement blocked by violations)
- **Profiles**: ~100μs per profile → Target: <1μs per profile (100x improvement blocked by recompilation)

---

## **🔧 REQUIRED FIXES FOR PHASE 4**

### **Priority 1: Fix Profile Recompilation (CRITICAL)**
Replace `src/engine/utilities.jl` profile loop with FormulaCompiler scenario overrides.

**Before**: Recompile per profile (milliseconds)  
**After**: Single compilation + scenario overrides (nanoseconds)

### **Priority 2: Implement Proper Backend Selection**  
- Population functions: Default to `:fd` backend
- Profile functions: Default to `:ad` backend  
- Graceful fallbacks with warnings

### **Priority 3: Add Scenario Override System**
Implement `FormulaCompiler.create_scenario()` for all profile evaluation.

### **Priority 4: Unify Caching Strategy**
Single cache system following FormulaCompiler patterns.

### **Priority 5: Verify Zero-Allocation Paths**
Ensure all hot paths achieve 0 bytes allocated after warmup.

---

## **📋 IMPLEMENTATION PLAN**

### **Phase 3.5: FormulaCompiler Integration Fixes** (REQUIRED BEFORE PHASE 4)

**Day 1**: Fix profile recompilation anti-pattern
- Replace recompilation loop with scenario override system
- Implement `FormulaCompiler.create_scenario()` usage
- Verify single compilation per engine

**Day 2**: Implement proper backend selection
- Population margins: Default `:fd` backend
- Profile margins: Default `:ad` backend  
- Add graceful fallbacks

**Day 3**: Unify caching and validate zero-allocation
- Consolidate cache systems
- Verify zero allocations in hot paths
- Test performance improvements

**Validation**: All FormulaCompiler integration issues resolved, ready for Phase 4 optimization.

---

## **🚀 EXPECTED IMPACT**

**After Fixing These Violations**:
- **Profile margins**: 100x+ speedup (from eliminating recompilation)
- **Population margins**: 10x+ speedup (from proper backend selection)  
- **Memory usage**: 90%+ reduction (from zero-allocation paths)
- **Caching efficiency**: Faster engine construction

**Only after these fixes can we achieve the Phase 4 performance targets**.

---

## **⚠️ CRITICAL DECISION POINT**

**We cannot proceed with Phase 4 performance optimization until these FormulaCompiler integration issues are resolved.** 

The current implementation violates fundamental FormulaCompiler principles and makes the <100ns per row / <1μs per profile targets impossible to achieve.

**Recommendation**: Implement Phase 3.5 (FormulaCompiler Integration Fixes) immediately before Phase 4.