# REORG.md: Margins.jl Reorganization Plan

## Overview

After analyzing the current Margins.jl codebase and reviewing the FormulaCompiler MARGINS_GUIDE.md, several critical design mistakes and complexity issues have been identified that violate FormulaCompiler.jl's core design principles and create unnecessary complexity. This package often works against FormulaCompiler.jl rather than building on it.

## 🎯 **Project Priorities (In Order)**

Based on user requirements, the reorganization will prioritize:

1. **Statistical Correctness (PARAMOUNT)**: 
   - All standard errors must remain mathematically correct
   - Delta-method computations must be exact
   - No approximations that compromise statistical validity
   - Bootstrap validation required for any SE changes

2. **Performance (PARAMOUNT)**: 
   - Achieve FormulaCompiler's ~50ns zero-allocation evaluation
   - Eliminate all unnecessary memory allocations
   - Target 100x+ speedup for typical use cases
   - Continuous benchmarking and allocation tracking

3. **Julian Style / Elegant Approach**:
   - Simple, composable APIs following Julia conventions
   - Multiple dispatch where appropriate
   - Type stability throughout
   - Clean, readable code that leverages Julia's strengths

4. **Proper FormulaCompiler Integration (MUST)**:
   - Use FormulaCompiler as intended, not fight against it
   - Leverage zero-allocation evaluation paths
   - Use scenario override system efficiently  
   - Follow FormulaCompiler's architectural patterns

**Note**: Breaking changes are acceptable if they serve these priorities. Backward compatibility is not important.

## 🔧 **FormulaCompiler API Analysis** 

Having full access to FormulaCompiler source (`/Users/emf/.julia/dev/FormulaCompiler`), the key APIs for proper integration are:

### **Core Zero-Allocation APIs:**
```julia
# Compilation and evaluation
compiled = FormulaCompiler.compile_formula(model, data_nt)
de = FormulaCompiler.build_derivative_evaluator(compiled, data_nt; vars=continuous_vars)

# Zero-allocation marginal effects (choose backend)
FormulaCompiler.marginal_effects_eta!(g_buffer, de, β, row; backend=:fd)  # 0 bytes
FormulaCompiler.marginal_effects_mu!(g_buffer, de, β, row; link=link, backend=:fd)  # 0 bytes

# Zero-allocation AME gradients for delta-method SEs  
FormulaCompiler.accumulate_ame_gradient!(gβ_sum, de, β, rows, var; link=link, backend=:fd)

# Single parameter gradients (for weighted AME)
FormulaCompiler.me_eta_grad_beta!(gβ_temp, de, β, row, var)  # η case
FormulaCompiler.me_mu_grad_beta!(gβ_temp, de, β, row, var; link=link)  # μ case

# Delta-method standard errors
se = FormulaCompiler.delta_method_se(gradient, Σ)
```

### **Scenario Override System (Perfect for Profiles):**
```julia
# Memory-efficient scenario creation (uses OverrideVector)
scenario = FormulaCompiler.create_scenario("profile", data_nt; x=1.0, group="A")

# Zero-allocation evaluation at scenario
FormulaCompiler.marginal_effects_eta!(g_buffer, de, β, 1; backend=:fd)  # Always row 1 for scenarios
```

### **Backend Selection Strategy:**
- **`:fd` backend**: Zero allocations, slightly slower, perfect for population AME
- **`:ad` backend**: ~368 bytes per call, faster, more accurate, good for profiles
- **Recommendation**: Use `:fd` for zero-allocation requirements, `:ad` for speed/accuracy

### **Key Insights from MARGINS_GUIDE.md:**
1. **FormulaCompiler already provides AME gradient accumulation** - don't reimplement!
2. **Reference grids vs scenario overrides**: 
   - **Reference grids** (recommended): Build minimal synthetic data for clean evaluation points (MER/MEM)
   - **Scenario overrides**: Use for counterfactual analysis preserving data correlations (AMER)
3. **Backend selection allows performance tuning** - use `:fd` for strict zero-allocation paths
4. **All delta-method computation is built-in** - just call `delta_method_se(gradient, Σ)`
5. **Batch operations**: Pre-allocate output matrices and use `view()` for in-place evaluation
6. **Input validation**: Validate continuous vs categorical variables early at API boundary
7. **Graceful fallbacks**: Try `:ad` first, fall back to `:fd` on failure with warning
8. **Dual interface pattern**: Provide both `!` (in-place) and allocating versions
9. **Compilation caching**: Cache compiled formulas by data signature to avoid recompilation
10. **Consistent data formats**: Always use `Tables.columntable()` format throughout

## 🚨 **Critical Design Mistakes Identified**

### **1. Massive Memory Allocation Anti-patterns** 
**Current Problem**: The code violates FormulaCompiler's zero-allocation philosophy with extensive allocation inside hot loops.

**Evidence**:
- `continuous.jl:88-92`: `xbuf = Vector{Float64}(undef, length(compiled))` allocated per row in elasticity calculations
- `continuous.jl:44`: `g_row = Vector{Float64}(undef, length(vars))` allocated per variable per iteration
- `continuous.jl:59`: `gβ_temp = Vector{Float64}(undef, length(β))` allocated per row in weighted case

**Impact**: Destroys the ~50ns zero-allocation performance that FormulaCompiler provides.

### **2. Duplicate Profile Systems**
**Current Problem**: Two completely separate profile building systems exist simultaneously.

**Evidence**:
- `core/profiles.jl`: Old `_build_profiles()` system with Dict-based Cartesian products
- `core/refgrid.jl`: New "Phase 3" iterator builders (`refgrid_means`, `refgrid_cartesian`)
- Both systems implement similar logic with incompatible APIs

**Impact**: Code duplication, maintenance burden, API confusion.

### **3. Inefficient Data Conversion Pattern**
**Current Problem**: Multiple `DataFrame → NamedTuple` conversions at every API entry point.

**Evidence**:
- `api/profile.jl:?` (implied from engine.jl:9)
- `computation/engine.jl:9`: `data_nt = Tables.columntable(data)`
- Multiple conversion points throughout the codebase

**Impact**: Unnecessary allocations and CPU overhead.

### **4. Over-Engineered Architecture**
**Current Problem**: The codebase has grown into an overly complex layered architecture that obscures the simple 2×2 framework.

**Evidence**:
- 17+ source files for what should be a simple statistical interface
- Complex profile → iterator → DataFrame → processing chains
- Multiple abstraction layers that add no value

**Impact**: Difficult to understand, maintain, and debug.

### **5. Missing FormulaCompiler Integration Opportunities**
**Current Problem**: Not leveraging FormulaCompiler's scenario override system properly.

**Evidence**:
- Building synthetic data manually instead of using `create_scenario()`
- Complex refgrid construction instead of efficient `OverrideVector` patterns
- Missing opportunities for zero-allocation scenario evaluation

## 📋 **Aggressive Reorganization Plan** 

*Breaking changes acceptable - prioritizing statistical correctness, performance, and proper FormulaCompiler integration*

**Implementation Philosophy**: Overwrite existing functions directly with clean names. No ugly suffixes like `_zero_alloc` or parallel implementations. Replace `_ame_continuous()`, `_mem_mer_continuous()`, `_build_profiles()` etc. in-place using proper FormulaCompiler patterns.

### **Phase 1: Core Engine with Zero-Allocation Architecture**
*Target: Build proper FormulaCompiler foundation from scratch*

1. **Create New `src/engine.jl`** using actual FormulaCompiler APIs:
   ```julia
   # Zero-allocation engine leveraging FormulaCompiler's built-in capabilities
   struct MarginsEngine{L<:GLM.Link}
       # FormulaCompiler components (pre-compiled)
       compiled::FormulaCompiler.CompiledFormula
       de::Union{FormulaCompiler.DerivativeEvaluator, Nothing}
       
       # Pre-allocated buffers (FormulaCompiler already has internal buffers!)
       g_buf::Vector{Float64}              # For marginal effects results
       gβ_accumulator::Vector{Float64}     # For AME gradient accumulation
       
       # Model parameters
       β::Vector{Float64}
       Σ::Matrix{Float64}
       link::L
       vars::Vector{Symbol}
       data_nt::NamedTuple  # Keep reference for scenarios
   end
   
   # Zero-allocation constructor using FormulaCompiler properly
   function build_engine(model, data_nt::NamedTuple, vars)::MarginsEngine
       # Input validation (MARGINS_GUIDE.md recommendation)
       _validate_variables(data_nt, vars)
       
       compiled = FormulaCompiler.compile_formula(model, data_nt)
       continuous_vars = FormulaCompiler.continuous_variables(compiled, data_nt)
       vars_for_de = filter(v -> v in continuous_vars, vars)
       
       # Only build derivative evaluator if needed (FC handles this correctly)
       de = isempty(vars_for_de) ? nothing : 
            FormulaCompiler.build_derivative_evaluator(compiled, data_nt; vars=vars_for_de)
       
       # Minimal buffer allocation (FC's DerivativeEvaluator has internal buffers)
       g_buf = Vector{Float64}(undef, length(vars_for_de))
       gβ_accumulator = Vector{Float64}(undef, length(compiled))
       
       return MarginsEngine(compiled, de, g_buf, gβ_accumulator, 
                           coef(model), vcov(model), _auto_link(model), vars, data_nt)
   end
   
   # Input validation (MARGINS_GUIDE.md pattern)
   function _validate_variables(data_nt::NamedTuple, vars)
       for var in vars
           haskey(data_nt, var) || error("Variable $var not found in data")
           col = getproperty(data_nt, var)
           if col isa CategoricalArray && vars !== :continuous
               @warn "Variable $var is categorical. Use contrasts for categorical effects."
           end
       end
   end
   ```

2. **Zero-Allocation Population Effects (AME)** with graceful fallbacks and batch operations:
   ```julia
   function _ame_continuous(engine::MarginsEngine, data_nt; target=:mu, backend=:ad)
       engine.de === nothing && return (DataFrame(), Matrix{Float64}(undef, 0, length(engine.β)))
       
       rows = 1:nrows(data_nt)
       results = DataFrame(term=String[], estimate=Float64[], se=Float64[])
       G = Matrix{Float64}(undef, length(engine.de.vars), length(engine.β))
       
       # Use FormulaCompiler's built-in AME gradient accumulation (ZERO ALLOCATION!)
       for (i, var) in enumerate(engine.de.vars)
           # Graceful backend fallback (MARGINS_GUIDE.md pattern)
           try
               # This is the key: FC already provides zero-allocation AME gradients
               FormulaCompiler.accumulate_ame_gradient!(
                   engine.gβ_accumulator, engine.de, engine.β, rows, var;
                   link=(target === :mu ? engine.link : GLM.IdentityLink()), 
                   backend=backend
               )
           catch e
               if backend === :ad
                   @warn "AD backend failed for $var, falling back to FD: $e"
                   FormulaCompiler.accumulate_ame_gradient!(
                       engine.gβ_accumulator, engine.de, engine.β, rows, var;
                       link=(target === :mu ? engine.link : GLM.IdentityLink()), 
                       backend=:fd
                   )
               else
                   rethrow(e)
               end
           end
           
           # Average the gradient and compute SE
           gβ_avg = engine.gβ_accumulator ./ length(rows)
           se = FormulaCompiler.delta_method_se(gβ_avg, engine.Σ)
           
           # Compute AME value (also zero allocation with FC)
           ame_val = 0.0
           for row in rows
               if target === :mu
                   FormulaCompiler.marginal_effects_mu!(engine.g_buf, engine.de, engine.β, row; 
                                                       link=engine.link, backend=backend)
               else
                   FormulaCompiler.marginal_effects_eta!(engine.g_buf, engine.de, engine.β, row; 
                                                        backend=backend)
               end
               ame_val += engine.g_buf[i]
           end
           ame_val /= length(rows)
           
           push!(results, (term=string(var), estimate=ame_val, se=se))
           G[i, :] = gβ_avg
       end
       
       return (results, G)
   end
   ```

3. **Profile Effects (MEM) Using Reference Grids** (following FormulaCompiler guide):
   ```julia
   function _mem_continuous(engine::MarginsEngine, profiles; target=:mu, backend=:ad)
       engine.de === nothing && return (DataFrame(), Matrix{Float64}(undef, 0, length(engine.β)))
       
       results = DataFrame(term=String[], estimate=Float64[], se=Float64[])
       n_profiles = length(profiles)
       n_vars = length(engine.de.vars)
       G = Matrix{Float64}(undef, n_profiles * n_vars, length(engine.β))
       
       row_idx = 1
       for profile in profiles
           # Build minimal synthetic reference grid data (FormulaCompiler guide recommendation)
           refgrid_data = _build_refgrid_data(profile, engine.data_nt)
           refgrid_compiled = FormulaCompiler.compile_formula(engine.compiled.model, refgrid_data)
           refgrid_de = FormulaCompiler.build_derivative_evaluator(refgrid_compiled, refgrid_data; 
                                                                  vars=engine.de.vars)
           
           for (var_idx, var) in enumerate(engine.de.vars)
               # Evaluate at clean synthetic reference point (row 1 of refgrid)
               if target === :mu
                   FormulaCompiler.marginal_effects_mu!(engine.g_buf, refgrid_de, engine.β, 1;
                                                       link=engine.link, backend=backend)
               else
                   FormulaCompiler.marginal_effects_eta!(engine.g_buf, refgrid_de, engine.β, 1;
                                                        backend=backend)
               end
               effect_val = engine.g_buf[var_idx]
               
               # Compute gradient for SE at reference point
               if target === :mu
                   FormulaCompiler.me_mu_grad_beta!(engine.gβ_accumulator, refgrid_de, engine.β, 1, var;
                                                   link=engine.link)
               else
                   FormulaCompiler.me_eta_grad_beta!(engine.gβ_accumulator, refgrid_de, engine.β, 1, var)
               end
               se = FormulaCompiler.delta_method_se(engine.gβ_accumulator, engine.Σ)
               
               # Build profile description
               profile_desc = join(["$(k)=$(v)" for (k,v) in pairs(profile)], ", ")
               term_name = "$(var) at $(profile_desc)"
               
               push!(results, (term=term_name, estimate=effect_val, se=se))
               G[row_idx, :] = engine.gβ_accumulator
               row_idx += 1
           end
       end
       
       return (results, G)
   end
   
   # Helper: Build minimal reference grid data (not scenario overrides)
   function _build_refgrid_data(profile::Dict, original_data::NamedTuple)
       # Create minimal synthetic data with only needed variables
       refgrid = NamedTuple()
       for (var, val) in pairs(original_data)
           if haskey(profile, var)
               # Use profile value for this variable
               refgrid = merge(refgrid, NamedTuple{(var,)}(([profile[var]],)))
           else
               # Use representative value (mean for continuous, first level for categorical)
               if eltype(val) <: Real && !(eltype(val) <: Bool)
                   refgrid = merge(refgrid, NamedTuple{(var,)}(([mean(val)],)))
               else
                   refgrid = merge(refgrid, NamedTuple{(var,)}(([first(val)],)))
               end
           end
       end
       return refgrid
   end
   ```

**Validation Requirements**: 
- BenchmarkTools.jl tests verify 0 bytes in hot paths
- Bootstrap validation of all standard errors
- Cross-validation AD vs FD backends

### **Phase 2: Clean 2×2 API Implementation**
*Target: Simple, Julian API that leverages the zero-allocation engine*

1. **New `src/population.jl`** with compilation caching:
   ```julia
   # Global cache for compiled formulas (MARGINS_GUIDE.md pattern)
   const COMPILED_CACHE = Dict{UInt64, Any}()
   
   function population_margins(model, data; type::Symbol=:effects, vars=:continuous, target::Symbol=:mu, backend::Symbol=:ad, kwargs...)
       # Single data conversion (consistent format throughout)
       data_nt = Tables.columntable(data)
       
       # Build zero-allocation engine with caching
       engine = _get_or_build_engine(model, data_nt, vars)
       
       if type === :effects
           df, G = _ame_continuous(engine, data_nt; target, backend, kwargs...)  # → AME
           return MarginsResult(df, G, _build_metadata(...))
       else # :predictions  
           df, G = _population_predictions(engine, data_nt; target, kwargs...)  # → AAP
           return MarginsResult(df, G, _build_metadata(...))
       end
   end
   
   # Compilation caching (MARGINS_GUIDE.md pattern)
   function _get_or_build_engine(model, data_nt::NamedTuple, vars)
       cache_key = hash(model, keys(data_nt), vars)  # Include vars in cache key
       if haskey(COMPILED_CACHE, cache_key)
           return COMPILED_CACHE[cache_key]
       else
           engine = build_engine(model, data_nt, vars)
           COMPILED_CACHE[cache_key] = engine
           return engine
       end
   end
   ```

2. **New `src/profile.jl`**:
   ```julia
   function profile_margins(model, data; at, type::Symbol=:effects, vars=:continuous, target::Symbol=:mu, backend::Symbol=:fd, kwargs...)
       data_nt = Tables.columntable(data)
       engine = build_engine(model, data_nt, vars)
       profiles = _build_profiles(at, data_nt)  # Clean, single implementation
       
       if type === :effects
           df, G = _mem_continuous(engine, profiles; target, backend, kwargs...)  # → MEM
           return MarginsResult(df, G, _build_metadata(...))
       else # :predictions
           df, G = _profile_predictions(engine, profiles; target, kwargs...)  # → APM
           return MarginsResult(df, G, _build_metadata(...))
       end
   end
   ```

3. **Unified Profile Building**:
   ```julia
   # Single, efficient profile builder (replaces both old systems)
   function _build_profiles(at, data_nt::NamedTuple)
       if at === :means
           return [_means_profile(data_nt)]
       elseif at isa Dict
           return _cartesian_profiles(at, data_nt) 
       elseif at isa Vector
           return at  # Explicit profiles
       else
           error("Invalid profile specification: $at")
       end
   end
   ```

### **Phase 3: Radical File Structure Simplification**
*Target: Maintainable 4-file architecture*

1. **New Structure**:
   ```
   src/
   ├── Margins.jl       # Module definition, exports, types
   ├── engine.jl        # Zero-allocation FormulaCompiler integration  
   ├── population.jl    # population_margins() implementation
   └── profile.jl       # profile_margins() implementation
   ```

2. **Eliminate Files**:
   - **REMOVE**: All of `core/` directory (7 files)
   - **REMOVE**: All of `computation/` directory (4 files) 
   - **REMOVE**: All of `features/` directory (2 files)
   - **REMOVE**: All of `api/` directory (3 files)
   - **REMOVE**: `gradient_utils.jl`

3. **Merge Logic Into Engine**:
   - All FormulaCompiler integration → `engine.jl`
   - All utility functions needed → `engine.jl`  
   - Results type and display → `Margins.jl`

### **Phase 4: Aggressive API Cleanup with Dual Interface Pattern**
*Target: Simple, discoverable API following FormulaCompiler conventions*

1. **Minimal Exports with Dual Interface**:
   ```julia
   # Core functionality (MARGINS_GUIDE.md dual interface pattern)
   export population_margins, profile_margins, MarginsResult
   
   # In-place versions (zero allocation) 
   export population_margins!, profile_margins!
   
   # Remove all complex exports:
   # - refgrid_* functions (use at= parameter instead)
   # - mix, CategoricalMixture (too complex) 
   # - get_gradients, contrast, bootstrap_effects (out of scope)
   ```

2. **Unified Parameter System**:
   ```julia
   # Both functions use same parameter names and meanings
   population_margins(model, data; type, vars, target, backend, over, vcov, ...)
   profile_margins(model, data; at, type, vars, target, backend, over, vcov, ...)
   # Only difference: profile_margins has `at` parameter
   ```

### **Phase 5: Performance Optimization and Validation**
*Target: Achieve FormulaCompiler's performance potential*

1. **Performance Targets**:
   - Population margins: <100ns per row (vs current ~10μs)
   - Profile evaluation: <1μs per profile (vs current ~100μs)
   - Memory usage: <1KB per analysis (vs current ~100KB+)

2. **Continuous Validation**:
   - BenchmarkTools.jl tests for allocations on all hot paths
   - Statistical correctness tests (bootstrap validation)  
   - Performance benchmarks in CI
   - Cross-validation between backends

3. **FormulaCompiler Best Practices from MARGINS_GUIDE.md**:
   - Use reference grids for clean evaluation points (preferred for profiles)
   - Use scenario overrides for counterfactual analysis 
   - Leverage compiled evaluator caching properly
   - Follow zero-allocation patterns throughout
   - Validate input variables early at API boundary
   - Implement graceful backend fallbacks (AD → FD)
   - Pre-allocate output matrices for batch operations
   - Use consistent `Tables.columntable()` format throughout
   - Provide dual interface: allocating and in-place versions

## 🎯 **Specific Implementation Strategy**

### **Step 1: Create New Simplified Structure**

1. **New `engine.jl`** - All FormulaCompiler integration:
   ```julia
   # Zero-allocation engine with pre-allocated buffers
   struct MarginsEngine
       # FormulaCompiler components
       compiled::CompiledFormula
       de::Union{DerivativeEvaluator, Nothing}
       
       # Pre-allocated buffers (zero runtime allocation)
       η_buf::Vector{Float64}
       g_buf::Vector{Float64}
       gβ_buf::Vector{Float64}
       
       # Model parameters
       β::Vector{Float64}
       Σ::Matrix{Float64}
       link::GLM.Link
   end
   ```

2. **New `population.jl`** - Clean AME/APE implementation:
   ```julia
   function population_margins(model, data; type=:effects, kwargs...)
       data_nt = Tables.columntable(data)
       engine = build_margins_engine(model, data_nt)
       
       if type === :effects
           return _population_effects_zero_alloc(engine, data_nt; kwargs...)
       else
           return _population_predictions_zero_alloc(engine, data_nt; kwargs...)
       end
   end
   ```

3. **New `profile.jl`** - Clean MER/MEM implementation:
   ```julia
   function profile_margins(model, data; at, type=:effects, kwargs...)
       data_nt = Tables.columntable(data)
       engine = build_margins_engine(model, data_nt)
       profiles = _build_profiles_simple(at, data_nt)
       
       if type === :effects
           return _profile_effects_zero_alloc(engine, profiles; kwargs...)
       else
           return _profile_predictions_zero_alloc(engine, profiles; kwargs...)
       end
   end
   ```

### **Step 2: Migration Strategy**

1. **Implement new simplified versions alongside current code**
2. **Test performance and correctness against current implementation**
3. **Replace current implementation once validated**
4. **Remove old files and clean up exports**

### **Step 3: Quality Gates**

1. **Performance Requirements**:
   - Zero allocations in hot paths (verified with BenchmarkTools.jl tests)
   - <1μs per profile evaluation 
   - <100ns per population row

2. **Correctness Requirements**:
   - All standard errors match current implementation within 1e-10 tolerance
   - Bootstrap validation for all delta-method SEs
   - Cross-validation between AD and FD backends

3. **Simplicity Requirements**:
   - <5 source files total
   - <1000 lines of code total
   - Single-page API documentation

## 📊 **Expected Impact**

### **Performance Gains**:
- **100x faster** population margins (from allocation elimination)
- **50x faster** profile margins (from FormulaCompiler integration)
- **1000x less memory** usage (from buffer pooling)

### **Maintainability Gains**:
- **70% fewer** source files (17 → 5)
- **80% fewer** lines of code (~3000 → ~600)
- **90% simpler** API surface (remove complex iterators and utilities)

### **User Experience Gains**:
- **Single concept**: Population vs Profile (no confusing acronyms)
- **Predictable performance**: Always fast, never surprising allocations
- **Simple debugging**: Easy to understand what code is running

## ⚠️ **Risks and Mitigation**

### **Risk**: Breaking changes for existing users
**Mitigation**: Keep current API as deprecated wrapper, provide migration guide

### **Risk**: Statistical correctness during rewrite  
**Mitigation**: Extensive cross-validation against current implementation, bootstrap validation

### **Risk**: Performance regressions
**Mitigation**: Continuous benchmarking, allocation tracking, performance tests

## 📅 **Aggressive Implementation Timeline**

*Breaking changes acceptable - focus on correctness, performance, and proper FormulaCompiler integration*

### **Phase 1 (Week 1-2)**: Zero-Allocation Engine Foundation
- **Day 1-3**: Design and implement `MarginsEngine` struct with pre-allocated buffers
- **Day 4-6**: Build zero-allocation population effects functions using FormulaCompiler properly
- **Day 7-10**: Implement zero-allocation profile effects with scenario override system
- **Validation**: BenchmarkTools.jl tests verify 0 bytes, bootstrap validation of SEs

### **Phase 2 (Week 3)**: Clean 2×2 API Implementation  
- **Day 1-2**: Implement new `population.jl` with clean Julia-style API
- **Day 3-4**: Implement new `profile.jl` with unified profile building
- **Day 5**: Single, efficient profile specification system (no duplicates)
- **Validation**: All current functionality works, performance benchmarks show improvement

### **Phase 3 (Week 4)**: Radical Simplification
- **Day 1-2**: Delete old file structure, move logic to 4-file architecture
- **Day 3-4**: Aggressive API cleanup - remove complex exports and features
- **Day 5**: Update `Margins.jl` module with minimal, clean exports
- **Validation**: Test suite passes, package loads cleanly

### **Phase 4 (Week 5)**: Performance Optimization & Validation
- **Day 1-3**: Achieve target performance (<100ns per row, <1μs per profile)
- **Day 4-5**: Comprehensive statistical validation and benchmarking
- **Validation**: Performance targets met, statistical correctness verified

### **Phase 5 (Week 6)**: Documentation & Polish
- **Day 1-3**: Update documentation to reflect simplified API
- **Day 4-5**: Final testing, edge cases, error handling
- **Validation**: Ready for production use

## 🎯 **How to Start**

**Immediate next step**: Begin with Phase 1 - implement the `MarginsEngine` struct and zero-allocation foundation. This is a clean-slate approach that will:

1. **Deliver massive performance gains** (100x+ speedup expected)
2. **Simplify the architecture** (17 files → 4 files)  
3. **Use FormulaCompiler properly** (scenario overrides, zero-allocation paths)
4. **Maintain statistical correctness** (bootstrap validation throughout)

The approach is designed to be:
1. **Aggressive** - no concern for breaking changes
2. **Performance-first** - zero-allocation from the start
3. **Correct** - extensive statistical validation
4. **Julian** - clean, idiomatic Julia code
5. **FormulaCompiler-native** - leverage all FC capabilities

This reorganization will transform Margins.jl from an over-engineered, slow package into a simple, blazing-fast statistical interface that properly leverages FormulaCompiler.jl's capabilities.