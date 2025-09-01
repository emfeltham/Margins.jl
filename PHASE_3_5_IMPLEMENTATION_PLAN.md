# PHASE_3_5_IMPLEMENTATION_PLAN.md: FormulaCompiler Integration Fixes

## 🎯 **DETAILED IMPLEMENTATION PLAN**

Based on analysis of `/Users/emf/.julia/dev/FormulaCompiler/MARGINS_GUIDE.md` and the actual FormulaCompiler source code, this document provides detailed specifications for fixing the critical FormulaCompiler integration violations identified in `PHASE_3_5.md`.

- [x] ## **📋 PRIORITY 2: Proper Backend Selection** ✅ **COMPLETED**

### **Current Implementation Issues**
- Using `:ad` backend everywhere (suboptimal for population margins)
- No graceful fallbacks
- Missing performance-optimal backend selection

### **✅ CORRECT BACKEND STRATEGY**

**Population Margins**: Use `:fd` backend for zero allocations across many rows
```julia
function _ame_continuous_and_categorical(engine::MarginsEngine, data_nt::NamedTuple; target=:mu, backend=:fd, measure=:effect)
    # Default to :fd for population - zero allocations across many rows
    recommended_backend = backend === :auto ? :fd : backend
    
    # Use FormulaCompiler's built-in AME gradient accumulation
    for var in continuous_vars
        try
            # Zero-allocation AME gradient accumulation (REORG.md lines 51-52)
            FormulaCompiler.accumulate_ame_gradient!(
                engine.gβ_accumulator, engine.de, engine.β, rows, var;
                link=(target === :mu ? engine.link : GLM.IdentityLink()),
                backend=recommended_backend  # :fd for zero allocations
            )
        catch e
            if recommended_backend === :ad
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
        
        # Compute AME value using zero-allocation marginal effects
        ame_val = 0.0
        for row in rows
            if target === :mu
                FormulaCompiler.marginal_effects_mu!(engine.g_buf, engine.de, engine.β, row;
                                                   link=engine.link, backend=recommended_backend)
            else
                FormulaCompiler.marginal_effects_eta!(engine.g_buf, engine.de, engine.β, row;
                                                    backend=recommended_backend)
            end
            ame_val += engine.g_buf[var_idx]
        end
        ame_val /= length(rows)
        
        # Average gradient and compute SE
        gβ_avg = engine.gβ_accumulator ./ length(rows)
        se = FormulaCompiler.delta_method_se(gβ_avg, engine.Σ)
        
        push!(results, (term=string(var), estimate=ame_val, se=se))
        G[cont_idx, :] = gβ_avg
        cont_idx += 1
    end
end
```

**Profile Margins**: Use `:ad` backend for speed/accuracy at specific points
```julia
function profile_margins(model, data; at=:means, type=:effects, backend=:auto, kwargs...)
    # Default to :ad for profiles - faster for single evaluations
    recommended_backend = backend === :auto ? :ad : backend
    
    # Use reference grid system with AD backend (correct approach)
    # Build reference grid and evaluate at specific representative points
    # ... (current reference grid implementation)
end
```

### **API Changes**:
```julia
# Update function signatures to support backend selection
population_margins(model, data; backend=:auto, kwargs...)  # Defaults to :fd
profile_margins(model, data; backend=:auto, kwargs...)     # Defaults to :ad
```

- [x] ## **📋 PRIORITY 4: Unified Caching Strategy** ✅ **COMPLETED**

### **Current Fragmented Implementation**
- `COMPILED_CACHE` in `population/core.jl`
- `TYPICAL_VALUES_CACHE` in `profile/refgrids.jl`
- Different cache keys and strategies

### **✅ UNIFIED ENGINE CACHING**

**Single cache system following FormulaCompiler patterns:**

```julia
# engine/caching.jl - Unified caching system
"""
Global cache for MarginsEngine instances to avoid recompilation.
Uses FormulaCompiler's recommended caching patterns.
"""
const ENGINE_CACHE = Dict{UInt64, MarginsEngine}()

"""
    get_or_build_engine(model, data_nt, vars) -> MarginsEngine

Get cached engine or build new one. Implements FormulaCompiler caching best practices.
"""
function get_or_build_engine(model, data_nt::NamedTuple, vars::Vector{Symbol})
    # Create comprehensive cache key including all relevant factors
    cache_key = hash((
        model,                    # Model object
        keys(data_nt),           # Data structure
        vars,                    # Variables for derivatives
        typeof(model),           # Model type for dispatch
        fieldnames(typeof(model)) # Model structure
    ))
    
    return get!(ENGINE_CACHE, cache_key) do
        build_engine(model, data_nt, vars)
    end
end

"""
    clear_engine_cache!()

Clear the engine cache. Useful for memory management in long-running sessions.
"""
function clear_engine_cache!()
    empty!(ENGINE_CACHE)
    return nothing
end

"""
    get_cache_stats()

Return cache statistics for monitoring and debugging.
"""
function get_cache_stats()
    return (
        entries = length(ENGINE_CACHE),
        memory_estimate = sum(sizeof(engine) for engine in values(ENGINE_CACHE)),
        keys = collect(keys(ENGINE_CACHE))
    )
end
```

**Update API functions to use unified caching:**

```julia
# population/core.jl - Remove old COMPILED_CACHE, use unified system
function population_margins(model, data; type=:effects, vars=nothing, kwargs...)
    data_nt = Tables.columntable(data)
    
    # Handle vars parameter
    if type === :effects
        vars = vars === nothing ? :all_continuous : vars
        if vars === :all_continuous
            temp_compiled = FormulaCompiler.compile_formula(model, data_nt)
            vars = FormulaCompiler.continuous_variables(temp_compiled, data_nt)
        end
    else
        vars = Symbol[]  # No derivatives needed for predictions
    end
    
    # Use unified caching system
    engine = get_or_build_engine(model, data_nt, vars)
    
    # ... rest of implementation
end

# profile/core.jl - Remove separate cache, use unified system  
function profile_margins(model, data; at=:means, kwargs...)
    data_nt = Tables.columntable(data)
    
    # Use unified caching (same key system as population)
    engine = get_or_build_engine(model, data_nt, vars)
    
    # ... rest of implementation using scenarios
end
```

- [ ] ## **📋 PRIORITY 5: Zero-Allocation Path Verification** ⚠️ **PARTIALLY VALID**

### **⚠️ MIXED VALIDITY - Some Claims Correct, Others Wrong**

**✅ Valid Claims:**
- Zero-allocation goals are worthwhile
- Buffer pre-allocation is good practice  
- Current MarginsEngine already has some buffers (`g_buf`, `gβ_accumulator`)

**❌ Incorrect Claims:**
- Claims MarginsEngine needs `η_buf` field that doesn't exist in current implementation
- References "scenarios" where it should say "reference grids"
- Some allocation tests may be based on incorrect performance assumptions

### **Current Allocation Issues**
Multiple allocation hotspots identified in MARGINS_GUIDE.md analysis.

### **✅ ALLOCATION AUDIT AND FIXES**

**1. Fix Buffer Management**
```julia
# engine/core.jl - Pre-allocate all buffers in engine construction
struct MarginsEngine{L<:GLM.Link}
    # FormulaCompiler components (compiled once)
    compiled::FormulaCompiler.UnifiedCompiled
    de::Union{FormulaCompiler.DerivativeEvaluator, Nothing}
    
    # Zero-allocation buffers (allocated once, reused forever)
    g_buf::Vector{Float64}              # Marginal effects results
    gβ_accumulator::Vector{Float64}     # Parameter gradients
    η_buf::Vector{Float64}              # Linear predictor buffer
    
    # Model parameters (immutable)
    model::Any
    β::Vector{Float64}
    Σ::Matrix{Float64}
    link::L
    vars::Vector{Symbol}
    data_nt::NamedTuple  # Keep reference for scenarios
end
```

**2. Eliminate In-Loop Allocations**
```julia
# Remove all Vector{Float64}(undef, ...) calls inside loops
# BEFORE (BAD):
for row in rows
    xbuf = Vector{Float64}(undef, length(compiled))  # ❌ Allocation per row
    # ...
end

# AFTER (GOOD): 
# Pre-allocate buffer once in engine construction
for row in rows
    # Reuse engine.η_buf  ✅ Zero allocations
    # ...
end
```

**3. Add Allocation Tests**
```julia
# test/test_zero_allocations.jl - Verify zero-allocation targets
using BenchmarkTools

@testset "Zero Allocation Verification" begin
    # Test data
    data = DataFrame(x = randn(1000), y = randn(1000), group = rand(["A", "B"], 1000))
    model = lm(@formula(y ~ x + group), data)
    
    # Test population margins - should be 0 allocations after warmup
    result1 = population_margins(model, data; type=:effects, backend=:fd)  # Warmup
    bench1 = @benchmark population_margins($model, $data; type=:effects, backend=:fd)
    @test bench1.memory == 0  # Zero allocations with FD backend
    
    # Test profile margins with AD backend - should be minimal allocations
    result2 = profile_margins(model, data; at=:means, type=:effects, backend=:ad)  # Warmup
    bench2 = @benchmark profile_margins($model, $data; at=:means, type=:effects, backend=:ad)
    @test bench2.memory < 1000  # Allow small AD allocations, but not recompilation
    
    # Test profile margins with FD backend - should be 0 allocations
    result3 = profile_margins(model, data; at=:means, type=:effects, backend=:fd)  # Warmup
    bench3 = @benchmark profile_margins($model, $data; at=:means, type=:effects, backend=:fd)
    @test bench3.memory == 0  # Zero allocations with FD backend
end
```

---

## **🔧 IMPLEMENTATION TIMELINE - UPDATED BASED ON ANALYSIS**

### **❌ ORIGINAL TIMELINE BASED ON INCORRECT ASSUMPTIONS**
The original timeline was based on implementing scenario overrides for profile analysis, which was **architecturally wrong**. Most priorities have been cancelled or corrected.

### **✅ FINAL STATUS AFTER COMPLETE AUDIT**
- **Priority 1**: ❌ **INCORRECTLY IMPLEMENTED** - Reverted to correct reference grid approach
- **Priority 2**: ✅ **COMPLETED** - Backend auto-selection now implemented for both population and profile margins
- **Priority 3**: ❌ **CANCELLED** - Based on architectural misunderstanding
- **Priority 4**: ✅ **COMPLETED** - Unified caching system implemented in engine/caching.jl
- **Priority 5**: ⚠️ **PARTIALLY VALID** - Zero-allocation goals valid, some implementation details wrong

---

## **📊 SUCCESS METRICS - CORRECTED**

### **❌ ORIGINAL METRICS WERE BOGUS**
Original metrics based on "100x+ speedup from eliminating recompilation" were based on incorrect problem analysis.

### **✅ REALISTIC PERFORMANCE EXPECTATIONS**
- **Population margins**: Current implementation already efficient with proper backend selection
- **Profile margins**: Current reference grid approach is architecturally correct
- **No major recompilation problems**: Analysis revealed current approach is already optimal

### **Correctness Validation**
- All standard errors within 1e-10 of current implementation
- Cross-backend validation: `:ad` and `:fd` results agree within tolerance
- Bootstrap validation of delta-method SEs

### **API Compatibility**
- No breaking changes to public API
- Backward compatible parameter handling
- Graceful fallbacks for all error conditions

This implementation plan leverages FormulaCompiler.jl's actual capabilities to achieve the aggressive performance targets while maintaining statistical correctness.