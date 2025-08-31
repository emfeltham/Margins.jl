# ARCHITECTURE.md: Margins.jl System Design

## Overview

Margins.jl implements a clean **2×2 framework** for marginal effects computation, built on FormulaCompiler.jl's zero-allocation foundation. The architecture prioritizes statistical correctness, performance, Julian style, and proper FormulaCompiler integration.

## 🏗️ **System Architecture**

### **The 2×2 Framework**

```
                   │  Effects      │  Predictions
                   │  (∂y/∂x)      │  (fitted values)
───────────────────┼───────────────┼─────────────────
Population         │  AME          │  AAP
(across sample)    │  (avg marg    │  (avg adjusted
                   │   effect)     │   prediction)
───────────────────┼───────────────┼─────────────────  
Profile            │  MEM          │  APM
(at specific pts)  │  (marg effect │  (adjusted pred
                   │   at mean)    │   at mean)
```

**Acronym Definitions:**
- **AME**: Average Marginal Effect (derivative for continuous, contrast for categorical/bool)
- **AAP**: Average Adjusted Prediction (population average of fitted values)
- **MEM**: Marginal Effect at the Mean (effect at representative point)  
- **APM**: Adjusted Prediction at the Mean (fitted value at representative point)

**User Interface:**
- `population_margins(model, data; type=:effects)` → **AME**
- `population_margins(model, data; type=:predictions)` → **AAP**
- `profile_margins(model, data; at=:means, type=:effects)` → **MEM**  
- `profile_margins(model, data; at=:means, type=:predictions)` → **APM**

### **File Organization (4 Files Only)**

```
src/
├── Margins.jl        # Module definition, exports, MarginsResult type
├── engine.jl         # FormulaCompiler integration, MarginsEngine struct
├── population.jl     # population_margins() implementation (AME/APE)
└── profile.jl        # profile_margins() implementation (MER/MEM)
```

**Design Principle**: Radical simplification from 17+ files to 4 essential files.

## 🔧 **Core Components**

### **MarginsEngine (engine.jl)**

```julia
struct MarginsEngine{L<:GLM.Link}
    # FormulaCompiler components (pre-compiled)
    compiled::FormulaCompiler.CompiledFormula
    de::Union{FormulaCompiler.DerivativeEvaluator, Nothing}
    
    # Pre-allocated buffers (zero runtime allocation)
    g_buf::Vector{Float64}              # Marginal effects results
    gβ_accumulator::Vector{Float64}     # AME gradient accumulation
    
    # Model parameters
    β::Vector{Float64}
    Σ::Matrix{Float64}
    link::L
    vars::Vector{Symbol}
    data_nt::NamedTuple  # Reference for scenarios/refgrids
end
```

**Key Features:**
- **Single compilation** per model+data combination
- **Pre-allocated buffers** - no allocation in hot paths
- **Caching support** - compiled engines cached by data signature
- **Input validation** - continuous vs categorical variable checking

### **Population Effects (population.jl)**

**Core Function:** `_ame_continuous(engine, data_nt; target, backend)` → **AME**

**Strategy:**
- Use **FormulaCompiler's built-in AME gradient accumulation**
- **Zero allocation** with `:fd` backend for large datasets
- **Graceful fallbacks** from `:ad` to `:fd` on failure

**Key APIs Used:**
```julia
FormulaCompiler.accumulate_ame_gradient!(gβ_sum, de, β, rows, var; link, backend)
FormulaCompiler.marginal_effects_eta!(g_buf, de, β, row; backend)
FormulaCompiler.delta_method_se(gradient, Σ)
```

### **Profile Effects (profile.jl)**

**Core Function:** `_mem_continuous(engine, profiles; target, backend)` → **MEM**

**Strategy:**
- Use **reference grids** (FormulaCompiler MARGINS_GUIDE.md recommendation)
- Build **minimal synthetic data** for clean evaluation points  
- **Per-profile compilation** for maximum accuracy

**Key Pattern:**
```julia
# Build minimal reference grid (not scenario overrides)
refgrid_data = _build_refgrid_data(profile, original_data)
refgrid_compiled = FormulaCompiler.compile_formula(model, refgrid_data)

# Evaluate at synthetic reference point
FormulaCompiler.marginal_effects_eta!(g_buf, refgrid_de, β, 1; backend)
```

## 📊 **Data Flow Architecture**

### **Population Margins Flow (AME/AAP)**
```
population_margins(model, data; type=:effects)  # AME
    ↓
Tables.columntable(data)         # Single conversion
    ↓  
build_engine(model, data_nt)     # Compilation + caching
    ↓
FormulaCompiler.accumulate_ame_gradient!()  # Zero-alloc AME
    ↓
FormulaCompiler.delta_method_se()           # Standard errors
    ↓
MarginsResult(df, G, metadata)   # Return structured result
```

### **Profile Margins Flow (MEM/APM)**
```
profile_margins(model, data; at=:means, type=:effects)  # MEM
    ↓
Tables.columntable(data)         # Single conversion  
    ↓
build_engine(model, data_nt)     # Compilation + caching
    ↓
_build_profiles(at, data_nt)     # Profile specification
    ↓
For each profile:
    _build_refgrid_data()        # Minimal synthetic data
    ↓
    compile_formula(refgrid)     # Per-profile compilation
    ↓
    marginal_effects_eta!()      # Zero-alloc evaluation
    ↓
MarginsResult(df, G, metadata)   # Return structured result
```

### **Buffer Management**
```
MarginsEngine.g_buf              # Reused for all marginal effects
MarginsEngine.gβ_accumulator     # Reused for all gradients  
DerivativeEvaluator.*_buffer     # FormulaCompiler's internal buffers
```

**Zero Allocation Strategy:** All buffers pre-allocated once, reused throughout computation.

## 🎯 **FormulaCompiler Integration Strategy**

### **When to Use Reference Grids vs Scenario Overrides**

**Reference Grids (Preferred for MER/MEM):**
```julia
# Build minimal synthetic data for clean evaluation
refgrid_data = (
    x = [profile_x],              # Specific value  
    group = [profile_group],      # Specific level
    z = [mean_z]                  # Representative value
)
```

**Use For:**
- Marginal effects at representative values (MER/MEM)
- Profile-based analysis  
- Clean synthetic evaluation points
- Memory efficiency (minimal data)

**Scenario Overrides (For Counterfactuals):**
```julia  
# Override variables while preserving data structure
scenario = create_scenario("policy", data; treatment=true, income=median)
```

**Use For:**
- "What if everyone had X" questions
- Policy counterfactuals preserving correlations
- Sensitivity analysis on real data

### **Backend Selection Strategy**

**Population (Large N):**
- **Default: `:fd`** - Zero allocation, good for AME across many rows
- **Fallback: None** - FD is the primary path

**Profile (Small N):**  
- **Default: `:ad`** - ~368 bytes/call, faster and more accurate
- **Fallback: `:fd`** - On AD failure, automatic fallback with warning

### **Key FormulaCompiler APIs Used**

**Compilation:**
```julia
FormulaCompiler.compile_formula(model, data_nt)
FormulaCompiler.build_derivative_evaluator(compiled, data_nt; vars)
FormulaCompiler.continuous_variables(compiled, data_nt)
```

**Zero-Allocation Evaluation:**
```julia
FormulaCompiler.marginal_effects_eta!(g_buf, de, β, row; backend)
FormulaCompiler.marginal_effects_mu!(g_buf, de, β, row; link, backend)
FormulaCompiler.accumulate_ame_gradient!(gβ_sum, de, β, rows, var; link, backend)
```

**Gradient Computation:**
```julia
FormulaCompiler.me_eta_grad_beta!(gβ_temp, de, β, row, var)
FormulaCompiler.me_mu_grad_beta!(gβ_temp, de, β, row, var; link)
FormulaCompiler.delta_method_se(gradient, Σ)
```

## 🚀 **Performance Architecture**

### **Zero-Allocation Targets**

**Population AME:** 
- **Target:** <100ns per row (vs current ~10μs)
- **Strategy:** FormulaCompiler's `accumulate_ame_gradient!()` with `:fd` backend
- **Allocation:** 0 bytes after warmup

**Profile Effects:**
- **Target:** <1μs per profile (vs current ~100μs)  
- **Strategy:** Reference grid compilation with `:ad` backend
- **Allocation:** ~368 bytes per profile (ForwardDiff overhead)

### **Memory Management**

**Engine Construction (Once):**
```julia
g_buf = Vector{Float64}(undef, n_vars)           # ~100 bytes
gβ_accumulator = Vector{Float64}(undef, n_coef)  # ~1000 bytes
# FormulaCompiler handles internal buffers
```

**Hot Paths (Reuse Buffers):**
```julia  
# Population: 0 bytes (pure buffer reuse)
accumulate_ame_gradient!(engine.gβ_accumulator, ...)  

# Profile: ~368 bytes (ForwardDiff)
marginal_effects_eta!(engine.g_buf, ...; backend=:ad)
```

### **Caching Strategy**

**Compilation Cache:**
```julia
const COMPILED_CACHE = Dict{UInt64, MarginsEngine}()

function _get_or_build_engine(model, data_nt, vars)
    cache_key = hash(model, keys(data_nt), vars)
    get!(COMPILED_CACHE, cache_key) do
        build_engine(model, data_nt, vars)
    end
end
```

**Cache Invalidation:** Automatic via hash-based keys including model and data signature.

## 🧪 **Testing & Validation Architecture**

### **Performance Tests** 
```julia
@testset "Zero Allocation Guarantees" begin
    # Population should be 0 bytes after warmup
    @test @allocated(population_margins(model, data)) == 0
    
    # Profile should be bounded (ForwardDiff overhead)
    @test @allocated(profile_margins(model, data; at=:means)) < 1000
end
```

### **Statistical Correctness Tests**
```julia
@testset "Bootstrap Validation" begin
    # All standard errors validated against bootstrap
    margins_se = population_margins(model, data).se
    bootstrap_se = bootstrap_standard_errors(model, data, 1000)
    @test margins_se ≈ bootstrap_se rtol=0.05
end
```

### **Cross-Validation Tests**
```julia
@testset "Backend Consistency" begin  
    # AD and FD should give same results
    result_ad = profile_margins(model, data; backend=:ad)
    result_fd = profile_margins(model, data; backend=:fd)
    @test result_ad.estimate ≈ result_fd.estimate rtol=1e-10
end
```

## 📋 **API Design Philosophy**

### **Dual Interface Pattern (FormulaCompiler Style)**

**Allocating Versions (Convenience):**
```julia
result = population_margins(model, data; type=:effects)
result = profile_margins(model, data; at=:means, type=:effects)
```

**In-Place Versions (Performance):**
```julia
population_margins!(df, G, engine, data_nt; type=:effects)  
profile_margins!(df, G, engine, profiles; type=:effects)
```

### **Parameter Consistency**
```julia
# Both functions share parameter semantics
population_margins(model, data; type, vars, target, backend, vcov, ...)
profile_margins(model, data; at, type, vars, target, backend, vcov, ...)
#                              ↑ only difference
```

### **Error Handling Strategy**

**Input Validation (Early):**
```julia
function _validate_variables(data_nt, vars)
    for var in vars
        haskey(data_nt, var) || error("Variable $var not found")
        col = getproperty(data_nt, var)
        if col isa CategoricalArray && vars !== :continuous
            @warn "Variable $var is categorical. Use contrasts for categorical effects."
        end
    end
end
```

**Graceful Fallbacks:**
```julia
try
    # Try AD first for accuracy
    accumulate_ame_gradient!(buffer, de, β, rows, var; backend=:ad)
catch e
    @warn "AD failed for $var, falling back to FD: $e"
    accumulate_ame_gradient!(buffer, de, β, rows, var; backend=:fd)
end
```

## 🎯 **Design Trade-offs & Decisions**

### **Why Reference Grids Over Scenario Overrides?**
- **Memory efficiency**: Minimal synthetic data vs full data copying
- **Clarity**: Clean evaluation points vs complex data correlations
- **Performance**: Single compilation per profile vs scenario management
- **FormulaCompiler recommendation**: MARGINS_GUIDE.md prefers reference grids for MER/MEM

### **Why 4 Files Instead of 17?**
- **Cognitive load**: Easier to understand and maintain
- **Dependencies**: Clear separation of concerns without over-abstraction
- **Performance**: Less indirection, easier optimization
- **Julian style**: Simple, direct code organization

### **Why Aggressive Caching?**
- **Compilation cost**: FormulaCompiler compilation is expensive (milliseconds)  
- **Repeated calls**: Users often call margins on same model+data combinations
- **Memory trade-off**: Small memory cost for major performance gain

### **Why Dual Interface (allocating + in-place)?**
- **FormulaCompiler pattern**: Follows established convention
- **User choice**: Convenience vs performance as needed
- **Composability**: In-place versions enable higher-level optimizations

This architecture delivers on all four priorities: **statistical correctness** (bootstrap validation), **performance** (zero-allocation paths), **Julian style** (clean, simple design), and **proper FormulaCompiler integration** (leverages all FC capabilities correctly).