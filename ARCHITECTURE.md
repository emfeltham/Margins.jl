# ARCHITECTURE.md: Margins.jl System Design

Document Hierarchy:

1. ARCHITECTURE.md - High-level system design and 2×2 framework
2. REORG.md - Problem analysis, design decisions, and reorganization
strategy
3. FILE_PLAN.md - Detailed implementation blueprint with exact
specifications

This creates a clear documentation flow from conceptual design → strategic
planning → detailed implementation, with proper cross-references so users
can navigate between the documents easily!

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
- **AME**: Average Marginal Effect (derivative for continuous, baseline contrast for categorical/bool)
- **AAP**: Average Adjusted Prediction (population average of fitted values)
- **MEM**: Marginal Effect at the Mean (derivative for continuous, row-specific baseline contrast for categorical)  
- **APM**: Adjusted Prediction at the Mean (fitted value at representative point)

**User Interface:**
- `population_margins(model, data; type=:effects, vars=[:x1, :x2])` → **AME**
- `population_margins(model, data; type=:effects, vars=[:x1], over=[:region])` → **AME stratified by region**
- `population_margins(model, data; type=:effects, vars=[:x1], over=Dict(:x2 => [1, 2, 3]))` → **AME at specific x2 values**
- `population_margins(model, data; type=:predictions)` → **AAP** (vars not needed)
- `profile_margins(model, reference_grid; type=:effects, vars=[:x1, :x2])` → **MEM** (derivatives/row-specific contrasts at each reference point)
- `profile_margins(model, reference_grid, data; type=:effects, vars=[:x1])` → **MEM** (smart: fills missing vars, row-specific contrasts for categorical)
- `profile_margins(model, reference_grid; type=:predictions)` → **APM** (fitted values at each reference point)

**Future Extensions:**
- `profile_contrasts(model, reference_grid; vars=[:x1])` → **Pairwise contrasts between reference points**
- `profile_contrasts(model, reference_grid; vars=[:catvar], contrast=:pairwise)` → **All pairwise categorical contrasts**

### **File Organization (Organized Subdirectories)**

```
src/
├── Margins.jl              # Module definition, exports
├── types.jl                 # MarginsResult, error types  
├── engine/
│   ├── core.jl             # MarginsEngine, construction
│   ├── utilities.jl        # Shared utilities, validation
│   └── caching.jl          # Compilation caching
├── population/
│   ├── core.jl             # Main population_margins()
│   ├── contexts.jl         # at/over parameter handling
│   └── effects.jl          # AME computation
└── profile/
    ├── core.jl             # Main profile_margins()
    ├── refgrids.jl         # Reference grid builders
    └── contrasts.jl        # Row-specific baseline contrasts
```

**Design Principle**: Organized simplification from 17+ files to 12 well-structured files with logical grouping.

**📋 Detailed Implementation**: See **[FILE_PLAN.md](FILE_PLAN.md)** for complete function specifications, implementation timeline, and success metrics.

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

**Core Functions:** 
- `_ame_continuous_and_categorical(engine, data_nt; target, backend)` → **AME for both variable types**
  - Continuous: derivatives (∂y/∂x)
  - Categorical: traditional baseline contrasts (ŷ|level - ŷ|baseline)
- `_mem_continuous_and_categorical(engine, profiles; target, backend)` → **MEM for both variable types**
  - Continuous: derivatives at each profile
  - Categorical: novel row-specific baseline contrasts

**Strategy:**
- **Continuous vars**: FormulaCompiler's built-in AME gradient accumulation
- **Categorical vars (Population)**: Traditional baseline contrasts (ŷ|level - ŷ|baseline across sample)
- **Categorical vars (Profile)**: Novel row-specific baseline contrasts (respects reference grid exactly)
- **Zero allocation** with `:fd` backend for large datasets
- **Auto-detection** of variable types using `FormulaCompiler.continuous_variables()`
- **Reference grid philosophy**: Each row specifies complete evaluation context
- **Graceful fallbacks** from `:ad` to `:fd` on failure

**Key APIs Used:**
```julia
# Continuous variables
FormulaCompiler.accumulate_ame_gradient!(gβ_sum, de, β, rows, var; link, backend)
FormulaCompiler.marginal_effects_eta!(g_buf, de, β, row; backend)

# Variable type detection
FormulaCompiler.continuous_variables(compiled, data_nt)

# Standard errors (both types)
FormulaCompiler.delta_method_se(gradient, Σ)
```

### **Profile Effects (profile.jl)**

**Core Function:** `profile_margins(model, reference_grid::DataFrame; ...)` → **MEM/APM**

**Strategy:**
- **User provides reference grid directly** (most efficient approach)
- **Single compilation** per reference grid structure  
- **Auto-detection** of variable types (continuous vs categorical)
- **Mixed variable support** in single call
- **Multiple dispatch** for convenience methods that build reference grids

**Core Implementation:**
```julia
# Primary method - user provides complete reference grid (most efficient)
function profile_margins(model, reference_grid::DataFrame; type=:effects, vars=[:x1], ...)
    data_nt = Tables.columntable(reference_grid)
    engine = build_engine(model, data_nt, vars)  # Single compilation
    
    # Auto-detect variable types
    continuous_vars = FormulaCompiler.continuous_variables(engine.compiled, data_nt)
    
    # Evaluate at each row of reference grid
    for row in 1:nrows(reference_grid)
        for var in vars
            if var ∈ continuous_vars
                # Marginal effect (derivative)
                FormulaCompiler.marginal_effects_eta!(engine.g_buf, engine.de, engine.β, row; backend)
            else
                # Row-specific baseline contrast: this row's category vs baseline at this profile
                _compute_row_specific_baseline_contrast!(engine.g_buf, engine, row, var)
            end
        end
    end
end

# Smart convenience method - handles both partial and complete reference grids
function profile_margins(model, reference_grid::DataFrame, data::DataFrame; kwargs...)
    # Auto-fills missing variables with typical values if needed
    complete_reference_grid = _ensure_complete_reference_grid(reference_grid, data, model)
    return profile_margins(model, complete_reference_grid; kwargs...)
end

# Other convenience methods build reference grids, then call primary method
function profile_margins(model, data::DataFrame; at=:means, kwargs...)
    reference_grid = _build_means_refgrid(data)
    return profile_margins(model, reference_grid; kwargs...)
end

function profile_margins(model, data::DataFrame; at::Dict, kwargs...)
    reference_grid = _build_cartesian_refgrid(at, data)  # Dict(:x => [1,2], :y => [3])
    return profile_margins(model, reference_grid; kwargs...)
end
```

## 📊 **Data Flow Architecture**

### **Population Margins Flow (AME/AAP)**

**Simple Population Margins:**
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

**Population Margins with `at` (Counterfactual):**
```
population_margins(model, data; type=:effects, at=Dict(:income => [30k, 50k]))
    ↓
For each at specification:
    Override specified variables for entire population
    ↓
    Compute AME on modified population
    ↓
Combine results with at identifiers
    ↓
MarginsResult(df, G, metadata)
```

**Population Margins with `over` (Subgroup Analysis):**
```
population_margins(model, data; type=:effects, over=(:age => [30, 50, 70], :region))
    ↓
Parse over specification:
    Continuous vars: create subgroups around specified values
    Categorical vars: use specified/all observed levels
    ↓
For each var in vars:
    Filter over spec (ignore if var appears in over)
    ↓
    Create subgroup combinations
    ↓
    Compute AME within each subgroup using actual data
    ↓
Combine results with group identifiers
    ↓
MarginsResult(df, G, metadata)
```

**Population Margins with Both `at` and `over`:**
```
population_margins(model, data; over=[:region], at=Dict(:income => [30k, 50k]))
    ↓
Create evaluation contexts: subgroups × counterfactual scenarios
    ↓
For each region × income combination:
    Take regional subgroup, override income, compute AME
    ↓
Combine results
```

### **Profile Margins Flow (MEM/APM)**

**Option 1: User provides complete reference grid (most efficient)**
```
profile_margins(model, reference_grid; type=:effects, vars=[:x1])  # MEM
    ↓
Tables.columntable(reference_grid)  # Single conversion
    ↓
build_engine(model, reference_data)  # Single compilation
    ↓
For each row in reference_grid:
    Auto-detect variable type (continuous vs categorical)
    ↓
    If continuous: marginal_effects_eta!(g_buf, de, β, row; backend)
    If categorical: compute_row_specific_baseline_contrast!(g_buf, engine, row, var)
    ↓
MarginsResult(df, G, metadata)   # Return structured result
```

**Option 2: Smart convenience method (reference grid + data)**
```
profile_margins(model, partial_grid, data; type=:effects, vars=[:x1])  # MEM
    ↓
_ensure_complete_reference_grid(partial_grid, data, model)  # Auto-fill missing vars
    ↓
profile_margins(model, complete_reference_grid; ...)  # Call primary method
```

**Option 3: Convenience method builds reference grid**
```
profile_margins(model, data; at=Dict(:x => [1,2], :y => [3]), type=:effects)  # MEM
    ↓
_build_cartesian_refgrid(at, data)   # Build reference grid from Dict + typical values
    ↓
profile_margins(model, reference_grid, data; ...)  # Call smart convenience method
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
FormulaCompiler.continuous_variables(compiled, data_nt)  # Variable type detection
```

**Zero-Allocation Evaluation:**
```julia
# Continuous variables
FormulaCompiler.marginal_effects_eta!(g_buf, de, β, row; backend)
FormulaCompiler.marginal_effects_mu!(g_buf, de, β, row; link, backend)
FormulaCompiler.accumulate_ame_gradient!(gβ_sum, de, β, rows, var; link, backend)

# Categorical variables (custom implementation)
_compute_row_specific_baseline_contrast!(g_buf, engine, row, var)  # Novel row-specific approach
_accumulate_categorical_baseline_ame!(gβ_sum, engine, rows, var)
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
# Both functions share core parameter semantics
population_margins(model, data; type, vars, target, backend, at, over, vcov, ...)
profile_margins(model, reference_grid; type, vars, target, backend, vcov, ...)
#                     ↑ at/over for counterfactuals/subgroups    ↑ reference_grid for evaluation points

# Convenience methods for profile_margins:
profile_margins(model, data; type, vars, target, backend, vcov, ...)           # at defaults to :means
profile_margins(model, data; at=:means, type, vars, target, backend, vcov, ...)
profile_margins(model, data; at=Dict(...), type, vars, target, backend, vcov, ...)
```

### **`at` vs `over` Parameters for Population Margins**

**Key Distinction (Following Stata's `margins` Command):**
- **`at`**: Counterfactual scenarios - override variable values for entire population
- **`over`**: Subgroup analysis - stratify by observed data groups

#### **`at` Parameter - Counterfactual Analysis (Stata's `at()` Option):**
```julia
# Equivalent to Stata: margins, dydx(education) at(income=(30000 50000))
population_margins(model, data; vars=[:education], at=Dict(:income => [30000, 50000]))
# "AME of education IF everyone had income=30k, IF everyone had income=50k"
```

**Process:**
1. Set **everyone's** income to 30,000
2. Compute education AME across full population with this override
3. Set **everyone's** income to 50,000  
4. Compute education AME across full population with this override

#### **`over` Parameter - Subgroup Analysis (Stata's `over()` Option):**

**Simple Categorical Stratification:**
```julia
# Equivalent to Stata: margins, dydx(education) over(region)
population_margins(model, data; vars=[:education], over=[:region])
# "AME of education within each observed region group"
```

**Enhanced Flexible Specification:**
```julia
# Hybrid syntax with precise control
population_margins(model, data; vars=[:education], 
                  over=(:age => [25, 45, 65],           # Continuous: specify subgroup centers
                       :gender => ["Man", "Woman"],     # Categorical: specify subset
                       :region))                         # Categorical: all observed levels
```

**Rules for Enhanced `over`:**
- **Continuous variables**: Must specify values (creates subgroups around those values)
- **Categorical variables**: 
  - Specified → use those levels only
  - Unspecified → use all observed levels
- **Error**: Continuous variables without specified values

#### **Combined Usage:**
```julia
population_margins(model, data; vars=[:education], 
                  over=[:region],                    # Subgroups: within each region
                  at=Dict(:income => [30k, 50k]))   # Counterfactual: at these income levels
# "AME of education within each region, at income=30k and at income=50k"
```

#### **Conflict Resolution:**
When a variable appears in both `vars` and `at`/`over`, the specification for that variable is ignored:
```julia
population_margins(model, data; vars=[:education, :income], at=Dict(:income => [30k, 50k]))
# education AME: computed at income = 30k, 50k
# income AME: computed across full sample (at[:income] ignored for income's own effect)
```

### **vars Parameter Usage**
```julia
# vars only needed for type=:effects (computing derivatives/contrasts)
population_margins(model, data; type=:effects, vars=[:x1, :catvar])  # ✅ Need vars (mixed types supported)
population_margins(model, data; type=:predictions)                  # ✅ No vars needed

profile_margins(model, refgrid; type=:effects, vars=[:x1, :catvar])  # ✅ Need vars (baseline contrasts for categorical)
profile_margins(model, refgrid; type=:predictions)                  # ✅ No vars needed
```

### **Categorical Variable Behavior**

**(Novel) Row-Specific Baseline Contrasts:**
```julia
# Reference grid specifies both covariate values AND categorical levels
reference_grid = DataFrame(
    x = [1, 2, 3],
    catvar = ["A", "B", "C"],  # A is baseline
    z = [10, 20, 30]
)

profile_margins(model, reference_grid; type=:effects, vars=[:catvar])
# Returns row-specific contrasts:
# Row 1: "catvar=A vs baseline at (x=1, z=10)" → 0.0 (A vs A)
# Row 2: "catvar=B vs baseline at (x=2, z=20)" → ŷ(x=2,catvar=B,z=20) - ŷ(x=2,catvar=A,z=20)
# Row 3: "catvar=C vs baseline at (x=3, z=30)" → ŷ(x=3,catvar=C,z=30) - ŷ(x=3,catvar=A,z=30)
```

**Key Innovation**: Each categorical effect is computed as the contrast between **that row's categorical value** and the **baseline**, evaluated at **that row's exact covariate profile**.

**Population Baseline Contrasts (Traditional):**
```julia
# Population margins use traditional baseline contrasts across sample
population_margins(model, data; type=:effects, vars=[:catvar]) 
# Returns average baseline contrasts: E[ŷ|catvar=level] - E[ŷ|catvar=baseline]
```

**Advanced Contrasts (Future Extensions):**
```julia
# For all pairwise contrasts at each reference point:
profile_contrasts(model, refgrid; vars=[:catvar], contrast=:pairwise)
# Would return: all possible pairwise comparisons at each row

# For traditional "one contrast per level" at each reference point:
profile_contrasts(model, refgrid; vars=[:catvar], contrast=:baseline_each_row)
# Would return: B vs A, C vs A at row 1; B vs A, C vs A at row 2; etc.
```

### **Error Handling Strategy**

**Input Validation (Early):**
```julia
function _validate_variables(data_nt, vars, continuous_vars)
    for var in vars
        haskey(data_nt, var) || error("Variable $var not found")
        # Auto-detection determines if continuous (derivative) or categorical (baseline contrast)
        # Both types supported in unified API
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