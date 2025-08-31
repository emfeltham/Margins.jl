# FILE_PLAN.md: Detailed Implementation Plan for 4-File Architecture

## Overview

This document provides the detailed implementation plan for the radical reorganization of Margins.jl from 17+ files to a clean organized architecture with logical subdirectories. Each file's responsibilities, functions, and implementation priorities are specified.

## 🎯 **Design Principles**

1. **Statistical Correctness (PARAMOUNT)**: All standard errors mathematically exact using delta-method
2. **Performance (PARAMOUNT)**: Zero-allocation paths leveraging FormulaCompiler's capabilities
3. **Julian Style**: Clean, idiomatic code with proper multiple dispatch
4. **FormulaCompiler Integration**: Use FC as intended, not fight against it

## 📁 **Realistic File Architecture**

```
src/
├── Margins.jl              # Module definition, exports (~50 lines)
├── types.jl                 # MarginsResult, error types (~100 lines)  
├── engine/
│   ├── core.jl             # MarginsEngine, construction (~150 lines)
│   ├── utilities.jl        # Shared utilities, validation (~100 lines)
│   └── caching.jl          # Compilation caching (~50 lines)
├── population/
│   ├── core.jl             # Main population_margins() (~100 lines)
│   ├── contexts.jl         # at/over parameter handling (~100 lines)
│   └── effects.jl          # AME computation (~100 lines)
└── profile/
    ├── core.jl             # Main profile_margins() (~100 lines)
    ├── refgrids.jl         # Reference grid builders (~100 lines)
    └── contrasts.jl        # Row-specific baseline contrasts (~100 lines)

Total: ~1050 lines across 12 files (vs current ~3000+ lines across 17+ files)
```

**Design Rationale:**
- **Organized subdirectories** for logical grouping and maintainability
- **Clear separation** of concerns while maintaining conceptual unity
- **Parallel development** friendly with minimal merge conflicts
- **Easy navigation** without excessive directory depth
- **Unit testing** friendly with isolated components

---

## 📄 **File 1: `src/Margins.jl` (~50 lines)**

**Purpose**: Module definition and exports only

### **Responsibilities:**
- Module definition and version info
- Public API exports
- Include statements for all submodules

### **Key Components:**

#### **Module Definition:**
```julia
module Margins

# Dependencies
using Tables, DataFrames, StatsModels, GLM
using FormulaCompiler
using LinearAlgebra: dot
using Statistics: mean

# Version info
const VERSION = v"2.0.0"

# Main exports
export population_margins, profile_margins, MarginsResult

# Advanced exports (future)
export population_margins!, profile_margins!  # In-place versions

# Include all submodules
include("types.jl")
include("engine/core.jl")
include("engine/utilities.jl")
include("engine/caching.jl")
include("population/core.jl")
include("population/contexts.jl")
include("population/effects.jl")
include("profile/core.jl")
include("profile/refgrids.jl")
include("profile/contrasts.jl")

end # module
```

### **Implementation Priority**: Phase 1 - Week 1 (Day 1)
- First file to implement (required for module structure)
- Minimal content - just exports and includes
- Enables other files to be developed in parallel

---

## 📄 **File 2: `src/types.jl` (~100 lines)**

**Purpose**: Result types, error types, and display methods

### **Responsibilities:**
- `MarginsResult` type definition
- Tables.jl interface implementation  
- Display and printing methods
- Custom error types

### **Key Components:**

#### **MarginsResult Type:**
```julia
"""
    MarginsResult

Container for marginal effects results with Tables.jl interface.

Fields:
- `df::DataFrame`: Results table with estimates, standard errors, etc.
- `gradients::Matrix{Float64}`: Parameter gradients (G matrix) for delta-method
- `metadata::Dict`: Analysis metadata (model info, options used, etc.)
"""
struct MarginsResult
    df::DataFrame
    gradients::Matrix{Float64}
    metadata::Dict{Symbol, Any}
end

# Tables.jl interface
Tables.istable(::Type{MarginsResult}) = true
Tables.rowaccess(::Type{MarginsResult}) = true
Tables.rows(mr::MarginsResult) = Tables.rows(mr.df)
Tables.schema(mr::MarginsResult) = Tables.schema(mr.df)

# DataFrame conversion
Base.convert(::Type{DataFrame}, mr::MarginsResult) = mr.df
DataFrame(mr::MarginsResult) = mr.df
```

#### **Display Methods:**
```julia
# Compact display showing key results
function Base.show(io::IO, mr::MarginsResult)
    n_effects = nrow(mr.df)
    n_vars = get(mr.metadata, :n_vars, "unknown")
    analysis_type = get(mr.metadata, :type, "unknown")
    
    println(io, "MarginsResult: $n_effects $analysis_type effects")
    println(io, "Variables: $n_vars")
    show(io, mr.df)
end

# Detailed display with metadata
function Base.show(io::IO, ::MIME"text/plain", mr::MarginsResult)
    show(io, mr)
    println(io, "\nMetadata:")
    for (k, v) in mr.metadata
        println(io, "  $k: $v")
    end
end
```

#### **Error Types:**
```julia
# Custom error types for clear user feedback
struct MarginsError <: Exception
    msg::String
end

struct StatisticalValidityError <: Exception
    msg::String
end
```

### **Implementation Priority**: Phase 1 - Week 1 (Day 1)
- Essential for other files to compile and test
- Provides basic result infrastructure

---

## 📄 **File 3: `src/engine/core.jl` (~150 lines)**

**Purpose**: MarginsEngine struct and construction logic

### **Responsibilities:**
- `MarginsEngine` struct definition
- Engine construction with FormulaCompiler integration
- Zero-allocation buffer allocation
- Link function detection

### **Key Components:**

#### **MarginsEngine Struct:**
```julia
"""
    MarginsEngine{L<:GLM.Link}

Zero-allocation engine for marginal effects computation.

Built on FormulaCompiler.jl with pre-allocated buffers for maximum performance.
"""
struct MarginsEngine{L<:GLM.Link}
    # FormulaCompiler components (pre-compiled)
    compiled::FormulaCompiler.CompiledFormula
    de::Union{FormulaCompiler.DerivativeEvaluator, Nothing}
    
    # Pre-allocated buffers (zero runtime allocation)
    g_buf::Vector{Float64}              # Marginal effects results
    gβ_accumulator::Vector{Float64}     # AME gradient accumulation
    
    # Model parameters
    model::Any                          # Reference to original model
    β::Vector{Float64}
    Σ::Matrix{Float64}
    link::L
    vars::Vector{Symbol}
    data_nt::NamedTuple  # Reference for scenarios/refgrids
end
```

#### **Engine Construction:**
```julia
"""
    build_engine(model, data_nt, vars) -> MarginsEngine

Construct zero-allocation margins engine with FormulaCompiler integration.
"""
function build_engine(model, data_nt::NamedTuple, vars)
    # Input validation (delegated to utilities.jl)
    _validate_variables(data_nt, vars)
    
    # Compile formula with FormulaCompiler
    compiled = FormulaCompiler.compile_formula(model, data_nt)
    continuous_vars = FormulaCompiler.continuous_variables(compiled, data_nt)
    vars_for_de = filter(v -> v in continuous_vars, vars)
    
    # Build derivative evaluator only if needed
    de = isempty(vars_for_de) ? nothing : 
         FormulaCompiler.build_derivative_evaluator(compiled, data_nt; vars=vars_for_de)
    
    # Pre-allocate buffers (zero runtime allocation)
    n_vars = length(vars_for_de)
    n_coef = length(compiled)
    g_buf = Vector{Float64}(undef, max(n_vars, 1))  # At least size 1
    gβ_accumulator = Vector{Float64}(undef, n_coef)
    
    return MarginsEngine(
        compiled, de, g_buf, gβ_accumulator,
        model, coef(model), vcov(model), _auto_link(model), vars, data_nt
    )
end

"""
    _auto_link(model) -> GLM.Link

Automatically determine link function from model.
"""
function _auto_link(model)
    if hasfield(typeof(model), :model) && hasfield(typeof(model.model), :rr)
        return model.model.rr.d.link  # GLM.jl pattern
    else
        return GLM.IdentityLink()  # Default fallback
    end
end
```

#### **Variable Type Detection:**
```julia
"""
    _validate_variables(data_nt, vars)

Validate that requested variables exist and are analyzable.
"""
function _validate_variables(data_nt::NamedTuple, vars)
    for var in vars
        haskey(data_nt, var) || throw(MarginsError("Variable $var not found in data"))
        # Additional validation as needed
    end
end

"""
    _get_baseline_level(model, var) -> baseline_level

Extract baseline level from model's contrast coding (statistically principled).
"""
function _get_baseline_level(model, var)
    # Extract baseline from model's contrast system
    if hasfield(typeof(model), :mf) && hasfield(typeof(model.mf), :contrasts)
        if haskey(model.mf.contrasts, var)
            contrast = model.mf.contrasts[var]
            if contrast isa StatsModels.DummyCoding
                return contrast.base
            elseif contrast isa StatsModels.EffectsCoding  
                return contrast.base
            # Add other contrast types as needed
            end
        end
    end
    
    throw(MarginsError("Could not determine baseline level for variable $var from model contrasts. " *
                      "Please ensure the model has proper contrast coding information."))
end

"""
    _is_continuous_variable(col) -> Bool

Determine if a data column represents a continuous variable.
"""
function _is_continuous_variable(col)
    return eltype(col) <: Real && !(eltype(col) <: Bool)
end
```

#### **Core Prediction Utilities:**
```julia
"""
    _predict_at_row(engine, row, target) -> Float64

Predict outcome at specific data row.
"""
function _predict_at_row(engine::MarginsEngine, row::Int, target::Symbol)
    # Use FormulaCompiler for zero-allocation prediction
    if target === :eta
        return FormulaCompiler.evaluate_eta(engine.compiled, engine.β, row)
    else # :mu
        η = FormulaCompiler.evaluate_eta(engine.compiled, engine.β, row)
        return GLM.linkinv(engine.link, η)
    end
end

"""
    _predict_at_profile(engine, profile, target) -> Float64

Predict outcome at specific profile (Dict of variable values).
"""
function _predict_at_profile(engine::MarginsEngine, profile::Dict, target::Symbol)
    # Create minimal reference data for this profile
    profile_data = _build_profile_data(profile, engine.data_nt)
    profile_compiled = FormulaCompiler.compile_formula(engine.model, profile_data)
    
    if target === :eta
        return FormulaCompiler.evaluate_eta(profile_compiled, engine.β, 1)
    else # :mu  
        η = FormulaCompiler.evaluate_eta(profile_compiled, engine.β, 1)
        return GLM.linkinv(engine.link, η)
    end
end
```

#### **Result Construction:**
```julia
"""
    _build_metadata(; kwargs...) -> Dict

Build metadata dictionary for MarginsResult.
"""
function _build_metadata(; 
    type=:unknown, 
    vars=Symbol[], 
    target=:mu, 
    backend=:ad,
    n_obs=0,
    model_type="unknown",
    timestamp=now(),
    kwargs...)
    
    return Dict{Symbol, Any}(
        :type => type,
        :vars => vars,
        :n_vars => length(vars),
        :target => target,
        :backend => backend, 
        :n_obs => n_obs,
        :model_type => string(typeof(model_type)),
        :timestamp => timestamp,
        :additional => Dict(kwargs...)
    )
end
```

#### **Utility Functions:**
```julia
"""
    _auto_link(model) -> GLM.Link

Automatically determine link function from model.
"""
function _auto_link(model)
    if hasfield(typeof(model), :model) && hasfield(typeof(model.model), :rr)
        return model.model.rr.d.link  # GLM.jl pattern
    else
        return GLM.IdentityLink()  # Default fallback
    end
end

"""
    _build_profile_data(profile, original_data) -> NamedTuple

Build minimal synthetic data for profile evaluation.
"""
function _build_profile_data(profile::Dict, original_data::NamedTuple)
    profile_data = NamedTuple()
    for (var, val) in pairs(original_data)
        if haskey(profile, var)
            # Use profile value
            profile_data = merge(profile_data, NamedTuple{(var,)}(([profile[var]],)))
        else
            # Use representative value
            if _is_continuous_variable(val)
                profile_data = merge(profile_data, NamedTuple{(var,)}(([mean(val)],)))
            else
                profile_data = merge(profile_data, NamedTuple{(var,)}(([first(val)],)))
            end
        end
    end
    return profile_data
end
```

### **Implementation Priority**: Phase 1 - Week 1 (Day 2)
- Core engine structure needed by all other files
- Zero-allocation buffer management critical for performance

---

## 📄 **File 4: `src/engine/utilities.jl` (~100 lines)**

**Purpose**: Shared utility functions used across the package

### **Responsibilities:**
- Variable type detection and validation
- Baseline level detection for categorical variables
- Core prediction utilities
- Result metadata construction
- Profile data building

### **Key Components:**

```julia
# Variable validation and type detection
_validate_variables(data_nt, vars)
_get_baseline_level(model, var)
_is_continuous_variable(col)

# Prediction utilities
_predict_at_row(engine, row, target)
_predict_at_profile(engine, profile, target)

# Result construction
_build_metadata(; kwargs...)
_build_profile_data(profile, original_data)
```

### **Implementation Priority**: Phase 1 - Week 1 (Day 2-3)
- Shared by both population and profile implementations
- Must be completed before core functionality

---

## 📄 **File 5: `src/engine/caching.jl` (~50 lines)**

**Purpose**: Engine compilation caching for performance

### **Responsibilities:**
- Global engine cache management
- Cache key generation
- Engine retrieval and storage

### **Key Components:**

```julia
# Global compilation cache (MARGINS_GUIDE.md pattern)
const ENGINE_CACHE = Dict{UInt64, Any}()

"""
    _get_or_build_engine(model, data_nt, vars) -> MarginsEngine

Get cached engine or build new one. Critical for performance.
"""
function _get_or_build_engine(model, data_nt::NamedTuple, vars)
    cache_key = hash(model, keys(data_nt), vars)
    return get!(ENGINE_CACHE, cache_key) do
        build_engine(model, data_nt, vars)
    end
end
```

### **Implementation Priority**: Phase 1 - Week 1 (Day 3)
- Performance optimization component
- Simple but critical for speed

---

## 📄 **File 6: `src/population/core.jl` (~100 lines)**

**Purpose**: Main population_margins() entry point

### **Responsibilities:**
- `population_margins()` main entry point function
- Parameter validation and preprocessing
- Delegation to specialized computation functions
- Simple cases without at/over parameters

### **Key Components:**

#### **Main Entry Point:**
```julia
"""
    population_margins(model, data; kwargs...) -> MarginsResult

Compute population marginal effects (AME) or predictions (AAP).

# Arguments
- `model`: Fitted statistical model (GLM.jl, MLJ.jl, etc.)  
- `data`: Data frame or table used to fit the model

# Keyword Arguments
- `type::Symbol=:effects`: `:effects` for marginal effects, `:predictions` for fitted values
- `vars=nothing`: Variables for effects (auto-detected if `nothing`)
- `target::Symbol=:mu`: `:mu` for response scale, `:eta` for link scale
- `backend::Symbol=:fd`: `:fd` for zero allocation, `:ad` for higher accuracy
- `at=nothing`: Counterfactual scenarios (Dict of variable => values)
- `over=nothing`: Subgroup analysis (Vector of vars or NamedTuple specification)
- `vcov=nothing`: Custom covariance matrix (future feature)

# Examples
```julia
# Basic AME across population
population_margins(model, data; type=:effects, vars=[:education, :income])

# AME at specific counterfactual values (like Stata's at() option)
population_margins(model, data; vars=[:education], at=Dict(:income => [30000, 50000]))

# AME within subgroups (like Stata's over() option)
population_margins(model, data; vars=[:education], over=[:region])

# Enhanced subgroup specification
population_margins(model, data; vars=[:education], 
                  over=(:age => [25, 45, 65], :region))

# Combined counterfactual and subgroup analysis
population_margins(model, data; vars=[:education], 
                  over=[:region], at=Dict(:income => [30000, 50000]))
```
"""
function population_margins(model, data; 
                           type::Symbol=:effects, 
                           vars=nothing, 
                           target::Symbol=:mu, 
                           backend::Symbol=:fd,  # Default to FD for zero allocation
                           at=nothing, 
                           over=nothing,
                           vcov=nothing,
                           kwargs...)
    # Single data conversion (consistent format throughout)
    data_nt = Tables.columntable(data)
    
    # Handle vars parameter (only needed for type=:effects)
    if type === :effects
        if vars === nothing
            # Auto-detect all continuous variables
            compiled = FormulaCompiler.compile_formula(model, data_nt)
            vars = FormulaCompiler.continuous_variables(compiled, data_nt)
            isempty(vars) && throw(MarginsError("No continuous variables found for effects. Specify vars explicitly."))
        end
    else # type === :predictions
        vars = Symbol[]  # Not needed for predictions
    end
    
    # Build zero-allocation engine with caching
    engine = _get_or_build_engine(model, data_nt, vars)
    
    # Handle at/over parameters for population contexts
    if at !== nothing || over !== nothing
        return _population_margins_with_contexts(engine, data_nt, vars, at, over; 
                                               type, target, backend, kwargs...)
    end
    
    # Simple case: no at/over parameters
    if type === :effects
        df, G = _population_effects(engine, data_nt; target, backend, kwargs...)
        metadata = _build_metadata(; type, vars, target, backend, n_obs=length(first(data_nt)), 
                                   model_type=typeof(model))
        return MarginsResult(df, G, metadata)
    else # :predictions
        df, G = _population_predictions(engine, data_nt; target, kwargs...)
        metadata = _build_metadata(; type, vars=Symbol[], target, n_obs=length(first(data_nt)), 
                                   model_type=typeof(model))
        return MarginsResult(df, G, metadata)
    end
end
```

#### **Context Handling:**
```julia
"""
    _population_margins_with_contexts(engine, data_nt, vars, at, over; kwargs...)

Handle population margins with at/over parameter contexts.
"""
function _population_margins_with_contexts(engine, data_nt, vars, at, over; 
                                         type, target, backend, kwargs...)
    results = DataFrame()
    all_gradients = Matrix{Float64}(undef, 0, length(engine.β))
    
    # Parse specifications
    at_specs = at === nothing ? [Dict()] : _parse_at_specification(at)
    over_specs = over === nothing ? [Dict()] : _parse_over_specification(over, data_nt)
    
    # Create all combinations of contexts
    for at_spec in at_specs, over_spec in over_specs
        for var in vars
            # Skip if this var appears in at/over (conflict resolution)
            if haskey(at_spec, var) || haskey(over_spec, var)
                continue
            end
            
            # Create context data for this combination
            context_data = _create_context_data(data_nt, at_spec, over_spec)
            
            # Compute effect/prediction in this context
            if type === :effects
                var_result, var_gradients = _compute_population_effect_in_context(
                    engine, context_data, var, target, backend)
            else
                var_result, var_gradients = _compute_population_prediction_in_context(
                    engine, context_data, target)
            end
            
            # Add context identifiers
            for (ctx_var, ctx_val) in merge(at_spec, over_spec)
                var_result[ctx_var] = ctx_val
            end
            
            append!(results, var_result)
            all_gradients = vcat(all_gradients, var_gradients)
        end
    end
    
    metadata = _build_metadata(; type, vars, target, backend, n_obs=length(first(data_nt)), 
                               model_type=typeof(engine.model), at_specs, over_specs)
    return MarginsResult(results, all_gradients, metadata)
end
```

#### **Core Population Effects:**
```julia
"""
    _population_effects(engine, data_nt; target, backend) -> (DataFrame, Matrix)

Compute population marginal effects for all variables (both continuous and categorical).
"""
function _population_effects(engine::MarginsEngine, data_nt; target=:mu, backend=:fd)
    engine.de === nothing && return (DataFrame(), Matrix{Float64}(undef, 0, length(engine.β)))
    
    rows = 1:length(first(data_nt))
    results = DataFrame(term=String[], estimate=Float64[], se=Float64[])
    n_effects = 0
    
    # Auto-detect variable types
    continuous_vars = FormulaCompiler.continuous_variables(engine.compiled, data_nt)
    
    # Count total effects for gradient matrix sizing
    for var in engine.vars
        if var ∈ continuous_vars
            n_effects += 1  # One effect per continuous variable
        else
            # Count non-baseline levels for categorical variables
            baseline = _get_baseline_level(engine.model, var)
            levels = unique(data_nt[var])
            n_effects += length(levels) - 1  # Exclude baseline
        end
    end
    
    G = Matrix{Float64}(undef, n_effects, length(engine.β))
    effect_idx = 1
    
    # Process each variable
    for var in engine.vars
        if var ∈ continuous_vars
            # Continuous variable: use FormulaCompiler's built-in AME
            ame_val, gβ_avg = _compute_continuous_ame(engine, var, rows, target, backend)
            se = FormulaCompiler.delta_method_se(gβ_avg, engine.Σ)
            
            push!(results, (term=string(var), estimate=ame_val, se=se))
            G[effect_idx, :] = gβ_avg
            effect_idx += 1
        else
            # Categorical variable: compute baseline contrasts
            baseline_results = _compute_categorical_baseline_contrasts(engine, var, rows, target, backend)
            for (level, estimate, gradient) in baseline_results
                se = FormulaCompiler.delta_method_se(gradient, engine.Σ)
                push!(results, (term="$var=$level vs baseline", estimate=estimate, se=se))
                G[effect_idx, :] = gradient
                effect_idx += 1
            end
        end
    end
    
    return (results, G)
end
```

#### **Parameter Parsing:**
```julia
"""
    _parse_at_specification(at) -> Vector{Dict}

Parse at parameter into evaluation contexts (counterfactual scenarios).
"""
function _parse_at_specification(at)
    if at isa Dict
        # Create all combinations of at values
        var_names = collect(keys(at))
        var_values = [at[k] for k in var_names]
        contexts = []
        for combo in Iterators.product(var_values...)
            context = Dict(zip(var_names, combo))
            push!(contexts, context)
        end
        return contexts
    else
        throw(MarginsError("at parameter must be a Dict specifying variable values"))
    end
end

"""
    _parse_over_specification(over, data_nt) -> Vector{Dict}

Parse over parameter into subgroup contexts.
"""
function _parse_over_specification(over, data_nt)
    if over isa NamedTuple
        # Enhanced flexible syntax: (:age => [25, 45, 65], :region)
        contexts = [Dict()]
        
        for (var, vals) in pairs(over)
            new_contexts = []
            if vals === nothing
                # Unspecified - use all observed levels (categorical only)
                if _is_continuous_variable(data_nt[var])
                    throw(MarginsError("Continuous variable $var in over() must specify values. " *
                                     "Use over=($var => [values], ...)"))
                end
                for ctx in contexts, val in unique(data_nt[var])
                    push!(new_contexts, merge(ctx, Dict(var => val)))
                end
            else
                # Specified values
                if _is_continuous_variable(data_nt[var])
                    # Create subgroups around specified values
                    subgroups = _create_continuous_subgroups(data_nt[var], vals)
                    for ctx in contexts, sg in subgroups
                        push!(new_contexts, merge(ctx, Dict(var => sg)))
                    end
                else
                    # Use specified categorical levels
                    for ctx in contexts, val in vals
                        push!(new_contexts, merge(ctx, Dict(var => val)))
                    end
                end
            end
            contexts = new_contexts
        end
        
        return contexts
    elseif over isa Vector
        # Simple vector syntax: [:region, :age_group] (categorical only)
        contexts = [Dict()]
        for var in over
            if _is_continuous_variable(data_nt[var])
                throw(MarginsError("Continuous variable $var in over() must specify values. " *
                                 "Use over=($var => [values], ...) syntax"))
            end
            new_contexts = []
            for ctx in contexts, val in unique(data_nt[var])
                push!(new_contexts, merge(ctx, Dict(var => val)))
            end
            contexts = new_contexts
        end
        return contexts
    else
        throw(MarginsError("over parameter must be a Vector or NamedTuple"))
    end
end
```

### **Implementation Priority**: Phase 2 - Week 2 (Day 1)
- Main user-facing API
- Coordinates between contexts and effects modules

---

## 📄 **File 7: `src/population/contexts.jl` (~100 lines)**

**Purpose**: Handle at/over parameter parsing and context creation

### **Responsibilities:**
- `at` parameter parsing (counterfactual scenarios)
- `over` parameter parsing (subgroup analysis)
- Context data creation and modification
- Population margins with complex contexts

### **Key Components:**
```julia
_population_margins_with_contexts(engine, data_nt, vars, at, over; kwargs...)
_parse_at_specification(at)
_parse_over_specification(over, data_nt)
_create_context_data(data_nt, at_spec, over_spec)
_create_continuous_subgroups(col, specified_values)
```

### **Implementation Priority**: Phase 2 - Week 2 (Day 2)
- Complex logic for at/over parameter handling
- Stata compatibility features

---

## 📄 **File 8: `src/population/effects.jl` (~100 lines)**

**Purpose**: Core population-level AME computation

### **Responsibilities:**
- Population effects computation (both continuous and categorical)
- FormulaCompiler integration for zero-allocation AME
- Traditional baseline contrasts for categorical variables
- Delta-method standard error computation

### **Key Components:**
```julia
_population_effects(engine, data_nt; target, backend)
_compute_continuous_ame(engine, var, rows, target, backend)
_compute_categorical_baseline_contrasts(engine, var, rows, target, backend)
```

### **Implementation Priority**: Phase 2 - Week 2 (Day 3)
- Core statistical computation
- Must achieve zero-allocation targets

---

## 📄 **File 9: `src/profile/core.jl` (~100 lines)**

**Purpose**: Main profile_margins() entry point with multiple dispatch

### **Responsibilities:**
- `profile_margins()` main entry point with multiple dispatch methods
- Parameter validation and preprocessing
- Delegation to specialized computation functions
- Coordination between reference grids and effects computation

### **Key Components:**

#### **Main Entry Points:**
```julia
"""
    profile_margins(model, reference_grid; kwargs...) -> MarginsResult

Primary method: user provides complete reference grid (most efficient).

# Arguments  
- `model`: Fitted statistical model
- `reference_grid::DataFrame`: Complete reference grid specifying evaluation points

# Examples
```julia
# User provides exact evaluation points
reference_grid = DataFrame(
    age = [25, 45, 65],
    income = [30000, 50000, 70000],
    region = ["West", "East", "North"]
)
profile_margins(model, reference_grid; type=:effects, vars=[:education])
```
"""
function profile_margins(model, reference_grid::DataFrame; 
                        type::Symbol=:effects, 
                        vars=nothing,
                        target::Symbol=:mu, 
                        backend::Symbol=:ad,  # Default to AD for accuracy in profiles
                        kwargs...)
    # Single data conversion  
    data_nt = Tables.columntable(reference_grid)
    
    # Handle vars parameter
    if type === :effects && vars === nothing
        throw(MarginsError("vars parameter required for type=:effects"))
    elseif type === :predictions
        vars = Symbol[]
    end
    
    # Single compilation per reference grid structure
    engine = build_engine(model, data_nt, vars)
    
    if type === :effects
        # Convert reference_grid to profiles for row-specific processing
        profiles = [Dict(pairs(row)) for row in eachrow(reference_grid)]
        df, G = _profile_effects(engine, profiles; target, backend, kwargs...)
        metadata = _build_metadata(; type, vars, target, backend, n_obs=nrow(reference_grid), 
                                   model_type=typeof(model))
        return MarginsResult(df, G, metadata)
    else # :predictions
        df, G = _profile_predictions(engine, data_nt; target, kwargs...)
        metadata = _build_metadata(; type, vars=Symbol[], target, n_obs=nrow(reference_grid), 
                                   model_type=typeof(model))
        return MarginsResult(df, G, metadata)
    end
end

"""
    profile_margins(model, reference_grid, data; kwargs...) -> MarginsResult

Smart convenience method: auto-fills missing variables with typical values.
"""
function profile_margins(model, reference_grid::DataFrame, data::DataFrame; kwargs...)
    # Auto-complete partial reference grids
    complete_reference_grid = _ensure_complete_reference_grid(reference_grid, data, model)
    return profile_margins(model, complete_reference_grid; kwargs...)
end

"""
    profile_margins(model, data; at=:means, kwargs...) -> MarginsResult

Convenience method: build reference grid from at specification.
"""
function profile_margins(model, data::DataFrame; at=:means, kwargs...)
    if at === :means
        reference_grid = _build_means_refgrid(data)
    elseif at isa Dict
        reference_grid = _build_cartesian_refgrid(at, data)
    elseif at isa Vector
        reference_grid = _build_explicit_refgrid(at, data)  
    else
        throw(MarginsError("at parameter must be :means, Dict, or Vector"))
    end
    return profile_margins(model, reference_grid; kwargs...)
end

# Default: at=:means if not specified
function profile_margins(model, data::DataFrame; kwargs...)
    return profile_margins(model, data; at=:means, kwargs...)
end
```

#### **Novel Row-Specific Effects:**
```julia
"""
    _profile_effects(engine, profiles; target, backend) -> (DataFrame, Matrix)

Compute profile marginal effects with novel row-specific baseline contrasts.
"""
function _profile_effects(engine::MarginsEngine, profiles; target=:mu, backend=:ad)
    engine.de === nothing && return (DataFrame(), Matrix{Float64}(undef, 0, length(engine.β)))
    
    results = DataFrame(term=String[], estimate=Float64[], se=Float64[])
    n_profiles = length(profiles)
    n_vars = length(engine.vars)
    
    # Auto-detect variable types
    continuous_vars = FormulaCompiler.continuous_variables(engine.compiled, engine.data_nt)
    
    # Count total effects for gradient matrix sizing
    total_effects = n_profiles * n_vars
    G = Matrix{Float64}(undef, total_effects, length(engine.β))
    effect_idx = 1
    
    for (profile_idx, profile) in enumerate(profiles)
        # Build minimal reference grid data for this profile
        profile_data = _build_profile_data(profile, engine.data_nt)
        profile_compiled = FormulaCompiler.compile_formula(engine.model, profile_data)
        
        # Build derivative evaluator for continuous variables at this profile
        continuous_vars_here = filter(v -> v in continuous_vars, engine.vars)
        profile_de = isempty(continuous_vars_here) ? nothing :
                    FormulaCompiler.build_derivative_evaluator(profile_compiled, profile_data; 
                                                             vars=continuous_vars_here)
        
        for var in engine.vars
            if var ∈ continuous_vars
                # Continuous variable: derivative at this profile
                if target === :mu
                    FormulaCompiler.marginal_effects_mu!(engine.g_buf, profile_de, engine.β, 1;
                                                       link=engine.link, backend=backend)
                else
                    FormulaCompiler.marginal_effects_eta!(engine.g_buf, profile_de, engine.β, 1;
                                                        backend=backend)
                end
                var_idx = findfirst(==(var), continuous_vars_here)
                effect_val = engine.g_buf[var_idx]
                
                # Compute gradient for SE
                if target === :mu
                    FormulaCompiler.me_mu_grad_beta!(engine.gβ_accumulator, profile_de, engine.β, 1, var;
                                                   link=engine.link)
                else
                    FormulaCompiler.me_eta_grad_beta!(engine.gβ_accumulator, profile_de, engine.β, 1, var)
                end
                se = FormulaCompiler.delta_method_se(engine.gβ_accumulator, engine.Σ)
                
                # Build term description
                profile_desc = join(["$(k)=$(v)" for (k,v) in pairs(profile)], ", ")
                term_name = "$var at ($profile_desc)"
                
                push!(results, (term=term_name, estimate=effect_val, se=se))
                G[effect_idx, :] = engine.gβ_accumulator
                effect_idx += 1
            else
                # Categorical variable: novel row-specific baseline contrast
                effect_val, gradient = _compute_row_specific_baseline_contrast(
                    engine, profile_data, profile, var, target, backend)
                se = FormulaCompiler.delta_method_se(gradient, engine.Σ)
                
                # Build term description showing specific contrast
                current_level = profile[var]
                baseline_level = _get_baseline_level(engine.model, var)
                profile_desc = join(["$(k)=$(v)" for (k,v) in pairs(profile)], ", ")
                term_name = "$var=$current_level vs $baseline_level at ($profile_desc)"
                
                push!(results, (term=term_name, estimate=effect_val, se=se))
                G[effect_idx, :] = gradient
                effect_idx += 1
            end
        end
    end
    
    return (results, G)
end

"""
    _compute_row_specific_baseline_contrast(engine, profile_data, profile, var, target, backend)

Novel approach: compute contrast between this row's category and baseline at this profile.
"""
function _compute_row_specific_baseline_contrast(engine, profile_data, profile, var, target, backend)
    baseline_level = _get_baseline_level(engine.model, var)
    current_level = profile[var]
    
    # If this row already has baseline level, contrast is 0
    if current_level == baseline_level
        zero_gradient = zeros(Float64, length(engine.β))
        return 0.0, zero_gradient
    end
    
    # Compute prediction at current profile
    current_pred = _predict_at_profile(engine, profile, target)
    
    # Compute prediction at baseline profile (same covariates, baseline category)
    baseline_profile = copy(profile)
    baseline_profile[var] = baseline_level
    baseline_pred = _predict_at_profile(engine, baseline_profile, target)
    
    # Compute gradient for this contrast (finite differences on parameter vector)
    gradient = _compute_contrast_gradient(engine, profile, baseline_profile, target)
    
    return current_pred - baseline_pred, gradient
end
```

#### **Reference Grid Builders:**
```julia
"""
    _build_means_refgrid(data) -> DataFrame

Build reference grid with means for continuous, first levels for categorical.
"""
function _build_means_refgrid(data::DataFrame)
    row = Dict{Symbol,Any}()
    for (name, col) in pairs(eachcol(data))
        if _is_continuous_variable(col)
            row[name] = mean(col)
        elseif col isa CategoricalArray
            row[name] = levels(col)[1] 
        elseif eltype(col) <: Bool
            row[name] = false
        else
            row[name] = first(col)
        end
    end
    return DataFrame([row])
end

"""
    _build_cartesian_refgrid(at, data) -> DataFrame

Build Cartesian product reference grid from Dict specification.
"""
function _build_cartesian_refgrid(at::Dict, data::DataFrame)
    # Get base row with typical values
    base_row = _build_means_refgrid(data)[1, :]
    
    # Create Cartesian product of specified values
    var_names = collect(keys(at))
    var_values = [at[k] for k in var_names]
    
    grid_rows = []
    for combo in Iterators.product(var_values...)
        row = copy(base_row)
        for (i, var) in enumerate(var_names)
            row[var] = combo[i]
        end
        push!(grid_rows, row)
    end
    
    return DataFrame(grid_rows)
end

"""
    _ensure_complete_reference_grid(reference_grid, original_data, model) -> DataFrame

Ensure reference grid contains all variables needed by the model.
"""
function _ensure_complete_reference_grid(reference_grid::DataFrame, original_data::DataFrame, model)
    # Get all variables needed by the model  
    model_vars = Set(Symbol.(StatsModels.coefnames(model.mf.f)))
    grid_vars = Set(Symbol.(names(reference_grid)))
    missing_vars = setdiff(model_vars, grid_vars)
    
    # If already complete, return as-is
    if isempty(missing_vars)
        return reference_grid
    end
    
    # Fill missing variables with typical values
    completed_grid = copy(reference_grid)
    n_rows = nrow(reference_grid)
    
    for var in missing_vars
        if hasproperty(original_data, var)
            col = getproperty(original_data, var)
            typical_val = _get_typical_value(col)
            completed_grid[!, var] = fill(typical_val, n_rows)
        end
    end
    
    return completed_grid
end

"""
    _get_typical_value(col) -> typical_value

Get representative value for a data column.
"""
function _get_typical_value(col)
    if _is_continuous_variable(col)
        return mean(col)
    elseif col isa CategoricalArray
        return levels(col)[1]  # First level
    elseif eltype(col) <: Bool  
        return mode(col)  # Most common boolean value
    elseif eltype(col) <: AbstractString
        return mode(col)  # Most common string
    else
        return first(col)  # Fallback
    end
end
```

### **Implementation Priority**: Phase 2 - Week 2 (Day 4)
- Main user-facing API for profile analysis
- Multiple dispatch coordination

---

## 📄 **File 10: `src/profile/refgrids.jl` (~100 lines)**

**Purpose**: Reference grid construction and validation

### **Responsibilities:**
- Reference grid builders for common use cases
- Automatic grid completion with typical values
- Cartesian product grid construction
- Grid validation and error checking

### **Key Components:**
```julia
_build_means_refgrid(data)
_build_cartesian_refgrid(at, data)
_build_explicit_refgrid(at, data)
_ensure_complete_reference_grid(reference_grid, original_data, model)
_get_typical_value(col)
```

### **Implementation Priority**: Phase 2 - Week 2 (Day 5)
- User convenience features
- Grid construction logic

---

## 📄 **File 11: `src/profile/contrasts.jl` (~100 lines)**

**Purpose**: Novel row-specific baseline contrast computation

### **Responsibilities:**
- Row-specific baseline contrast computation (novel approach)
- Profile-level marginal effects computation
- Mixed continuous/categorical variable handling
- Gradient computation for contrasts

### **Key Components:**
```julia
_profile_effects(engine, profiles; target, backend)
_compute_row_specific_baseline_contrast(engine, profile_data, profile, var, target, backend)
_compute_contrast_gradient(engine, profile, baseline_profile, target)
```

### **Implementation Priority**: Phase 2 - Week 3 (Day 1)
- Novel statistical approach
- Core innovation of the package

---

## 🚀 **Revised Implementation Timeline**

### **Phase 1: Foundation (Week 1)**
**Day 1**: `Margins.jl`, `types.jl` - Module structure, result types, exports  
**Day 2**: `engine/core.jl` - MarginsEngine struct and construction
**Day 3**: `engine/utilities.jl` - Shared utilities and validation  
**Day 4**: `engine/caching.jl` - Performance caching
**Day 5**: Integration testing of engine components

**Validation**: Zero-allocation engine construction, basic functionality

### **Phase 2: Core Functions (Week 2)**  
**Day 1**: `population/core.jl` - Main population_margins() entry
**Day 2**: `population/contexts.jl` - at/over parameter handling
**Day 3**: `population/effects.jl` - AME computation
**Day 4**: `profile/core.jl` - Main profile_margins() entry
**Day 5**: `profile/refgrids.jl` - Reference grid builders

**Validation**: Both main functions work, basic feature completeness

### **Phase 3: Advanced Features (Week 3)**
**Day 1**: `profile/contrasts.jl` - Novel row-specific contrasts
**Day 2-3**: Integration testing, error handling across all files
**Day 4-5**: Performance optimization, allocation tracking

**Validation**: All functionality working, performance targets met

### **Phase 4: Polish (Week 4)**
**Day 1-3**: Statistical validation (bootstrap comparison)
**Day 4-5**: Documentation, final testing

**Validation**: Publication-ready statistical correctness

## 📊 **Success Metrics**

### **Performance Targets:**
- **Population margins**: <100ns per row (0 bytes allocated)  
- **Profile margins**: <1μs per profile (~400 bytes allocated)
- **Engine construction**: <10ms with caching

### **Code Quality Targets:**
- **Total lines**: ~1050 lines (vs current ~3000+)
- **Files**: 12 files in organized structure (vs current 17+ files)  
- **Exports**: <10 public functions (vs current 20+)

### **Statistical Correctness:**
- **Bootstrap validation**: All SEs within 5% of bootstrap estimates
- **Cross-validation**: AD vs FD backends agree within 1e-10
- **Error handling**: Clear, actionable error messages

This file plan provides the complete roadmap for implementing the aggressive reorganization into a maintainable subdirectory structure while maintaining statistical rigor and achieving dramatic performance improvements.