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

### **Design Decisions for Novel Row-Specific Profile Margins**

**Issue 1: Reference Grid Structure with Repeated Categorical Levels**
- **Decision**: Allow and embrace multiple rows with same categorical level
- **Rationale**: Shows how categorical effects vary across covariate contexts
- **Example**: `catvar=["B", "B"]` at different `x` values shows effect heterogeneity
- **Implementation**: No aggregation - each row evaluated independently

**Issue 2: Baseline Level Detection for Categorical Contrasts**  
- **Decision**: Extract baseline from model's contrast coding (Option 1)
- **Rationale**: Statistically principled - respects actual model contrasts
- **Implementation**: `_get_baseline_level(model, var)` extracts from `model.mf.contrasts`
- **Fallback**: Error with clear message if contrast info unavailable
- **Benefits**: Ensures contrasts match model's coefficient interpretation

**Issue 3: Reference Grid Value Validation (Age = 200, etc.)**
- **Decision**: Out of scope - do not validate plausibility of user inputs
- **Rationale**: Users know their domain and research context best
- **Implementation**: Trust user inputs, compute faithfully, fail only on computational impossibilities
- **Philosophy**: We are a statistical computation tool, not a data validator

**Issue 4: Zero Contrasts for Baseline-Level Rows**
- **Decision**: Keep zero results (baseline vs baseline = 0.0)
- **Rationale**: Transparent - every reference grid row gets evaluated as specified
- **User Control**: Users can exclude baseline rows if they don't want zeros
- **Consistency**: No hidden filtering or surprises in output

**Issue 5: Computational Efficiency of Row-Specific Approach**
- **Decision**: Current approach is optimal - no concern needed
- **Rationale**: Only computes what user requested (2 predictions per categorical row)
- **Efficiency**: Avoids traditional approach waste (computing all contrasts everywhere)
- **Performance**: Direct targeting of user's research question

**Issue 6: Multi-Level Categorical Variables (Only One Contrast Per Row)**
- **Decision**: Follow reference grid exactly - this is the design intent
- **Rationale**: User specifies exactly which levels to evaluate at which profiles
- **User Control**: Want all contrasts? Build reference grid with all desired levels
- **Philosophy**: Complete respect for user's reference grid specification

### **Design Decisions for Population Margins `at`/`over` Parameters**

**Issue 1: Computational Efficiency with Complex `over` Specifications**
- **Concern**: Many subgroup combinations (e.g., 6 ages × 4 regions × 2 genders = 48 combinations)
- **Decision**: Not a design concern - this is FormulaCompiler's core strength
- **Rationale**: Zero-allocation evaluation makes large numbers of contexts computationally trivial
- **Implementation**: Leverage FormulaCompiler's efficiency for the "whole ballgame"

**Issue 2: Small Subgroup Sizes in Continuous Variable Subgrouping**
- **Decision**: User risk - standard errors will signal the issue appropriately
- **Rationale**: Users control subgroup specification and should interpret SEs accordingly
- **Statistics**: Large SEs naturally indicate small sample concerns - this is how statistics should work
- **Philosophy**: Trust users to specify meaningful subgroups for their research context

**Issue 3: `at` Parameter Statistical Validity (Age = 200, etc.)**
- **Decision**: Out of scope - consistent with earlier profile margins decision
- **Rationale**: Same principle as reference grid validation - users know their domain
- **Implementation**: Compute faithfully, don't validate plausibility of counterfactual scenarios

**Issue 4: Complex `at`/`over` Interaction Interpretability**
- **Decision**: User education - behavior is well-defined and documented
- **Rationale**: Advanced users creating complex specifications should understand the results
- **Documentation**: Clear examples and data flow diagrams explain the behavior

**Issue 5: Memory Usage with Large Context Combinations**
- **Decision**: Manageable with zero-allocation patterns
- **Rationale**: Efficient data handling and FormulaCompiler integration minimize memory concerns
- **Implementation**: Use views and zero-allocation evaluation where possible

**Overall Population Margins Design Status: ✅ READY FOR IMPLEMENTATION**
- Stata-compatible `at`/`over` semantics
- Efficient leveraging of FormulaCompiler capabilities
- User-controlled with appropriate statistical error signaling  
- Well-documented with clear behavioral specifications

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

**📋 Complete Implementation Blueprint**: See **[FILE_PLAN.md](FILE_PLAN.md)** for detailed function specifications, exact code examples, implementation timeline, and success metrics for the organized file architecture.

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
   function _validate_variables(data_nt::NamedTuple, vars, continuous_vars)
       for var in vars
           haskey(data_nt, var) || error("Variable $var not found in data")
           # Both continuous and categorical variables supported
           # Auto-detection handles the dispatch - no warnings needed
       end
   end
   ```

2. **Zero-Allocation Population Effects (AME)** with graceful fallbacks and batch operations:
   ```julia
   function _ame_continuous_and_categorical(engine::MarginsEngine, data_nt; target=:mu, backend=:ad)
       engine.de === nothing && return (DataFrame(), Matrix{Float64}(undef, 0, length(engine.β)))
       
       rows = 1:nrows(data_nt)
       results = DataFrame(term=String[], estimate=Float64[], se=Float64[])
       G = Matrix{Float64}(undef, length(engine.de.vars), length(engine.β))
       
       # Auto-detect variable types and process accordingly
       continuous_vars = FormulaCompiler.continuous_variables(engine.compiled, data_nt)
       
       # Process continuous variables with FC's built-in AME gradient accumulation (ZERO ALLOCATION!)
       cont_idx = 1
       for var in engine.de.vars
           if var ∈ continuous_vars
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
                   ame_val += engine.g_buf[cont_idx]
               end
               ame_val /= length(rows)
               
               push!(results, (term=string(var), estimate=ame_val, se=se))
               G[cont_idx, :] = gβ_avg
               cont_idx += 1
               
           else  # Categorical variable
               # Compute traditional baseline contrasts for population margins
               ame_val, gβ_avg = _compute_categorical_baseline_ame(engine, var, rows, target, backend)
               se = FormulaCompiler.delta_method_se(gβ_avg, engine.Σ)
               
               push!(results, (term="$(var) (baseline contrast)", estimate=ame_val, se=se))
               G[cont_idx, :] = gβ_avg  # Still increment for output indexing
               cont_idx += 1
           end
       end
       
       return (results, G)
   end
   ```

3. **Profile Effects (MEM) Using Reference Grids** (following FormulaCompiler guide):
   ```julia
   function _mem_continuous_and_categorical(engine::MarginsEngine, profiles; target=:mu, backend=:ad)
       engine.de === nothing && return (DataFrame(), Matrix{Float64}(undef, 0, length(engine.β)))
       
       results = DataFrame(term=String[], estimate=Float64[], se=Float64[])
       n_profiles = length(profiles)
       n_vars = length(engine.de.vars)
       G = Matrix{Float64}(undef, n_profiles * n_vars, length(engine.β))
       
       # Auto-detect variable types
       continuous_vars = FormulaCompiler.continuous_variables(engine.compiled, engine.data_nt)
       
       row_idx = 1
       for profile in profiles
           # Build minimal synthetic reference grid data (FormulaCompiler guide recommendation)
           refgrid_data = _build_refgrid_data(profile, engine.data_nt)
           refgrid_compiled = FormulaCompiler.compile_formula(engine.compiled.model, refgrid_data)
           refgrid_de = FormulaCompiler.build_derivative_evaluator(refgrid_compiled, refgrid_data; 
                                                                  vars=engine.de.vars)
           
           for (var_idx, var) in enumerate(engine.de.vars)
               if var ∈ continuous_vars
                   # Continuous variable: compute derivative
                   if target === :mu
                       FormulaCompiler.marginal_effects_mu!(engine.g_buf, refgrid_de, engine.β, 1;
                                                           link=engine.link, backend=backend)
                   else
                       FormulaCompiler.marginal_effects_eta!(engine.g_buf, refgrid_de, engine.β, 1;
                                                            backend=backend)
                   end
                   effect_val = engine.g_buf[var_idx]
               else
                   # Categorical variable: compute row-specific baseline contrast
                   # Novel approach: contrast this row's category vs baseline at this profile
                   effect_val = _compute_row_specific_baseline_contrast(engine, refgrid_de, profile, var, target, backend)
               end
               
               # Compute gradient for SE at reference point
               if var ∈ continuous_vars
                   if target === :mu
                       FormulaCompiler.me_mu_grad_beta!(engine.gβ_accumulator, refgrid_de, engine.β, 1, var;
                                                       link=engine.link)
                   else
                       FormulaCompiler.me_eta_grad_beta!(engine.gβ_accumulator, refgrid_de, engine.β, 1, var)
                   end
               else
                   # Compute gradient for row-specific categorical contrast
                   _row_specific_contrast_grad_beta!(engine.gβ_accumulator, engine, refgrid_de, profile, var, target)
               end
               se = FormulaCompiler.delta_method_se(engine.gβ_accumulator, engine.Σ)
               
               # Build profile description
               profile_desc = join(["$(k)=$(v)" for (k,v) in pairs(profile)], ", ")
               if var ∈ continuous_vars
                   term_name = "$(var) at $(profile_desc)"
               else
                   # Show the specific contrast being computed
                   current_level = profile[var]
                   baseline_level = _get_baseline_level(engine, var)
                   term_name = "$(var)=$(current_level) vs $(baseline_level) at $(profile_desc)"
               end
               
               push!(results, (term=term_name, estimate=effect_val, se=se))
               G[row_idx, :] = engine.gβ_accumulator
               row_idx += 1
           end
       end
       
       return (results, G)
   end
   
   # Helper: Build minimal reference grid data for row-specific contrasts
   function _build_refgrid_data(profile::Dict, original_data::NamedTuple)
       # Create minimal synthetic data with only needed variables
       refgrid = NamedTuple()
       for (var, val) in pairs(original_data)
           if haskey(profile, var)
               # Use profile value for this variable (including categorical levels)
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
   
   # Helper: Compute row-specific baseline contrast (novel approach)
   function _compute_row_specific_baseline_contrast(engine, refgrid_de, profile, var, target, backend)
       baseline_level = _get_baseline_level(engine.model, var)
       current_level = profile[var]
       
       # If this row already has baseline level, contrast is 0
       if current_level == baseline_level
           return 0.0
       end
       
       # Compute prediction at current profile
       current_pred = _predict_at_profile(engine, profile, target)
       
       # Compute prediction at baseline profile (same covariates, baseline category)
       baseline_profile = copy(profile)
       baseline_profile[var] = baseline_level
       baseline_pred = _predict_at_profile(engine, baseline_profile, target)
       
       # Return contrast
       return current_pred - baseline_pred
   end
   
   # Helper: Extract baseline level from model's contrast coding (statistically principled)
   function _get_baseline_level(model, var)
       # Extract baseline from model's contrast system
       if hasfield(typeof(model), :mf) && hasfield(typeof(model.mf), :contrasts)
           # GLM.jl style models
           if haskey(model.mf.contrasts, var)
               contrast = model.mf.contrasts[var]
               # Different contrast types have different ways to get baseline
               if contrast isa StatsModels.DummyCoding
                   return contrast.base
               elseif contrast isa StatsModels.EffectsCoding  
                   return contrast.base
               # Add other contrast types as needed
               end
           end
       end
       
       # If we can't extract from model, error with helpful message
       error("Could not determine baseline level for variable $var from model contrasts. " *
             "Please ensure the model has proper contrast coding information.")
   end
   ```

**Validation Requirements**: 
- BenchmarkTools.jl tests verify 0 bytes in hot paths
- Bootstrap validation of all standard errors
- Cross-validation AD vs FD backends

### **Phase 2: Clean 2×2 API Implementation**
*Target: Simple, Julian API that leverages the zero-allocation engine*

1. **New `src/population.jl`** with compilation caching and vars clarification:
   ```julia
   # Global cache for compiled formulas (MARGINS_GUIDE.md pattern)
   const COMPILED_CACHE = Dict{UInt64, Any}()
   
   function population_margins(model, data; type::Symbol=:effects, vars=nothing, target::Symbol=:mu, backend::Symbol=:ad, at=nothing, over=nothing, kwargs...)
       # Single data conversion (consistent format throughout)
       data_nt = Tables.columntable(data)
       
       # Handle vars parameter (only needed for type=:effects)
       if type === :effects
           vars = vars === nothing ? :all_continuous : vars  # Default to all continuous variables
       else # type === :predictions
           vars = nothing  # Not needed for predictions
       end
       
       # Build zero-allocation engine with caching
       engine = _get_or_build_engine(model, data_nt, vars)
       
       # Handle at/over parameters for population contexts
       if at !== nothing || over !== nothing
           return _population_margins_with_contexts(engine, data_nt, vars, at, over; type, target, backend, kwargs...)
       end
       
       if type === :effects
           df, G = _ame_continuous_and_categorical(engine, data_nt; target, backend, kwargs...)  # → AME (both continuous and categorical)
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
   
   # Handle population margins with at/over contexts (Stata-compatible)
   function _population_margins_with_contexts(engine, data_nt, vars, at, over; type, target, backend, kwargs...)
       results = DataFrame()
       
       # Parse specifications
       at_specs = at === nothing ? [Dict()] : _parse_at_specification(at)
       over_specs = over === nothing ? [Dict()] : _parse_over_specification(over, data_nt)
       
       # Create all combinations of contexts
       for at_spec in at_specs, over_spec in over_specs
           context_data = _create_context_data(data_nt, at_spec, over_spec)
           
           for var in vars
               # Skip if this var appears in at/over (conflict resolution)
               if haskey(at_spec, var) || haskey(over_spec, var)
                   continue
               end
               
               # Compute effect in this context
               if type === :effects
                   var_result = _compute_population_effect_in_context(engine, context_data, var, target, backend)
               else
                   var_result = _compute_population_prediction_in_context(engine, context_data, target)
               end
               
               # Add context identifiers
               for (ctx_var, ctx_val) in merge(at_spec, over_spec)
                   var_result[ctx_var] = ctx_val
               end
               
               append!(results, var_result)
           end
       end
       
       return MarginsResult(results, gradients, metadata)
   end
   
   # Parse at specification (counterfactual scenarios)
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
           error("at parameter must be a Dict specifying variable values")
       end
   end
   
   # Parse over specification (subgroup analysis)
   function _parse_over_specification(over, data_nt)
       if over isa NamedTuple
           # Enhanced flexible syntax
           contexts = []
           
           for (var, vals) in pairs(over)
               if vals === nothing
                   # Unspecified - must be categorical, use all levels
                   if _is_continuous_variable(data_nt[var])
                       error("Continuous variable $var in over() must specify values")
                   end
                   contexts = isempty(contexts) ? [{var => v} for v in unique(data_nt[var])] : 
                             [merge(ctx, Dict(var => v)) for ctx in contexts, v in unique(data_nt[var])]
               else
                   # Specified values
                   if _is_continuous_variable(data_nt[var])
                       # For continuous: create subgroups around specified values
                       subgroups = _create_continuous_subgroups(data_nt[var], vals)
                       contexts = isempty(contexts) ? [{var => sg} for sg in subgroups] :
                                 [merge(ctx, Dict(var => sg)) for ctx in contexts, sg in subgroups]
                   else
                       # For categorical: use specified subset
                       contexts = isempty(contexts) ? [{var => v} for v in vals] :
                                 [merge(ctx, Dict(var => v)) for ctx in contexts, v in vals]
                   end
               end
           end
           
           return contexts
       elseif over isa Vector
           # Simple vector syntax - all categorical
           contexts = [Dict()]
           for var in over
               if _is_continuous_variable(data_nt[var])
                   error("Continuous variable $var in over() must specify values. Use over=($var => [values])")
               end
               new_contexts = []
               for ctx in contexts, val in unique(data_nt[var])
                   push!(new_contexts, merge(ctx, Dict(var => val)))
               end
               contexts = new_contexts
           end
           return contexts
       else
           error("over parameter must be a NamedTuple or Vector")
       end
   end
   
   # Helper: Check if variable is continuous
   function _is_continuous_variable(col)
       return eltype(col) <: Real && !(eltype(col) <: Bool)
   end
   
   # Helper: Create subgroups around specified continuous values
   function _create_continuous_subgroups(col, specified_values)
       subgroups = []
       for val in specified_values
           # Create subgroup of indices within range of this value
           # Could use percentile-based bins, fixed ranges, etc.
           indices = findall(x -> abs(x - val) <= 2.5, col)  # ±2.5 units
           push!(subgroups, (center=val, indices=indices, label="$(val)±2.5"))
       end
       return subgroups
   end
   
   # Helper: Create context data (counterfactual overrides + subgroup filtering)
   function _create_context_data(data_nt, at_spec, over_spec)
       # Start with full data
       context_data = deepcopy(data_nt)
       
       # Apply counterfactual overrides (at)
       for (var, val) in at_spec
           if haskey(context_data, var)
               # Override all values with specified value
               n_rows = length(first(context_data))
               context_data = merge(context_data, NamedTuple{(var,)}((fill(val, n_rows),)))
           end
       end
       
       # Apply subgroup filtering (over)
       indices_to_keep = collect(1:length(first(context_data)))
       for (var, spec) in over_spec
           if haskey(context_data, var)
               if spec isa NamedTuple && haskey(spec, :indices)
                   # Continuous subgroup - intersect with these indices
                   indices_to_keep = intersect(indices_to_keep, spec.indices)
               else
                   # Categorical - filter by value
                   var_indices = findall(==(spec), context_data[var])
                   indices_to_keep = intersect(indices_to_keep, var_indices)
               end
           end
       end
       
       # Subset context_data to selected indices
       if length(indices_to_keep) < length(first(context_data))
           subset_data = NamedTuple()
           for (var, col) in pairs(context_data)
               subset_data = merge(subset_data, NamedTuple{(var,)}((col[indices_to_keep],)))
           end
           return subset_data
       end
       
       return context_data
   end
   ```

2. **New `src/profile.jl`** with reference grid approach:
   ```julia
   # Primary method: user provides complete reference grid (most efficient)
   function profile_margins(model, reference_grid::DataFrame; type::Symbol=:effects, vars=[:x1], target::Symbol=:mu, backend::Symbol=:ad, kwargs...)
       data_nt = Tables.columntable(reference_grid)
       engine = build_engine(model, data_nt, vars)  # Single compilation
       
       if type === :effects
           # Convert reference_grid to profiles for row-specific processing
           profiles = [Dict(pairs(row)) for row in eachrow(reference_grid)]
           df, G = _mem_continuous_and_categorical(engine, profiles; target, backend, kwargs...)  # → MEM with row-specific contrasts
           return MarginsResult(df, G, _build_metadata(...))
       else # :predictions  
           df, G = _profile_predictions(engine, data_nt; target, kwargs...)  # → APM
           return MarginsResult(df, G, _build_metadata(...))
       end
   end
   
   # Convenience method: reference grid + data (handles both partial and complete grids)
   function profile_margins(model, reference_grid::DataFrame, data::DataFrame; kwargs...)
       # Smart completion: fill missing variables with typical values if needed
       complete_reference_grid = _ensure_complete_reference_grid(reference_grid, data, model)
       return profile_margins(model, complete_reference_grid; kwargs...)
   end
   
   # Convenience methods build reference grids, then call primary method
   function profile_margins(model, data::DataFrame; at=:means, kwargs...)
       reference_grid = _build_means_refgrid(data)
       return profile_margins(model, reference_grid; kwargs...)
   end
   
   # Default: at=:means if not specified
   function profile_margins(model, data::DataFrame; kwargs...)
       return profile_margins(model, data; at=:means, kwargs...)
   end
   
   function profile_margins(model, data::DataFrame; at::Dict, kwargs...)
       reference_grid = _build_cartesian_refgrid(at, data)  # Dict(:x => [1,2], :y => [3])
       return profile_margins(model, reference_grid; kwargs...)
   end
   
   function profile_margins(model, data::DataFrame; at::Vector, kwargs...)
       reference_grid = _build_explicit_refgrid(at, data)  # [(x=1, y=3), (x=2, y=3)]
       return profile_margins(model, reference_grid; kwargs...)
   end
   ```

3. **Reference Grid Builders** (clean, efficient implementations):
   ```julia
   # Build reference grid with means for continuous, first levels for categorical
   function _build_means_refgrid(data::DataFrame)
       row = Dict{Symbol,Any}()
       for (name, col) in pairs(eachcol(data))
           if eltype(col) <: Real && !(eltype(col) <: Bool)
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
   
   # Build Cartesian product reference grid from Dict specification
   function _build_cartesian_refgrid(at::Dict, data::DataFrame)
       # Cartesian product of specified values, means for others
       base_row = _build_means_refgrid(data)[1, :]  # Get base row with means
       
       # Override with user specifications and expand
       grid_rows = []
       var_names = collect(keys(at))
       var_values = [at[k] for k in var_names]
       
       for combo in Iterators.product(var_values...)
           row = copy(base_row)
           for (i, var) in enumerate(var_names)
               row[var] = combo[i]
           end
           push!(grid_rows, row)
       end
       
       return DataFrame(grid_rows)
   end
   
   # Build reference grid from explicit vector of NamedTuples/Dicts
   function _build_explicit_refgrid(at::Vector, data::DataFrame)
       base_row = _build_means_refgrid(data)[1, :]  # Get base row with means
       
       grid_rows = []
       for profile in at
           row = copy(base_row)
           for (k, v) in pairs(profile)
               row[k] = v
           end
           push!(grid_rows, row)
       end
       
       return DataFrame(grid_rows)
   end
   
   # Helper: Ensure reference grid is complete (works for both partial and complete grids)
   function _ensure_complete_reference_grid(reference_grid::DataFrame, original_data::DataFrame, model)
       # Get all variables needed by the model
       model_vars = Set(Symbol.(StatsModels.coefnames(model.mf.f)))  # All variables in formula
       grid_vars = Set(Symbol.(names(reference_grid)))
       missing_vars = setdiff(model_vars, grid_vars)
       
       # If already complete, just return it
       if isempty(missing_vars)
           return reference_grid
       end
       
       # Otherwise, fill missing variables with typical values
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
   
   # Helper: Get typical value for a column
   function _get_typical_value(col)
       if eltype(col) <: Real && !(eltype(col) <: Bool)
           return mean(col)  # Mean for continuous
       elseif col isa CategoricalArray
           return levels(col)[1]  # First level for categorical
       elseif eltype(col) <: Bool  
           return StatsBase.mode(col)  # Most common value for Boolean
       elseif eltype(col) <: AbstractString
           return StatsBase.mode(col)  # Most common string
       else
           return first(col)  # Fallback
       end
   end
   ```

### **Phase 3: Radical File Structure Simplification**
*Target: Maintainable organized architecture with logical subdirectories*

1. **New Organized Structure**:
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
- **30% fewer** source files (17 → 12) with **better organization**
- **65% fewer** lines of code (~3000 → ~1050)
- **90% simpler** API surface (remove complex iterators and utilities)
- **Logical grouping** for easier navigation and development

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

- [x] **Phase 1 (Week 1-2)**: Zero-Allocation Engine Foundation
   - [x] **Day 1-3**: Design and implement `MarginsEngine` struct with pre-allocated buffers
   - [x] **Day 4-6**: Build zero-allocation population effects functions using FormulaCompiler properly
   - [x] **Day 7-10**: Implement zero-allocation profile effects with scenario override system
   - **Validation**: BenchmarkTools.jl tests verify 0 bytes, bootstrap validation of SEs

- [x] **Phase 2 (Week 3)**: Clean 2×2 API Implementation  
   - [x] **Day 1-2**: Implement new `population.jl` with clean Julia-style API
   - [x] **Day 3-4**: Implement new `profile.jl` with unified profile building
   - [x] **Day 5**: Single, efficient profile specification system (no duplicates)
   - [x] Fix categorical typical values to use frequency-weighted mixtures instead of first-levet default (e.g., when left unspecified in the reference grid)
   - **Validation**: All current functionality works, performance benchmarks show improvement

- [x] **Phase 3 (Week 4)**: Radical Simplification
   - [x] **Day 1-2**: Delete old file structure, move logic to new organized architecture
   - [x] **Day 3-4**: Aggressive API cleanup - remove complex exports and features
   - [x] **Day 5**: Update `Margins.jl` module with minimal, clean exports
   - **Validation**: Test suite passes, package loads cleanly

- [ ] **Phase 4 (Week 5)**: Performance Optimization & Validation
   - [ ] **Day 1-3**: Achieve target performance (<100ns per row, <1μs per profile)
   - [ ] **Day 4-5**: Comprehensive statistical validation and benchmarking
   - **Validation**: Performance targets met, statistical correctness verified

- [ ] **Phase 5 (Week 6)**: Documentation & Polish
   - [ ] **Day 1-3**: Update documentation to reflect simplified API
   - [ ] **Day 4-5**: Final testing, edge cases, error handling
   - [ ] **Validation**: Ready for production use

## 🎯 **How to Start**

**Immediate next step**: Begin with Phase 1 - implement the `MarginsEngine` struct and zero-allocation foundation. This is a clean-slate approach that will:

1. **Deliver massive performance gains** (100x+ speedup expected)
2. **Simplify the architecture** (17 files → 12 organized files)  
3. **Use FormulaCompiler properly** (scenario overrides, zero-allocation paths)
4. **Maintain statistical correctness** (bootstrap validation throughout)

The approach is designed to be:
1. **Aggressive** - no concern for breaking changes
2. **Performance-first** - zero-allocation from the start
3. **Correct** - extensive statistical validation
4. **Julian** - clean, idiomatic Julia code
5. **FormulaCompiler-native** - leverage all FC capabilities

This reorganization will transform Margins.jl from an over-engineered, slow package into a simple, blazing-fast statistical interface that properly leverages FormulaCompiler.jl's capabilities.

**📋 For Complete Implementation Details**: See **[FILE_PLAN.md](FILE_PLAN.md)** - comprehensive organized architecture specification with exact function signatures, implementation timeline, performance targets, and success metrics.