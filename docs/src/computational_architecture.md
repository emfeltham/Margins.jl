# Computational Architecture

*The foundational computational engine powering Margins.jl*

## FormulaCompiler.jl: The Foundation

Margins.jl is built on [FormulaCompiler.jl](https://github.com/emfeltham/FormulaCompiler.jl), a high-performance formula evaluation and differentiation engine specifically designed for econometric analysis. This architectural foundation helps explain how Margins.jl achieves both statistical correctness and exceptional performance.

### Why FormulaCompiler.jl Matters

FormulaCompiler.jl provides the **zero-allocation computational core** that enables Margins.jl to:

1. **Process large econometric datasets efficiently**: O(1) profile margins regardless of dataset size
2. **Maintain statistical rigor**: Exact derivatives for delta-method standard errors  
3. **Support complex formulas**: Nested functions like `log(1 + income)` with proper differentiation
4. **Handle mixed data types**: Automatic Float64 conversion for derivatives without runtime cost
5. **Ensure numerical stability**: Machine-precision accuracy for gradient computations

**Without FormulaCompiler.jl**, marginal effects computation would require either:
- Slow symbolic differentiation (intractable for large datasets)
- Unreliable numerical approximations (compromising statistical validity)
- Massive memory allocations (preventing large-scale analysis)

### The Compilation Strategy

FormulaCompiler.jl transforms regression formulas into highly optimized computational kernels:

#### Formula Compilation
```julia
# From StatsModels.jl formula...
@formula(y ~ log(income) + age + education)

# ...to zero-allocation evaluator
compiled = FormulaCompiler.compile_formula(model, data)
# Single compilation, reused across all margin computations
```

#### Derivative System
```julia
# Automatic derivative evaluators for marginal effects
de = FormulaCompiler.build_derivative_evaluator(compiled, data; vars=[:income, :age])
# Zero allocation per derivative computation
```

#### Type-Safe Overrides
```julia
# Efficient scenario analysis with fractional specifications
overrides = Dict(:income => 50000, :treatment => 0.5)  # 50% treatment probability
result = FormulaCompiler.evaluate_scenario(compiled, overrides)
# Supports categorical mixtures and continuous overrides seamlessly
```

## Computational Kernels

### Zero-Allocation Formula Evaluation

The core of all marginal effects computation is formula evaluation. FormulaCompiler.jl achieves **~7 nanoseconds per evaluation** with zero allocations:

```julia
# Population margins: evaluate formula n times (once per observation)
for i in 1:n_observations
    η[i] = compiled_evaluator(data_row[i])  # 7ns, 0 bytes
end

# Profile margins: evaluate formula k times (once per scenario)  
for j in 1:n_scenarios
    η[j] = compiled_evaluator(scenario[j])   # 7ns, 0 bytes, independent of n_observations
end
```

**Key insight**: Profile margins achieve **O(1) scaling** because they evaluate formulas only at specified scenarios, not across the entire dataset.

### Derivative Computation

Marginal effects require computing ∂η/∂x for each variable. FormulaCompiler.jl provides two backends:

#### Automatic Differentiation (`:ad`) - **RECOMMENDED**
- **Accuracy**: Machine precision (exact derivatives)
- **Allocation**: Zero bytes after warmup
- **Domain safety**: Handles log(), sqrt(), 1/x functions correctly
- **Use case**: Default choice for reliability and accuracy

#### Finite Differences (`:fd`)
- **Accuracy**: Numerical approximation (typically sufficient)
- **Allocation**: Zero bytes in all cases
- **Performance**: Slightly faster for simple formulas
- **Use case**: Production optimization when domain is well-behaved

### Buffer Management System

Margins.jl pre-allocates computational buffers to achieve zero-allocation performance:

```julia
# Pre-allocated in MarginsEngine struct
struct MarginsEngine
    η_buf::Vector{Float64}          # Linear predictor evaluations
    g_buf::Vector{Float64}          # Gradient computations  
    gβ_accumulator::Vector{Float64} # Parameter gradient accumulation
    # ... other buffers
end
```

These buffers are **reused across all computations**, eliminating runtime allocations while maintaining thread safety.

## Data Type Architecture

### Mixed Type Handling

FormulaCompiler.jl automatically handles the mixed data types common in econometric analysis:

#### Integer Variables
- **Runtime behavior**: Automatic Float64 conversion during derivative computation
- **Performance impact**: Zero (conversion happens during compilation, not evaluation)
- **Mathematical correctness**: Preserves exact derivative computation
- **Example**: `age::Int64` treated as continuous for marginal effects

#### Categorical Variables  
- **Bool variables**: Treated as categorical with fractional override support
- **CategoricalArray**: Supports both baseline and pairwise contrasts
- **Frequency weighting**: Unspecified categoricals use sample composition
- **Example**: `treatment::Bool` supports `0.7` for 70% treatment probability

#### Continuous Variables
- **Float64**: Native support with full arithmetic operations
- **Complex expressions**: `log(1 + income)`, `sqrt(age)` handled correctly
- **Chain rule**: Automatic differentiation through nested functions

### Type-Safe Scenario System

FormulaCompiler.jl enables sophisticated scenario analysis while maintaining type safety:

```julia
# Representative scenarios with mixed types
scenarios = (
    :income => [30000, 50000, 80000],        # Continuous override
    :education => ["High School", "College"], # Categorical override  
    :treatment => [0.2, 0.8]                 # Fractional Bool override
)

# Automatic Cartesian product: 3×2×2 = 12 scenarios
# Each scenario maintains type consistency and statistical validity
```

## Statistical Computation Architecture

### Delta-Method Standard Errors

The statistical rigor of Margins.jl depends on proper delta-method computation:

```julia
# Delta-method formula: Var(g(β)) = g'(β) Σ g'(β)ᵀ
# Where g'(β) = ∂(marginal_effect)/∂β and Σ = vcov(model)

# FormulaCompiler.jl computes g'(β) with zero allocation:
gradient = FormulaCompiler.compute_parameter_gradient(compiled, β, data_point)
variance = gradient' * vcov_matrix * gradient
standard_error = sqrt(variance)
```

**Critical**: This computation requires **exact derivatives** to ensure statistical validity. Approximate gradients would compromise the mathematical foundation of inference.

### Covariance Matrix Integration

FormulaCompiler.jl integrates seamlessly with Julia's covariance matrix ecosystem:

- **GLM.jl**: Uses `vcov(model)` automatically
- **CovarianceMatrices.jl**: Supports robust/clustered standard errors
- **MixedModels.jl**: Compatible with mixed model covariance structures
- **Custom matrices**: Accepts user-provided covariance matrices

## Performance Implications of Architecture

### Why Profile Margins Are O(1)

```julia
# Profile margins evaluate k scenarios (typically 1-50)
n_scenarios = length(expand_scenarios(at_specification))
computational_cost = n_scenarios * 7ns  # Independent of dataset size

# Population margins evaluate n observations  
computational_cost = n_observations * 7ns  # Scales with data
```

**Architectural insight**: Profile margins achieve constant-time performance because FormulaCompiler.jl **decouples** formula evaluation from data size.

### Memory Architecture

```julia
# Constant memory footprint regardless of dataset size:
memory_usage = sizeof(η_buf) + sizeof(g_buf) + sizeof(gβ_accumulator) + compilation_cache
# Total: ~few KB, independent of whether you have 1k or 1M observations
```

### Compilation Caching

FormulaCompiler.jl automatically caches compiled evaluators:

```julia
# First call: compilation cost
result1 = population_margins(model, data)  # ~milliseconds (compile + compute)

# Subsequent calls: pure computation  
result2 = profile_margins(model, data; at=:means)  # ~microseconds (reuse compilation)
```

## Integration with JuliaStats Ecosystem

### StatsModels.jl Integration

FormulaCompiler.jl directly processes StatsModels.jl formulas:

```julia
# From StatsModels formula specification...
formula = @formula(log_wage ~ education + experience + education&experience)

# ...to compiled computational kernel with proper derivatives
compiled = FormulaCompiler.compile_formula(formula, model, data)
# Handles interaction terms, transformations, and categorical expansions
```

### GLM.jl Integration

Link functions are handled transparently:

```julia
# For GLMs, chain rule automatically applied:
# ∂μ/∂x = (∂μ/∂η) × (∂η/∂x)
# Where μ = linkinv(η) and ∂μ/∂η computed by FormulaCompiler.jl

# Both link scale (:eta) and response scale (:mu) supported
margin_eta = compute_margin(compiled, :eta)  # Direct derivative  
margin_mu = compute_margin(compiled, :mu)    # Chain rule applied
```

### MixedModels.jl Integration

Mixed models require special covariance matrix handling:

```julia
# FormulaCompiler.jl extracts fixed effects for differentiation:
β_fixed = fixef(mixed_model)
V_fixed = vcov(mixed_model)  # Fixed effects covariance only

# Marginal effects computed relative to fixed effects:
# Random effects treated as integrated out (conditional on data)
```

## Extensibility Architecture

### Custom Function Support

FormulaCompiler.jl supports user-defined functions with automatic differentiation:

```julia
# Custom transformations with exact derivatives
my_transform(x) = log(1 + exp(x))  # Softplus function

# Automatic differentiation handles custom functions:
@formula(y ~ my_transform(income) + age)  # Works seamlessly
```

### Backend Extensibility

The architecture supports additional computational backends:

```julia
# Current backends
population_margins(model, data; backend=:ad)  # Automatic differentiation
population_margins(model, data; backend=:fd)  # Finite differences

# Future extensibility:  
# population_margins(model, data; backend=:symbolic)  # Symbolic differentiation
# population_margins(model, data; backend=:gpu)      # GPU acceleration
```

## Architectural Principles

### 1. Separation of Concerns

- **FormulaCompiler.jl**: Low-level computational primitives
- **Margins.jl**: High-level statistical interface and methodology
- **Result**: Clean abstraction boundaries and maintainable code

### 2. Performance Without Compromise

- **Statistical integrity**: Performance optimizations maintain statistical validity
- **Exact computation**: Delta-method standard errors use exact derivatives
- **Memory efficiency**: Zero-allocation core with pre-allocated buffers

### 3. Type Safety and Correctness

- **Compile-time checks**: Type errors caught during formula compilation
- **Runtime safety**: Automatic type conversions preserve mathematical properties  
- **Statistical validity**: Architecture enforces proper delta-method computation

### 4. JuliaStats Ecosystem Compatibility

- **Protocol adherence**: Follows established conventions (vcov, predict, etc.)
- **Seamless integration**: Works with existing model types and data formats
- **Future compatibility**: Architecture supports ecosystem evolution

---

*This computational architecture enables Margins.jl to deliver both statistical rigor and exceptional performance for econometric analysis. For performance-specific guidance, see [Performance Guide](performance.md). For the mathematical foundation, see [Mathematical Foundation](mathematical_foundation.md).*