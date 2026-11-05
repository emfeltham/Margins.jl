# Computational Architecture

*The foundational computational engine powering Margins.jl*

## FormulaCompiler.jl: The Foundation

Margins.jl is built on [FormulaCompiler.jl](https://github.com/emfeltham/FormulaCompiler.jl), a high-performance formula evaluation and differentiation engine specifically designed for econometric analysis. This architectural foundation helps explain how Margins.jl achieves both statistical correctness and exceptional performance.

### Why FormulaCompiler.jl Matters

FormulaCompiler.jl provides the **zero-allocation computational core** that enables Margins.jl to:

1. **Process large econometric datasets efficiently**: allocation-free row evaluation whose per-row cost does not depend on the dataset size; profile margins evaluate only the grid rows
2. **Maintain statistical rigor**: Exact derivatives for delta-method standard errors  
3. **Support complex formulas**: Nested functions like `log(1 + income)` with proper differentiation
4. **Handle mixed data types**: Integer columns are converted to Float64 once per call (about 8 bytes per row per column)
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
de = FormulaCompiler.derivativeevaluator(:ad, compiled, data, [:income, :age])
# Zero allocation per derivative computation
```

#### Type-Safe Overrides
```julia
# Counterfactual values replace a column in the evaluation (that column is rebuilt with the new
# value; the rest of the data is not copied)
population_margins(model, data; scenarios=(income=50000,))
# A categorical variable can be held at a frequency mixture in a reference grid
profile_margins(model, data, cartesian_grid(treatment=[mix("control" => 0.5, "treated" => 0.5)]))
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

**Key insight**: Profile margins evaluate formulas only at the reference-grid rows, not across the entire dataset, so their cost is dominated by the grid size. Each call still does a small amount of O(n) work on the data (see [Performance Implications of Architecture](#Performance-Implications-of-Architecture)).

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
    me_ws::MEWorkspace              # Continuous-effects accumulator workspace
    g_buf::Vector{Float64}          # Gradient computations  
    gβ_accumulator::Vector{Float64} # Parameter gradient accumulation
    # ... other buffers
end
```

These buffers are **reused across the rows of a call**, eliminating per-row allocations. Each call builds its own engine (only the compiled formula is cached, by FormulaCompiler), and every engine evaluates through its own copy of the compiled formula's scratch buffer. Concurrent Margins calls from several tasks or threads are therefore safe: no two calls share an engine (see [Caching: Thread Safety](caching.md#Thread-Safety)).

## Data Type Architecture

### Mixed Type Handling

FormulaCompiler.jl automatically handles the mixed data types common in econometric analysis:

#### Integer Variables
- **Runtime behavior**: Margins copies integer columns to Float64 at the start of each call
- **Performance impact**: One O(n) copy per call (about 8 bytes per row per integer column); evaluation itself is unaffected
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

# Margins accumulates g'(β) from FormulaCompiler's model rows and Jacobians
# (derivative_modelrow!) with zero allocation per row; then
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

### Profile and Population Cost

```julia
# Profile margins evaluate the k grid rows (typically 1-50)
computational_cost = k * cost_per_row          # the model evaluations
# plus a small per-call O(n) part: integer columns copied to Float64, the derived-column
# check, and typical values over the estimation sample for variables the grid leaves out

# Population margins evaluate n observations
computational_cost = n * cost_per_row          # scales with data
```

FormulaCompiler.jl's per-row cost does not depend on the dataset size, so profile margins are dominated by the grid size. Measured on a 65-coefficient logit (Apple M4 Pro), profile predictions at 4 grid rows take 0.27, 0.59 and 1.8 ms at 5,000, 50,000 and 500,000 rows, and the AME of all variables 18 ms, 182 ms and 1.8 s.

### Memory Architecture

The row loops allocate nothing: the engine's buffers (`g_buf`, `gβ_accumulator`, the `MEWorkspace`) are sized by the number of coefficients and variables, not by n. A call's allocations still include O(n) column copies made per call: integer columns converted to Float64 (about 8 bytes per row per column), the columns a scenario replaces, and, for FixedEffectModels, the subset to the estimation sample. In the measurements above, profile predictions allocate 0.78, 3.2 and 27 MiB and the AME of all variables 0.95, 3.4 and 27 MiB at 5,000, 50,000 and 500,000 rows.

### Compilation Caching

FormulaCompiler.jl automatically caches compiled evaluators:

```julia
# First call: compilation cost
result1 = population_margins(model, data)  # seconds in a fresh session (JIT compilation + compute)

# Subsequent calls reuse the compiled formula; the engine (buffers, evaluators, Σ) is
# rebuilt per call, a fraction of a millisecond
result2 = profile_margins(model, data, means_grid(data))
```

## Integration with JuliaStats Ecosystem

### StatsModels.jl Integration

FormulaCompiler.jl directly processes StatsModels.jl formulas:

```julia
# From StatsModels formula specification...
formula = @formula(log_wage ~ education + experience + education&experience)

# ...to compiled computational kernel with proper derivatives
compiled = FormulaCompiler.compile_formula(model, Tables.columntable(data))
# Handles interaction terms, transformations, and categorical expansions
```

### GLM.jl Integration

Link functions are handled transparently:

```julia
# For GLMs, chain rule automatically applied:
# ∂μ/∂x = (∂μ/∂η) × (∂η/∂x)
# Where μ = linkinv(η) and ∂μ/∂η computed by FormulaCompiler.jl

# Both link scale (:link) and response scale (:response) supported
margin_link = population_margins(model, data; scale=:link)          # Direct derivative
margin_response = population_margins(model, data; scale=:response)  # Chain rule applied
```

### MixedModels.jl Integration

Mixed models require special covariance matrix handling:

```julia
# FormulaCompiler.jl extracts fixed effects for differentiation:
β_fixed = fixef(mixed_model)
V_fixed = vcov(mixed_model)  # Fixed effects covariance only

# Random-effects terms are dropped before compilation, so by default
# (random_effects = :zero) predictions and marginal effects set every random
# effect to zero (a typical group, not the population average under a nonlinear
# link). random_effects = :conditional appends the random effects to the
# compiled formula as extra coefficients (β, b̂) with Henderson's joint
# covariance; :marginal averages the inverse link over N(0, z'Σ̂z) by
# Gauss–Hermite quadrature. See Mixed Models (mixed_models.md).
```

## Extensibility Architecture

### Function Support

FormulaCompiler.jl differentiates the functions it has operations for (`log`, `exp`, `sqrt`, powers, and the others listed in its user guide under "Supported Functions"). A formula that calls any other function, including a user-defined one, fails in `compile_formula` with an `ArgumentError` naming the term:

```julia
my_transform(x) = log(1 + exp(x))  # Softplus function
@formula(y ~ my_transform(income) + age)  # ArgumentError: unsupported function
# Precompute the column instead (it is then held fixed by counterfactuals; see
# "Derived Columns" in Population Scenarios), or write it with supported functions:
@formula(y ~ log(1 + exp(income)) + age)
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
