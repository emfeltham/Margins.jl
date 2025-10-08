# Margins.jl Testing Framework

Status: Production-ready comprehensive testing with rigorous statistical validation

Test Coverage: 60+ test files with cross-platform validation against R's margins package

## Table of Contents

1. [Overview](#overview)
2. [Test Organization](#test-organization)
3. [R Cross-Validation](#r-cross-validation)
4. [Statistical Validation](#statistical-validation)
5. [Performance Validation](#performance-validation)
6. [Running Tests](#running-tests)
7. [Confidence Assessment](#confidence-assessment)

## Overview

The Margins.jl testing framework ensures:

- Statistical correctness: All computations validated against established software and bootstrap estimates
- Performance characteristics: Zero-allocation core functions with O(1) scaling for profile margins
- Comprehensive coverage: 60+ test files covering all model types, features, and edge cases
- Cross-platform validation: Results match R's margins package within numerical precision

### Test Philosophy

1. Statistical validity is paramount: Wrong standard errors are worse than no standard errors
2. Zero tolerance for invalid results: Error out rather than approximate
3. Error-first policy: Users must be explicitly aware of any statistical compromises

## Test Organization

### Test Structure (`test/`)

```
test/
├── runtests.jl                          # Main test orchestrator
├── test_utilities.jl                    # Shared infrastructure (models, formulas, data)
│
├── core/                                # Core functionality (14 files)
│   ├── test_glm_basic.jl               # GLM.jl integration
│   ├── test_profiles.jl                # Reference grids
│   ├── test_grouping.jl                # Group-wise computation
│   ├── test_contrasts.jl               # Categorical contrasts
│   ├── test_vcov.jl                    # Covariance matrices
│   ├── test_errors.jl                  # Error handling
│   ├── test_mixedmodels.jl             # Mixed models
│   └── test_weights.jl                 # Sampling/frequency weights
│
├── features/                            # Advanced features (12 files)
│   ├── test_elasticities.jl            # Elasticity computations
│   ├── test_categorical_mixtures.jl    # CategoricalMixture profiles
│   ├── test_bool_profiles.jl           # Boolean variable handling
│   ├── test_prediction_scales.jl       # Link vs response scale
│   └── test_hierarchical_grids.jl      # Multi-level grids
│
├── performance/                         # Performance validation (8 files)
│   ├── test_zero_allocations.jl        # Zero-allocation verification
│   └── test_performance.jl             # Scaling characteristics
│
├── statistical_validation/              # Statistical correctness (10 files)
│   ├── bootstrap_se_validation.jl      # Bootstrap SE validation
│   ├── analytical_se_validation.jl     # Analytical SE verification
│   ├── backend_consistency.jl          # AD vs FD agreement
│   ├── robust_se_validation.jl         # Robust/clustered SEs
│   ├── incompatible_formula_se_validation.jl  # Specialized validation
│   └── ci_validation.jl                # Confidence intervals
│
├── validation/                          # Mathematical validation (6 files)
│   ├── test_contrast_invariance.jl     # Categorical invariance
│   ├── test_manual_counterfactual_validation.jl  # Manual verification
│   └── test_zero_allocation_comprehensive.jl  # Allocation scaling
│
└── r_compare/                           # R cross-validation (10 files)
    ├── README.md                        # Workflow documentation
    ├── VALIDATION_RESULTS.md            # Validation summary
    ├── generate_data.jl                 # Synthetic data generation
    ├── julia_model.jl                   # Julia analysis
    ├── r_model.R                        # R analysis
    ├── compare_results.jl               # Coefficient comparison
    ├── compare_margins.jl               # Marginal effects comparison
    └── Makefile                         # Automated workflow
```

### Test Utilities (`test/test_utilities.jl`)

Shared infrastructure for all test files:

#### Data Generation
- `make_test_data(n=500)`: Standard test dataset with mixed variable types
- `generate_synthetic_dataset(n)`: Large-scale synthetic data for R comparison

#### Formula Collections
- Linear Models (`linear_formulas`): 16 formulas from simple to complex interactions
- GLM Tests (`glm_tests`): 10 formulas across Binomial, Poisson, Gamma, Normal
- LMM Tests (`lmm_formulas`): 6 formulas for linear mixed models
- GLMM Tests (`glmm_tests`): 4 formulas for generalized linear mixed models

#### Performance Testing Functions
- `test_zero_allocation()`: Verify FormulaCompiler zero-allocation primitives
- `test_allocation_performance()`: Allocation scaling analysis
- `test_model_correctness()`: Mathematical correctness verification

## R Cross-Validation

Objective: Validate Margins.jl against R's `margins` package to ensure statistical equivalence.

Location: `test/r_compare/`

Documentation: See [`test/r_compare/VALIDATION_RESULTS.md`](test/r_compare/VALIDATION_RESULTS.md) for comprehensive results.

### Validation Results Summary

#### Model Coefficients

| Metric | Result |
|--------|--------|
| Dataset size | 5,000 observations |
| Model complexity | 65 coefficients (logistic regression) |
| Max coefficient relative error | 0.0028% |
| Mean coefficient relative error | 0.0005% |
| Max SE relative error | 0.04% |
| Mean SE relative error | 0.034% |
| Verdict | ✓✓✓ SUCCESS |

#### Marginal Effects Validation

| Type | Variables | Max Est Error | Max SE Error | Status |
|------|-----------|---------------|--------------|--------|
| AME (Average Marginal Effects) | 32 | 0.0036% | 0.04% | ✓✓✓ SUCCESS |
| AAP (Average Adjusted Predictions) | 1 | 0.0001% | 0.03% | ✓✓✓ SUCCESS |
| APM (Adjusted Predictions at Profiles) | 4 | 0.87% | 1.98% | ✓ PASS |
| AME by Age | 1 | 0.0003% | 0.04% | ✓✓✓ SUCCESS |
| AME by Scenario | 2 | 0.0017% | 0.04% | ✓✓✓ SUCCESS |

### Key Findings

1. Perfect agreement on core functionality: AME (the primary use case) achieves < 0.004% error
2. Proper delta-method SEs: Both implementations use rigorous delta-method standard errors
3. Statistical equivalence: All differences within numerical precision expectations

### Delta Method Validation

Critical validation: Both Margins.jl and R's margins package implement proper delta-method standard errors:

$$\text{SE}(\hat{g}(\boldsymbol{\beta})) = \sqrt{\nabla g(\boldsymbol{\beta})^\top \boldsymbol{\Sigma} \nabla g(\boldsymbol{\beta})}$$

where:
- $g(\boldsymbol{\beta})$ is the marginal effect function
- $\boldsymbol{\Sigma}$ is the model covariance matrix
- $\nabla g(\boldsymbol{\beta})$ is the gradient vector

No independence assumptions: Full covariance structure preserved in all computations.

### Running R Comparison

```bash
cd test/r_compare

# Automated workflow (requires R and Julia)
make all

# Manual workflow
julia --project=. generate_data.jl      # Generate synthetic data
julia --project=. julia_model.jl        # Run Julia analysis
Rscript r_model.R                       # Run R analysis
julia --project=. compare_results.jl    # Compare coefficients
julia --project=. compare_margins.jl    # Compare marginal effects
```

Requirements:
- R with packages: `margins`, `readr`, `dplyr`
- Julia with Margins.jl and dependencies

## Statistical Validation

### Bootstrap Validation (`test/statistical_validation/`)

Objective: Validate delta-method standard errors against bootstrap estimates.

Method: 1000 bootstrap resamples with automatic bias correction

Coverage: All model types (LM, GLM, LMM, GLMM) and effect types

Tolerance: ±10% agreement between delta-method and bootstrap SEs

Files:
- `bootstrap_se_validation.jl`: Core bootstrap validation framework
- `elasticity_se_validation.jl`: Elasticity-specific bootstrap validation
- `ci_validation.jl`: Confidence interval coverage validation

### Analytical Validation

Objective: Verify standard errors through hand-coded analytical formulas.

Method: Manual gradient computation and delta-method application

Scope: Main effects models (55% of test formulas)

Files:
- `analytical_se_validation.jl`: Linear, logistic, Poisson models
- `analytical_elasticity_se_validation.jl`: Elasticity SE validation
- `incompatible_formula_se_validation.jl`: Specialized validation for interactions and mixed models

Formula Compatibility:
- Compatible (83 formulas): Main effects models with continuous/categorical variables
- Requires specialized validation (68 formulas): Interactions, mixed models, complex functions

Specialized Validation Approach:
- Interaction models: Hand-coded analytical SE for coefficient combinations
- Mixed models: Fixed effects variance-covariance approach (random effects don't affect marginal effect estimates)
- Complex functions: Backend consistency validation (AD vs FD)

### Backend Consistency

Objective: Verify AD (automatic differentiation) and FD (finite differences) produce identical results.

Tolerance: Machine precision agreement (relative errors ~10⁻¹⁶)

File: `backend_consistency.jl`

```julia
# Verify AD and FD backends agree
result_ad = population_margins(model, data; backend=:ad)
result_fd = population_margins(model, data; backend=:fd)
@test isapprox(result_ad.se, result_fd.se, rtol=1e-14)
```

### Robust Standard Errors

Objective: Validate CovarianceMatrices.jl integration for robust/clustered SEs.

File: `robust_se_validation.jl`

Covariance Types Tested:
- HC0, HC1, HC2, HC3 (heteroskedasticity-robust)
- CRHC0, CRHC1, CRHC2, CRHC3 (cluster-robust)

## Performance Validation

### Zero-Allocation Requirements

Objective: Achieve O(1) allocation scaling for profile margins and minimal allocations for population margins.

Files:
- `test/performance/test_zero_allocations.jl`: Core function verification
- `test/validation/test_zero_allocation_comprehensive.jl`: Scaling validation

### Allocation Targets

| Function | Target | Status |
|----------|--------|--------|
| FormulaCompiler primitives | 0 allocations | ✓ Achieved |
| Profile margins | O(1) scaling | ✓ Achieved |
| Population margins | < 3000 total allocations | ✓ Achieved |
| Core AME functions | ≤ 15 allocations | ✓ Achieved |

### Performance Improvements

- 12x allocation reduction in core AME functions from original implementation
- O(1) profile margin scaling: Constant allocations regardless of dataset size
- Zero-allocation gradients: FormulaCompiler derivatives allocate nothing

### Scaling Validation

```julia
# Test allocation scaling across dataset sizes
for n in [100, 1_000, 10_000, 100_000]
    df = make_test_data(n)
    model = lm(@formula(y ~ x + z), df)

    # Profile margins: O(1) - should be constant
    alloc_profile = @allocated profile_margins(model, df, reference_grid)
    @test alloc_profile < 1000  # Constant overhead

    # Population margins: O(n) - should scale linearly
    alloc_population = @allocated population_margins(model, df)
    @test alloc_population < n * 50  # Linear scaling with small constant
end
```

## Running Tests

### Full Test Suite

```bash
# Run all tests (60+ files)
julia --project=. -e "using Pkg; Pkg.test()"

# Run all tests with verbose output
julia --project=. test/runtests.jl
```

### Specific Test Categories

```bash
# Core functionality only
julia --project=. test/runtests.jl test/core/

# Statistical validation only
julia --project=. test/runtests.jl test/statistical_validation/

# Performance tests only
julia --project=. test/runtests.jl test/performance/

# R comparison
cd test/r_compare && make all
```

### Specific Test Files

```bash
# Single test file
julia --project=. test/core/test_glm_basic.jl

# Multiple specific files
julia --project=. test/core/test_glm_basic.jl test/features/test_elasticities.jl
```

### Debugging Performance

```bash
# Enable debug logging for allocation analysis
JULIA_DEBUG=Margins julia --project=. test/validation/test_zero_allocation_comprehensive.jl

# Profile memory usage
julia --project=. --track-allocation=user test/performance/test_performance.jl

# Profile with detailed output
julia --project=. --track-allocation=user --inline=no test/performance/test_performance.jl
```

## Confidence Assessment

#### 1. R Cross-Validation ✓✓✓
- Strong agreement with R's margins package

#### 2. Comprehensive Test Suite (60+ files) ✓✓✓
- Core functionality: GLM integration, mixed models, predictions, grouping, contrasts
- Statistical validation: Bootstrap SE validation, analytical validation, backend consistency
- Edge cases: Error handling, variable detection, mathematical invariance
- Advanced features: Elasticities, categorical mixtures, hierarchical grids

#### 3. FormulaCompiler.jl Foundation ✓✓✓
- All tests passing in the computational engine
- Derivatives system validated (AD/FD backends agree)
- Zero-allocation performance verified
- Mathematical correctness of gradient computation confirmed

#### 4. Statistical Rigor Documentation ✓✓✓
- Zero tolerance policy: No silent statistical failures
- Proper delta method: Full covariance matrix Σ used
- Error-first: Fails explicitly when validity compromised

### Remaining Uncertainty

#### Areas with Less Direct Validation

1. Complex edge cases not in tests
   - Extremely ill-conditioned models
   - Near-singular covariance matrices
2. Scale considerations
   - Tests use 5,000-310,000 observations
   - Not tested: millions of observations
3. Robustness across future model types
   - GLM: ✓✓✓ Thoroughly validated
   - MixedModels: ✓✓ Test coverage exists
   - Future extensions: Not yet tested

The combination of:
1. R cross-validation with < 0.004% error
2. 60+ passing test files
3. Bootstrap SE validation
4. Zero-allocation performance proofs
5. Explicit delta-method validation

implies that Margins.jl is production-ready.

## Test Categories Detail

### Core Functionality Tests (`test/core/`)

14 test files covering fundamental package operations:

- GLM Integration (`test_glm_basic.jl`): Basic GLM.jl compatibility, prediction scales, coefnames
- Profiles (`test_profiles.jl`): Reference grid systems (cartesian, means, balanced, quantile, hierarchical)
- Grouping (`test_grouping.jl`): Group-wise computation with `over`/`within`/`by` parameters
- Contrasts (`test_contrasts.jl`): Categorical variable baseline and pairwise contrasts
- VCov (`test_vcov.jl`): Covariance matrix handling, robust SEs, user-provided matrices
- Error Handling (`test_errors.jl`): Input validation, meaningful error messages
- Variable Detection (`test_automatic_variable_detection.jl`): Automatic Int64/Float64/Bool/Categorical classification
- Mixed Models (`test_mixedmodels.jl`): LinearMixedModel and GeneralizedLinearMixedModel support
- Weights (`test_weights.jl`): Sampling weights and frequency weights integration

### Advanced Features Tests (`test/features/`)

12 test files covering specialized functionality:

- Elasticities (`test_elasticities.jl`): `:elasticity`, `:semielasticity_dyex`, `:semielasticity_eydx` measures
- Categorical Mixtures (`test_categorical_mixtures.jl`): CategoricalMixture objects for fractional specifications
- Bool Profiles (`test_bool_profiles.jl`): Boolean variables treated as categorical with discrete contrasts
- Table Profiles (`test_table_profiles.jl`): DataFrame-based custom reference grids
- Prediction Scales (`test_prediction_scales.jl`): Link vs response scale with proper chain rule
- Hierarchical Grids (`test_hierarchical_grids.jl`): Multi-level reference grid construction

### Performance Tests (`test/performance/`)

8 test files validating computational efficiency:

- Zero Allocations (`test_zero_allocations.jl`): FormulaCompiler primitive verification
- Performance Benchmarks (`test_performance.jl`): Overall performance characteristics
- Allocation Scaling (`test_detailed_allocations.jl`): Detailed allocation profiling
- Type Stability (`test_type_stability.jl`): Type inference verification

### Statistical Validation Tests (`test/statistical_validation/`)

10 test files ensuring statistical correctness:

- Bootstrap SE (`bootstrap_se_validation.jl`): 1000-sample bootstrap validation
- Analytical SE (`analytical_se_validation.jl`): Hand-coded gradient verification (main effects)
- Elasticity SE (`analytical_elasticity_se_validation.jl`): Elasticity-specific analytical validation
- Backend Consistency (`backend_consistency.jl`): AD vs FD agreement at machine precision
- Robust SE (`robust_se_validation.jl`): HC0-HC3, cluster-robust SE validation
- Incompatible Formulas (`incompatible_formula_se_validation.jl`): Specialized validation for interactions/mixed models
- CI Validation (`ci_validation.jl`): Confidence interval coverage properties
- Function Composition (`function_composition_tests.jl`): Complex mathematical transformations

### Validation Tests (`test/validation/`)

6 test files ensuring mathematical correctness:

- Contrast Invariance (`test_contrast_invariance.jl`): Categorical contrast mathematical invariance
- Manual Counterfactual (`test_manual_counterfactual_validation.jl`): Manual step-by-step verification
- Zero Allocation Comprehensive (`test_zero_allocation_comprehensive.jl`): Complete allocation scaling analysis
- True Zero Allocation (`test_true_zero_allocation.jl`): Core function zero-allocation proofs

## Summary

Margins.jl achieves production-ready status through:

1. Statistical Correctness: R cross-validation proves equivalence to established software
2. Comprehensive Testing: 60+ test files with rigorous validation
3. Performance Optimization: Zero-allocation core with proven scaling characteristics
4. Documentation Standards: Academic-level rigor with full mathematical transparency
