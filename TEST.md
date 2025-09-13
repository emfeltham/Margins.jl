# Margins.jl Testing Framework

**Current Status**: Production-ready comprehensive testing suite with 60+ test files and statistical validation framework.

**Test utilities**: `test/test_utilities.jl` provides shared functions, models, and formulas across all test files.

**Test structure**: Formal `@test`/`@testset` tests with `@debug` logging for performance analysis.

## Overview

The Margins.jl testing framework ensures statistical correctness, performance characteristics, and comprehensive coverage across different model types and dataset sizes. The framework includes core functionality tests, performance validation, statistical correctness validation, and cross-platform validation against R's marginaleffects package.

## Testing Categories

### 1. **Core Functionality** (`test/core/`)
- **GLM Integration** (`test_glm_basic.jl`): Basic GLM.jl compatibility and marginal effects computation
- **Profiles** (`test_profiles.jl`): Reference grid systems and profile marginal effects  
- **Grouping** (`test_grouping.jl`): Group-wise computation with over/within/by parameters
- **Contrasts** (`test_contrasts.jl`): Categorical variable contrast computation
- **VCov** (`test_vcov.jl`): Covariance matrix handling and robust standard errors
- **Error Handling** (`test_errors.jl`): Input validation and error conditions
- **Variable Detection** (`test_automatic_variable_detection.jl`): Automatic variable type detection
- **Mixed Models** (`test_mixedmodels.jl`): LinearMixedModel and GeneralizedLinearMixedModel support
- **Weights** (`test_weights.jl`): Sampling weights and frequency weights

### 2. **Advanced Features** (`test/features/`)
- **Elasticities** (`test_elasticities.jl`): Elasticity and semi-elasticity computations
- **Categorical Mixtures** (`test_categorical_mixtures.jl`): CategoricalMixture objects in reference grids
- **Bool Profiles** (`test_bool_profiles.jl`): Boolean variable handling as categorical
- **Table Profiles** (`test_table_profiles.jl`): DataFrame-based reference grids
- **Prediction Scales** (`test_prediction_scales.jl`): Link vs response scale predictions
- **Hierarchical Grids** (`test_hierarchical_grids.jl`): Multi-level reference grid construction

### 3. **Performance Validation** (`test/performance/`)
- **Zero Allocations** (`test_zero_allocations.jl`): FormulaCompiler zero-allocation primitives
- **Performance Benchmarks** (`test_performance.jl`): Overall performance characteristics

### 4. **Statistical Correctness** (`test/statistical_validation/`)
- **Bootstrap Validation** (`bootstrap_se_validation.jl`): Standard error validation against bootstrap estimates
- **Analytical Validation** (`analytical_se_validation.jl`): Analytical standard error verification
- **Backend Consistency** (`backend_consistency.jl`): AD vs FD computational agreement
- **Robust Standard Errors** (`robust_se_validation.jl`): CovarianceMatrices.jl integration
- **Elasticity SE Validation** (`elasticity_se_validation.jl`): Elasticity standard error verification
- **CI Validation** (`ci_validation.jl`): Confidence interval computation validation

### 5. **Validation Tests** (`test/validation/`)
- **Contrast Invariance** (`test_contrast_invariance.jl`): Categorical contrast mathematical invariance
- **Manual Counterfactual** (`test_manual_counterfactual_validation.jl`): Manual verification of marginal effects
- **Zero Allocation Comprehensive** (`test_zero_allocation_comprehensive.jl`): Complete allocation scaling validation
- **True Zero Allocation** (`test_true_zero_allocation.jl`): Core function zero-allocation verification

### 6. **Cross-Platform Validation** (`R_compare/`)
- **R Comparison Framework**: Systematic comparison against R's marginaleffects package
- **Dataset Coverage**: mtcars, ToothGrowth, Titanic datasets with matched model specifications
- **Statistical Agreement**: Validation of estimates and standard errors under matched assumptions

## Test Infrastructure

### **Test Utilities** (`test/test_utilities.jl`)
Provides shared infrastructure for all test files:

#### **Data Generation**
- `make_test_data(n=500)`: Standard test dataset with mixed variable types
- `test_data(n=200)`: Alternative test dataset for specific scenarios

#### **Formula Collections**
Comprehensive formula collections organized by model type:
- **Linear Models** (`linear_formulas`): 15 formulas from intercept-only to complex interactions
- **GLM Tests** (`glm_tests`): 10 formulas across Binomial, Poisson, Gamma, Normal distributions  
- **LMM Tests** (`lmm_formulas`): 6 formulas for linear mixed models with random effects
- **GLMM Tests** (`glmm_tests`): 4 formulas for generalized linear mixed models

#### **Performance Testing Functions**
- `test_zero_allocation()`: Zero-allocation validation for FormulaCompiler primitives
- `test_allocation_performance()`: Allocation scaling analysis across dataset sizes
- `test_model_correctness()`: Mathematical correctness verification

### **Test Organization**
Tests are organized into logical groups in `test/runtests.jl`:
- **Core Functionality**: Basic API and integration tests
- **Advanced Features**: Specialized functionality (elasticities, mixtures, etc.)
- **Performance**: Zero-allocation and scaling validation  
- **Statistical Correctness**: Statistical validity and SE verification
- **Validation Tests**: Mathematical invariance and edge case validation

## R Comparison Framework (`R_compare/`)

### **Objective**
Systematic validation of Margins.jl against R's established `marginaleffects` package to ensure statistical agreement and methodological consistency.

### **Framework Structure**
- **R Implementation** (`R_compare.md`): Complete specification for R-side computation
- **Julia Implementation** (`compare.jl`): Corresponding Julia computations  
- **Comparison Analysis**: Automated comparison of estimates and standard errors
- **Results Documentation**: Statistical summary of differences and validation

### **Dataset Coverage**
Standard econometric datasets with comprehensive model specifications:

#### **mtcars** (Motor Trend Car Road Tests)
- **Variables**: 32 observations, mixed continuous/categorical
- **Models**: 5 specifications from simple linear to complex interactions
- **Focus**: Linear regression marginal effects validation

#### **ToothGrowth** (Guinea Pig Tooth Growth)  
- **Variables**: 60 observations, factorial design
- **Models**: 5 specifications testing interactions and ordered factors
- **Focus**: Factorial design and interaction effects

#### **Titanic** (Survival Data)
- **Variables**: 2201 observations, multiple categorical predictors
- **Models**: 5 logistic regression specifications
- **Focus**: GLM marginal effects and categorical contrasts

### **Validation Approach**
- **Matched Assumptions**: Identical model specifications, data, and covariance estimators
- **Statistical Agreement**: Estimates within 1e-8 tolerance, standard errors within 1e-6
- **Scale Consistency**: Explicit matching of link/response scale computations
- **Categorical Handling**: Verification of identical contrast specifications

### **Key Challenges Addressed**
- **Factor Level Ordering**: Explicit factor level matching between R and Julia
- **Boolean Variables**: Consistent treatment as categorical with discrete contrasts  
- **Covariance Matrices**: Matched robust standard error computations
- **Profile Specifications**: Handling differences in "means" vs "modal" categorical defaults

## Performance Validation

### **Zero-Allocation Requirements**
Production-ready performance characteristics validated across all computational pathways:

#### **Core Functions** (`test/performance/test_zero_allocations.jl`)
- **FormulaCompiler Primitives**: Zero allocations for `modelrow!`, compiled evaluators
- **Derivative Computation**: Zero allocations for gradient and Hessian computations
- **Mathematical Operations**: Zero allocations for core mathematical functions

#### **Scaling Validation** (`test/validation/test_zero_allocation_comprehensive.jl`)
Comprehensive validation of O(1) allocation scaling achievements:
- **Population Margins**: Sublinear allocation growth validated across dataset sizes
- **Profile Margins**: O(1) constant allocation behavior regardless of dataset size  
- **Core AME Functions**: `_compute_all_continuous_ame_batch` maintains ≤15 allocations
- **Performance Improvements**: 12x+ allocation reduction from original implementation

#### **Allocation Targets**
- **FormulaCompiler Operations**: Strict zero allocations required
- **Population Margins**: <3000 allocations for full workflow (allowing infrastructure overhead)
- **Profile Margins**: O(1) scaling with reasonable fixed overhead
- **Categorical Operations**: <2500 allocations (accounting for DataFrame operations and contrast computation)

## Statistical Validation Framework

### **Bootstrap Validation** (`test/statistical_validation/`)
Comprehensive validation of standard errors against bootstrap estimates:

#### **Bootstrap SE Validation** (`bootstrap_se_validation.jl`)
- **Method**: 1000 bootstrap resamples with automatic bias correction
- **Coverage**: All model types (LM, GLM, LMM, GLMM) and effect types
- **Tolerance**: ±10% agreement between delta-method and bootstrap standard errors
- **Scope**: Population effects, profile effects, elasticities, and categorical contrasts

#### **Analytical Validation** (`analytical_se_validation.jl`)  
- **Method**: Manual analytical computation of gradients and delta-method standard errors
- **Verification**: Matrix computation validation against FormulaCompiler derivatives
- **Coverage**: Simple models with known analytical solutions

#### **Backend Consistency** (`backend_consistency.jl`)
- **AD vs FD**: Automatic differentiation vs finite difference computational agreement
- **Tolerance**: Machine precision agreement for derivative computation
- **Coverage**: All mathematical functions and model specifications

### **Robust Standard Errors** (`robust_se_validation.jl`)
Integration with CovarianceMatrices.jl for robust and clustered standard errors:
- **HC0, HC1, HC2, HC3**: Heteroskedasticity-robust standard errors  
- **Clustered**: Cluster-robust standard errors with arbitrary grouping
- **Validation**: Agreement with manual covariance matrix computation

## Development Commands

### **Running Tests**
```bash
# Full test suite
julia --project=. -e "using Pkg; Pkg.test()"

# Specific test files
julia --project=. test/runtests.jl test/core/test_glm_basic.jl

# Performance tests only
julia --project=. test/runtests.jl test/performance/test_zero_allocations.jl

# Statistical validation
julia --project=. test/runtests.jl test/statistical_validation/bootstrap_se_validation.jl
```

### **R Comparison Workflow**
```bash
# Generate R results (in R)
# Rscript R_compare/generate_r_results.R

# Run Julia comparison
julia --project=. R_compare/compare.jl

# Analyze differences
julia --project=. R_compare/analyze_differences.jl
```

### **Debugging Performance**
```bash
# Enable debug logging for allocation analysis
JULIA_DEBUG=Margins julia --project=. test/validation/test_zero_allocation_comprehensive.jl

# Profile memory usage
julia --project=. --track-allocation=user test/performance/test_performance.jl
```

## Current Status: Production Ready (September 2025)

### **Achievements**
- **60+ Test Files**: Comprehensive coverage across all functionality
- **Statistical Validation**: All standard errors validated against bootstrap estimates  
- **Performance Optimization**: O(1) allocation scaling achieved for core functions
- **Cross-Platform Validation**: R comparison framework ensures methodological consistency
- **Zero-Allocation Core**: FormulaCompiler primitives maintain strict zero-allocation guarantees

### **Success Criteria Met**

#### **Statistical Correctness**
- ✅ Delta-method standard errors validated against 1000-sample bootstrap estimates
- ✅ Backend consistency (AD vs FD) verified at machine precision
- ✅ Robust standard error integration with CovarianceMatrices.jl
- ✅ Cross-platform validation against R's marginaleffects package

#### **Performance Characteristics** 
- ✅ FormulaCompiler operations: Zero allocations achieved
- ✅ Population margins: O(n) scaling with reasonable overhead
- ✅ Profile margins: O(1) scaling independent of dataset size
- ✅ Core AME functions: 12x+ allocation reduction from original implementation

#### **Comprehensive Coverage**
- ✅ **Core functionality**: GLM integration, profiles, grouping, contrasts, error handling
- ✅ **Advanced features**: Elasticities, categorical mixtures, hierarchical grids  
- ✅ **Model types**: Linear, GLM, LMM, GLMM across all major distributions
- ✅ **Edge cases**: Mathematical invariance, manual counterfactual validation

### **Quality Assurance**
- **Test Organization**: Logical grouping by functionality with shared utilities
- **Continuous Integration**: Automated testing across Julia versions and operating systems
- **Documentation Standards**: Academic-level documentation with mathematical rigor
- **Error Handling**: Comprehensive input validation and meaningful error messages

The testing framework ensures Margins.jl meets publication-grade standards for econometric analysis with zero tolerance for invalid statistical results.