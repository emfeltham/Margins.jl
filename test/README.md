# Testing Framework

## Quick Start

```julia
# Default
julia --project=. test/runtests.jl

# Comprehensive: also runs the extra elasticity and categorical-mixture checks
MARGINS_COMPREHENSIVE_TESTS=true julia --project=. test/runtests.jl
```

## Test Structure

### Core Functionality Tests (Always Run)
- GLM basic functionality
- Profile margins
- Grouping and contrasts
- VCov handling
- Error validation
- **Quick statistical validation** (backend consistency)

### Statistical Validation (Always Run)
`statistical_validation/statistical_validation.jl` is organized in numbered tiers:

1. **Tier 1**: Direct coefficient validation (all 2×2 quadrants)
2. **Tier 1A**: Analytical SE validation - linear models
3. **Tier 1B**: Analytical SE validation - GLM chain rules
4. **Tier 2**: Function transformations
5. **Tier 3**: GLM chain rules
6. **Tier 4**: Systematic model coverage
7. **Tier 5**: Edge cases and robustness
8. **Tier 6**: Integer variable systematic coverage

Further validation lives in separate files in `statistical_validation/`:

- `backend_consistency.jl`: AD vs FD agreement
- `incompatible_formula_se_validation.jl`: SEs for formulas that need special handling
- `elasticity_se_validation.jl`: elasticity SEs (bootstrap)
- `analytical_elasticity_se_validation.jl`: elasticity SEs (hand-calculated)
- `bootstrap_validation_tests.jl`: bootstrap SE agreement
- `robust_se_tests.jl`: CovarianceMatrices.jl robust and clustered SEs
- `population_context_bootstrap_validation.jl`: bootstrap SEs for `population_margins` with scenarios
- `backend_reliability_tests.jl`: AD vs FD behavior on domain-sensitive functions
- `specialized_se_tests.jl`: integer variables, elasticities, categorical mixtures, error propagation

`MARGINS_COMPREHENSIVE_TESTS=true` enables additional checks in `elasticity_se_validation.jl`.

## Statistical Validation Features

- **Analytical Verification**: Hand-calculated SE verification across linear and GLM models
- **Bootstrap Validation**: Empirical SE agreement testing  
- **Robust SE Integration**: CovarianceMatrices.jl sandwich estimators (HC0-HC3) and clustered SEs
- **Specialized Cases**: Integer variables, elasticities, categorical mixtures, error propagation
- **Publication Grade**: All SE computations meet econometric publication standards