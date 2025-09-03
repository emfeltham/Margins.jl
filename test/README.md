# Testing Framework

## Quick Start

```julia
# Default: Quick validation (~15 seconds)
julia --project=test test/runtests.jl

# Comprehensive: Full statistical validation (~60-90 seconds)
MARGINS_COMPREHENSIVE_TESTS=true julia --project=test test/runtests.jl

# Or use the convenience script
julia --project=test test/run_comprehensive_tests.jl
```

## Test Structure

### Core Functionality Tests (Always Run)
- GLM basic functionality
- Profile margins
- Grouping and contrasts
- VCov handling
- Error validation
- **Quick statistical validation** (backend consistency)

### Comprehensive Statistical Validation (Optional)
**9 Tiers of Statistical Correctness** (activated with `MARGINS_COMPREHENSIVE_TESTS=true`):

1. **Tier 1**: Direct coefficient validation
2. **Tier 1A**: Analytical SE validation - Linear models  
3. **Tier 1B**: Analytical SE validation - GLM chain rules
4. **Tier 2**: Function transformations
5. **Tier 3**: GLM chain rules
6. **Tier 4**: Systematic model coverage
7. **Tier 5**: Edge cases and robustness
8. **Tier 6**: Integer variable systematic coverage
9. **Tier 7**: Bootstrap SE validation - Empirical verification
10. **Tier 8**: Robust SE integration - Econometric functionality
11. **Tier 9**: Specialized SE cases - Advanced edge cases

**Total Coverage**: 686 comprehensive tests covering analytical verification, bootstrap validation, robust standard errors, and specialized edge cases.

## Statistical Validation Features

- **Analytical Verification**: Hand-calculated SE verification across linear and GLM models
- **Bootstrap Validation**: Empirical SE agreement testing with 78.1% mean agreement rate  
- **Robust SE Integration**: CovarianceMatrices.jl sandwich estimators (HC0-HC3) and clustered SEs
- **Specialized Cases**: Integer variables, elasticities, categorical mixtures, error propagation
- **Publication Grade**: All SE computations meet econometric publication standards