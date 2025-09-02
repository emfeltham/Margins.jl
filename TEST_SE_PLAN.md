# Standard Error Testing Plan for Margins.jl

## Executive Summary

**CURRENT STATUS**: Standard error testing is **PARTIALLY COMPREHENSIVE** but needs systematic expansion.

**KEY FINDING**: While we extensively test point estimates with analytical validation across 6 tiers, our standard error testing relies primarily on:
1. **Basic finite/positive validation** (extensive coverage)
2. **Single bootstrap comparison test** (limited scope) 
3. **Backend consistency testing** (AD vs FD agreement)

**CRITICAL GAP**: We need the same level of **analytical validation** for standard errors that we have for point estimates.

## Current SE Testing Assessment

### ✅ **What We Test Well**

#### 1. **Basic Validity Testing** (Comprehensive Coverage)
- **Location**: Used throughout all test files via `validate_all_finite_positive()`
- **Coverage**: Every test validates SE finitude and non-negativity
- **Pattern**:
  ```julia
  @test validate_all_finite_positive(result_df).all_valid
  ```
- **Assessment**: **EXCELLENT** - catches numerical issues systematically

#### 2. **Backend Consistency Testing** (Strong)
- **Location**: `test/statistical_validation/backend_consistency.jl`
- **Coverage**: Validates AD vs FD SE agreement across all 2×2 quadrants
- **Tolerances**: `rtol=1e-8` for SE agreement 
- **Pattern**:
  ```julia
  ses_agree = all(isapprox.(ad_df.se, fd_df.se; rtol=rtol_se))
  ```
- **Assessment**: **STRONG** - ensures computational consistency

#### 3. **Bootstrap Validation** (Limited but Present)
- **Location**: `test/test_statistical_validation.jl`
- **Coverage**: Single linear model test with bootstrap comparison
- **Validation**: 500 bootstrap samples, SE ratio agreement within 10%
- **Assessment**: **GOOD PROOF OF CONCEPT** but needs expansion

### ❌ **Critical Gaps in SE Testing**

#### 1. **No Analytical SE Validation**
- **Issue**: Point estimates have hand-calculated analytical verification, SEs don't
- **Example**: We verify `∂y/∂x = β₁/x` for log models, but not `SE(∂y/∂x) = SE(β₁)/x`
- **Impact**: We trust delta-method implementation without mathematical verification

#### 2. **Limited Bootstrap Coverage**
- **Issue**: Only one basic linear model tested against bootstrap
- **Missing**: GLM models, categorical effects, profile margins, elasticities
- **Impact**: Most complex SE computations never validated empirically

#### 3. **No Coverage Probability Testing**
- **Issue**: We don't validate that 95% confidence intervals have ~95% coverage
- **Missing**: Systematic coverage testing across model types
- **Impact**: Cannot verify CI reliability for publication use

#### 4. **No Robust SE Testing**
- **Issue**: No validation of sandwich/clustered standard errors
- **Missing**: CovarianceMatrices.jl integration testing
- **Impact**: Robust SEs untested despite being critical for econometrics

## Recommended SE Testing Expansion

### **TIER 1: Analytical SE Validation** (Priority: CRITICAL)

Develop hand-calculated SE verification for simple cases, following our point estimate pattern.

#### Linear Model SE Validation
```julia
@testset "Analytical SE Verification - Linear Models" begin
    # For y ~ x model: SE(β₁) should match GLM vcov
    model = lm(@formula(y ~ x), df)
    vcov_matrix = GLM.vcov(model)
    manual_se_x = sqrt(vcov_matrix[2,2])  # SE of x coefficient
    
    # Population margins SE should equal coefficient SE (linear case)
    result = population_margins(model, df; type=:effects, vars=[:x])
    @test DataFrame(result).se[1] ≈ manual_se_x atol=1e-12
end
```

#### GLM Chain Rule SE Validation  
```julia
@testset "Analytical SE Verification - GLM Chain Rule" begin
    # For logistic: SE(∂μ/∂x) = SE(β₁) × |∂²μ/∂x∂β₁| via delta method
    model = glm(@formula(y ~ x), df, Binomial(), LogitLink())
    
    # Hand-calculate delta method SE for specific profile
    x_test = 1.0
    β₁ = coef(model)[2]
    se_β₁ = sqrt(GLM.vcov(model)[2,2])
    
    η_test = GLM.predict(model, DataFrame(x=x_test), scale=:linear)[1]
    μ_test = 1 / (1 + exp(-η_test))
    
    # Chain rule: ∂μ/∂β₁ = μ(1-μ) × x
    chain_derivative = μ_test * (1 - μ_test) * x_test
    manual_se_profile = abs(chain_derivative) * se_β₁
    
    result = profile_margins(model, df; type=:effects, vars=[:x], 
                           at=Dict(:x => x_test), target=:mu)
    @test DataFrame(result).se[1] ≈ manual_se_profile atol=1e-10
end
```

### **TIER 2: Systematic Bootstrap Validation** (Priority: HIGH)

Expand bootstrap testing to match our comprehensive model coverage.

#### Multi-Model Bootstrap Testing
```julia
test_models = [
    (lm, @formula(y ~ x + z), :linear),
    (m -> glm(m, df, Binomial(), LogitLink()), @formula(y ~ x + z), :logistic),  
    (m -> glm(m, df, Poisson(), LogLink()), @formula(y ~ x + z), :poisson)
]

for (model_func, formula, model_type) in test_models
    @testset "Bootstrap SE Validation - $model_type" begin
        model = model_func(formula)
        
        # Test all 2×2 quadrants against bootstrap
        validate_bootstrap_agreement(model, df; 
                                   quadrants=[:population_effects, :population_predictions,
                                            :profile_effects, :profile_predictions])
    end
end
```

#### Bootstrap Utilities to Develop
```julia
function validate_bootstrap_agreement(model, data; n_bootstrap=500, tolerance=0.15)
    # Bootstrap all four quadrants systematically
    # Return SE agreement rates for each quadrant
end

function bootstrap_confidence_intervals(model, data; confidence=0.95)
    # Generate bootstrap CIs for coverage testing
end
```

### **TIER 3: Coverage Probability Testing** (Priority: HIGH)

Validate that confidence intervals have correct coverage rates.

#### Coverage Testing Framework
```julia
@testset "Confidence Interval Coverage Testing" begin
    # Generate multiple datasets, check CI coverage rates
    coverage_rates = []
    
    for sim_id in 1:100  # Multiple simulations
        df_sim = generate_test_data(n=500, sim_id=sim_id)
        model = lm(@formula(y ~ x + z), df_sim) 
        
        result = population_margins(model, df_sim; type=:effects, vars=[:x])
        result_df = DataFrame(result)
        
        # Check if true parameter (known from simulation) falls in CI
        true_effect = 0.5  # Known from data generation
        ci_lower = result_df.estimate[1] - 1.96 * result_df.se[1] 
        ci_upper = result_df.estimate[1] + 1.96 * result_df.se[1]
        
        coverage = (ci_lower <= true_effect <= ci_upper)
        push!(coverage_rates, coverage)
    end
    
    empirical_coverage = mean(coverage_rates)
    @test abs(empirical_coverage - 0.95) < 0.05  # Within 5% of nominal
end
```

### **TIER 4: Robust SE Testing** (Priority: MEDIUM)

Validate robust/clustered standard errors integration.

#### CovarianceMatrices.jl Integration
```julia
@testset "Robust Standard Errors Integration" begin
    using CovarianceMatrices
    
    # Test sandwich estimator
    robust_vcov = CovarianceMatrices.HC1(model)
    result = population_margins(model, df; vcov=robust_vcov)
    
    # Compare to manual robust SE calculation
    validate_robust_se_computation(result, model, robust_vcov)
end
```

### **TIER 5: Specialized SE Cases** (Priority: LOW)

Address edge cases and advanced features.

#### Integer Variable SE Testing
```julia
@testset "Integer Variable SE Validation" begin
    # Verify that integer→float conversion doesn't affect SEs
    # Test that SE computation handles integer variables correctly
end
```

#### Elasticity SE Testing  
```julia
@testset "Elasticity Standard Errors" begin
    # Validate delta method for elasticity transformations
    # Compare to numerical derivatives of elasticity formula
end
```

#### Categorical Mixture SE Testing
```julia
@testset "Categorical Mixture SE Validation" begin
    # Test SE computation for weighted categorical scenarios
    # Verify delta method handles mixture weights correctly
end
```

## Implementation Strategy

- [ ] **Phase 1: Foundation** (Weeks 1-2)
    - [ ] **Analytical SE verification utilities**
        - [ ] Delta method SE calculators for common cases
        - [ ] Hand-calculation verification functions
        - [ ] Linear model analytical SE testing
    - [ ] **Enhanced bootstrap framework**
        - [ ] Multi-model bootstrap testing utilities
        - [ ] Systematic coverage testing framework
        - [ ] Performance optimization for large bootstrap samples

- [ ] **Phase 2: Core Coverage** (Weeks 3-4 
    - [ ] **Tier 1 analytical validation**
        - [ ] Linear model SE verification across all 2×2 quadrants
        - [ ] GLM chain rule SE verification for logistic/Poisson
        - [ ] Integration into main statistical_validation.jl
    - [ ] **Tier 2 bootstrap expansion**
        - [ ] Bootstrap testing for GLM models
        - [ ] Profile margins bootstrap validation
        - [ ] Categorical effects bootstrap testing

- [ ] **Phase 3: Advanced Features** (Weeks 5-6)
    - [ ]  **Tier 3 coverage testing**
        - [ ] Systematic CI coverage rate validation
        - [ ] Multiple simulation coverage testing
        - [ ] Coverage testing across model types
    - [ ]  **Tier 4 robust SE integration** 
        - [ ] CovarianceMatrices.jl integration testing
        - [ ] Sandwich estimator validation
        - [ ] Clustered SE testing framework

- [ ]  **Phase 4: Specialized Cases** (Weeks 7-8)
    - [ ]  **Tier 5 edge cases**
        - [ ]  Integer variable SE edge cases
        - [ ]  Elasticity SE validation  
        - [ ]  Categorical mixture SE testing
        - [ ]  Error propagation testing

## Success Metrics

### **Statistical Validity Standards**
- **SE Agreement**: Bootstrap validation within 10% for 95% of test cases
- **Coverage Rates**: 95% CIs achieve 93-97% empirical coverage 
- **Analytical Verification**: Hand-calculated SEs match computed SEs within numerical precision
- **Backend Consistency**: AD/FD SE agreement within 1e-8 tolerance

### **Test Coverage Goals**
- **Model Coverage**: Linear, Logistic, Poisson, MixedModels SE testing
- **Feature Coverage**: All 2×2 quadrants, elasticities, categoricals, robust SEs
- **Edge Case Coverage**: Integer variables, small samples, extreme coefficients
- **Performance Coverage**: Large dataset SE computation validation

## Risk Assessment

### **High Risk Issues**
1. **Delta Method Implementation Bugs**: Could produce plausible but wrong SEs
2. **Chain Rule Errors**: GLM marginal effect SEs particularly vulnerable  
3. **Covariance Matrix Handling**: Robust SE integration failure modes

### **Mitigation Strategies**
1. **Multi-Method Validation**: Bootstrap + analytical + coverage testing
2. **Known-Answer Testing**: Use cases with hand-calculable SE solutions
3. **Comparative Validation**: Cross-check against Stata margins command SEs

## Integration with Existing Tests

### **Minimal Disruption Approach**
- Extend existing `validate_all_finite_positive()` for enhanced SE checking
- Add SE-specific validation to `test_2x2_framework_quadrants()`
- Integrate bootstrap utilities into existing test infrastructure
- Maintain current analytical validation pattern but expand to SEs

### **Backward Compatibility**
- All existing tests continue to work unchanged
- New SE validation is additive to current testing
- Optional detailed SE validation flags for performance
- Graceful degradation when bootstrap/coverage tests fail

## Conclusion

The standard error testing plan addresses the critical gap between our excellent point estimate validation and our limited SE verification. By implementing analytical SE validation, systematic bootstrap testing, and coverage probability validation, we will achieve the same publication-grade SE standards that we have for point estimates.

**Priority**: Focus on **Tier 1 (analytical validation)** and **Tier 2 (bootstrap expansion)** first, as these provide the highest statistical confidence improvement with manageable implementation complexity.

**Timeline**: 8-week implementation provides comprehensive SE testing infrastructure matching the rigor of our current point estimate validation framework.

**Impact**: Enables full confidence in Margins.jl standard errors for econometric publication use, completing our statistical correctness objectives.