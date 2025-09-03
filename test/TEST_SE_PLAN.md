# Standard Error Testing Plan for Margins.jl

> **Note**: Use `--project="test"` to run test files.

## Current Status

SE testing infrastructure exists but **cannot be verified** due to test environment failures.

N.B., the tests themselves are likely written for an older API in many cases, which has changed quite a bit.

## Action Plan

### Phase 1: Restore Basic Functionality ⚠️ **URGENT**

- [ ] **Fix test environment**
  - [ ] Resolve dependency/precompilation errors preventing Margins from loading in test environment
  - [ ] Ensure `julia --project=test -e "using Margins"` works without errors
  - [ ] Verify test environment has all required dependencies

- [ ] **Verify core API functionality**
  - [ ] Test basic `population_margins()` call works
  - [ ] Test basic `profile_margins()` call works  
  - [ ] Verify `DataFrame(result)` produces expected columns (including `.se`)

- [ ] **Run minimal SE test**
  - [ ] Create simple test: fit model, compute margins, check SE is finite and positive
  - [ ] Verify SE values are reasonable (not NaN, not zero, appropriate magnitude)

### Phase 2: Validate Existing Infrastructure 📋 **HIGH PRIORITY**

- [ ] **Test analytical SE validation**
  - [ ] Run `test/statistical_validation/analytical_se_validation.jl`
  - [ ] Verify linear model SE validation works
  - [ ] Check GLM chain rule SE validation works
  - [ ] Confirm hand-calculated vs computed SE agreement

- [ ] **Test bootstrap framework** 
  - [ ] Run bootstrap SE validation tests
  - [ ] Verify bootstrap vs delta-method SE comparison works
  - [ ] Check if claimed "78% agreement" is achievable

- [ ] **Test backend consistency**
  - [ ] Verify AD vs FD backend SE agreement
  - [ ] Check numerical tolerances are appropriate
  - [ ] Test graceful fallback when AD fails

### Phase 3: Advanced SE Features 🎯 **MEDIUM PRIORITY**

- [ ] **Robust SE integration**
  - [ ] Test CovarianceMatrices.jl integration works
  - [ ] Verify sandwich estimators (HC0, HC1, HC2, HC3) function
  - [ ] Test clustered standard errors
  - [ ] Check robust vs classical SE differences are reasonable

- [ ] **Specialized cases**
  - [ ] Test integer variable SE computation
  - [ ] Verify elasticity measure SE calculation
  - [ ] Test categorical mixture SE validation
  - [ ] Check edge case handling (small samples, extreme values)

### Phase 4: Integration and Documentation 📚 **LOW PRIORITY**

- [ ] **Integrate with main test suite**
  - [ ] Add SE tests to `test/runtests.jl`
  - [ ] Set up environment flags for comprehensive vs quick testing
  - [ ] Ensure tests run in CI environment

- [ ] **Update documentation**
  - [ ] Document SE testing framework for developers
  - [ ] Create troubleshooting guide for SE test failures
  - [ ] Update this plan with actual verified results

## Notes

### Infrastructure Files
- Analytical SE validation: `test/statistical_validation/analytical_se_validation.jl`
- Bootstrap validation: Various files in `test/statistical_validation/`
- Backend consistency: `test/statistical_validation/backend_consistency.jl`
- Robust SE integration: `test/statistical_validation/robust_se_*`

### Success Metrics
- **SE Agreement**: Bootstrap validation within 10-15% for valid test cases
- **Analytical Verification**: Hand-calculated SEs match computed SEs within numerical precision
- **Backend Consistency**: AD/FD SE agreement within appropriate tolerances

### Risk Mitigation
- **Multi-Method Validation**: Use bootstrap + analytical verification
- **Known-Answer Testing**: Start with cases where SEs are hand-calculable
- **Gradual Expansion**: Restore basic functionality before testing advanced features
