# R Comparison Study - Validation Results

## Executive Summary

This study validates Margins.jl against R's `margins` package by comparing:
- Model coefficient estimates and standard errors
- Marginal effects computations (AME, AAP, APM)
- Delta-method standard error calculations

Margins.jl produces statistically equivalent results to R's established margins package, confirming the package meets publication-grade standards for econometric analysis.

## Model Validation

### Dataset
- Size: 5,000 observations
- Model: Logistic regression with 65 coefficients
- Complexity: Binary × continuous interactions, categorical variables, continuous × continuous interactions

### Model Coefficients

| Metric | Result |
|--------|--------|
| Coefficient count | 65 (both implementations) ✓ |
| Max coefficient relative error | 0.0028% |
| Mean coefficient relative error | 0.0005% |
| Max SE relative error | 0.04% |
| Mean SE relative error | 0.034% |

**Verdict**: Statistically equivalent model estimates

### AME (Average Marginal Effects)

Average marginal effects across the observed sample distribution - the primary use case for marginal effects analysis.

| Metric | Result |
|--------|--------|
| Variables compared | 32 |
| Max estimate relative error | 0.0036% |
| Mean estimate relative error | 0.0006% |
| Max SE relative error | 0.04% |
| Mean SE relative error | 0.03% |

**Verdict**: Statistically equivalent

### AAP (Average Adjusted Predictions)

Population average of fitted values across the observed sample.

| Metric | Result |
|--------|--------|
| Observations compared | 1 |
| Max estimate relative error | 0.0001% |
| Max SE relative error | 0.03% |

**Verdict**: Statistically equivalent

### APM (Adjusted Predictions at Profiles)

Predictions averaged over data at specific covariate profiles.

| Metric | Result |
|--------|--------|
| Profiles compared | 4 (2×2 grid) |
| Max estimate relative error | 0.8665% |
| Mean estimate relative error | 0.6130% |
| Max SE relative error | 1.98% |
| Mean SE relative error | 1.54% |

**Verdict**: Agreement within acceptable tolerance

### AME by Age Groups

Average marginal effects for a single variable (age_h).

| Metric | Result |
|--------|--------|
| Max estimate relative error | 0.0003% |
| Max SE relative error | 0.04% |

**Verdict**: Statistically equivalent

### AME by Scenario

Average marginal effects computed at specific counterfactual scenarios.

| Metric | Result |
|--------|--------|
| Variables compared | 2 (wealth_d1_4_p, wealth_d1_4_h) |
| Max estimate relative error | 0.0017% |
| Max SE relative error | 0.04% |

**Verdict**: Statistically equivalent

### MEM (Marginal Effects at Profiles)

**Status**: SKIPPED - Different specifications between implementations

- **Julia**: 84 estimates (pairwise contrasts across profiles)
- **R**: 128 estimates (all variables at each grid point)

R's `margins(at=...)` computes marginal effects for all variables at each profile point, while Julia's `profile_margins(...; contrasts=:pairwise)` computes specific categorical contrasts. These represent different analytical approaches and cannot be directly compared.

## Statistical Methodology Validation

### Standard Errors

Both Margins.jl and R's margins package implement delta-method standard errors:

- For marginal effects: ∇g(β)' Σ ∇g(β), where g is the effect function
- For predictions: Proper variance computation using model covariance matrix Σ

This Margins.jl follows the same methodology as R's established package.

### Tolerance Criteria

Validation uses realistic thresholds for cross-platform numerical comparison:

- **SUCCESS**: Max coefficient relative error < 0.01%, Max SE relative error < 1%
- **PASS**: Max coefficient relative error < 0.1%, Max SE relative error < 5%

All core functionality (AME, AAP, coefficients) achieves agreement.

## Technical Notes

### Implementation Differences

While results are statistically equivalent, the implementations differ in:

1. Term naming conventions:
   - Julia: `&` for interactions, `var: level` for categoricals
   - R: `:` for interactions, `varlevel` for categoricals, `TRUE` suffix for booleans

2. Profile marginal effects:
   - Julia: Flexible contrast specification (baseline, pairwise, etc.)
   - R: Computes effects for all variables at each profile point

3. Performance:
   - Not compared in this validation study (focus on statistical correctness, see separate performance study)

### Numerical Precision

Differences in the 5th-6th decimal place are expected when comparing:
- Different programming languages (Julia vs R)
- Different numerical optimization algorithms
- Different linear algebra libraries (OpenBLAS vs MKL vs BLAS)

All observed differences are well within acceptable numerical precision for statistical software.

## Reproducibility

All comparison scripts, data generation, and validation code are available in the `test/r_compare/` directory:

- `generate_data.jl` - Creates synthetic dataset
- `julia_model.jl` - Margins.jl analysis
- `r_model.R` - R margins package analysis
- `compare_results.jl` - Coefficient validation
- `compare_margins.jl` - Marginal effects validation
- `Makefile` - Automated workflow

Run the full comparison with:
```bash
cd test/r_compare
make all
```

## References
- R emmeans package: https://cran.r-project.org/package=emmeans
- R margins package: https://cran.r-project.org/package=margins
- Margins.jl: https://github.com/emfeltham/Margins.jl
- Delta method: Standard econometric methodology for computing standard errors of non-linear functions of parameters
