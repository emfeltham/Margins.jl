# R oracles for `random_effects` (mixed models)

`re_reference.R` writes `re_data.csv` (600 rows from R's RNG, so both languages read the same
data) and `re_reference.csv`, and `test/validation/test_oracle_random_effects.jl` (in
`runtests.jl`) compares Margins against them. R is not a test-time dependency.

```bash
R_LIBS=<private lib with lme4> Rscript re_reference.R
```

Versions: R 4.6.2, lme4 2.1.0 (with nloptr 2.2.1 against Homebrew's NLopt 2.11.0, in a private
library on `R_LIBS`), marginaleffects 1.0.0, numDeriv 2016.8.1.1. Julia side: MixedModels 4.38.1.

## Models and quantities

`lmm`: `y ~ x + w + (1 + w | rater) + (1 | village)` (ML); `glmm`: the same for `yb`, binomial
logit, Laplace. Rows of `re_reference.csv`:

| mode | R | Margins |
|---|---|---|
| `zero` | `avg_predictions` / `avg_slopes` with `re.form = NA` | `random_effects = :zero` |
| `cond_fixed` | the same with `re.form = NULL` (SEs on β only) | `:conditional`, `re_se = :fixed` |
| `cond_joint` | AAP, SE from Henderson's covariance built from lme4's X, Z, Λ and working weights | `:conditional` (`re_se = :joint`) |
| `marginal` | GLMM AAP and AMEs of x and w by R's adaptive `integrate` per row, numDeriv for the slopes and the β gradient | `:marginal` |
| `fit` | `fixef` and its SEs | the fit itself |

## Numerical settings

- **Fits.** lme4 with `nloptwrap` at `xtol_rel = 1e-14`, `ftol_rel = ftol_abs = 1e-15` (and
  `tolPwrss = 1e-12` for the GLMM); MixedModels with `ftol_rel = ftol_abs = 1e-14`,
  `xtol_rel = 1e-12` (both models) and `fast = false` for the GLMM. The log-likelihoods then
  agree to 1e-10; the tests use rtol 2e-6 (LMM) and 2e-5 (GLMM).
- **Covariance of β.** MixedModels' `vcov` is the RX-based one. For a GLMM that is lme4's
  `vcov(m, use.hessian = FALSE)`; lme4's default for `glmer` (`use.hessian = TRUE`, a
  finite-difference Hessian over (θ, β)) differs by 2.3% here. The script passes the RX-based
  one to marginaleffects (`vcov =`) and checks that Henderson's β block equals it (1e-14).
- **marginaleffects numerics.** `numderiv = "richardson"`, `eps = 1e-4`, as in `../oracle_3_0/`.
- **`:marginal`.** `integrate(..., rel.tol = 1e-13)`, independent of Margins' Gauss–Hermite rule.

## Observed agreement (max relative difference, Margins vs R)

| | LMM estimate | LMM SE | GLMM estimate | GLMM SE |
|---|---|---|---|---|
| `zero` | 2.9e-9 | 8.0e-8 | 2.7e-7 | 2.8e-7 |
| `cond_fixed` | 1.2e-9 | 8.0e-8 | 1.6e-7 | 3.4e-7 |
| `cond_joint` | 1.5e-15 | 4.1e-9 | 1.8e-8 | 3.8e-9 |
| `marginal` | | | 1.7e-7 | 2.6e-6 |

The differences are those of the two fits, except the `:marginal` GLMM SE, where R's nested
numerical derivatives (numDeriv over `integrate`) dominate.

## Found by this oracle

`MixedModels.ranef` on a `fast = false` GLMM fit is not the b̂ of the fitted linear predictor
(it differs from λu by up to 0.012 here, while `fitted(m)` uses λu); Margins reads b̂ = λu from the
fit instead (MixedModels 4.38.1; `fast = true` fits agree to 1e-15).
