# R oracles for Margins 3.0

Reference values from R for the paths added or changed in Margins 3.0: the default
`at = :observed` second difference, fits with estimation weights, offsets, and
FixedEffectModels. As with `../second_differences/` and `../elasticity/`, R is not a test-time
dependency. The R scripts here write CSV fixtures, which are committed, and
`test/validation/test_oracle_3_0.jl` (in `runtests.jl`) compares Margins against them.

## Files

| Script | Fixture | Rows | What it covers |
|---|---|---|---|
| `oracle_data.R` | | | the shared deterministic dataset (400 rows, no RNG); Julia: `test/validation/oracle_3_0_data.jl` |
| `data_checksums.R` | `data_checksums.csv` | 10 | column sums and sums of squares, so the test can check that both languages built the same data |
| `observed_sd_reference.R` | `observed_sd_reference.csv` | 32 | `second_differences(m, d, focal, :w)` with the default `at = :observed` |
| `weights_reference.R` | `weights_reference.csv` | 216 | AME, AAP, MEM and `groups = :g` for fits with integer (`fw`) and non-integer (`sw`) weights |
| `offsets_reference.R` | `offsets_reference.csv` | 156 | Poisson with `offset(log(expo))`: AAP, AME, MEM and APM, with the offset given at the grid |
| `fixest_reference.R` | `fixest_reference.csv` | 129 | `reg(... + fe(st) + fe(yr))` against `fixest::feols(y ~ x * z \| st + yr)` |
| `rank_deficient_reference.R` | `rank_deficient_reference.csv` | 90 | rank-deficient lm, logit and `feols` (a dropped collinear coefficient), with the refit's values alongside |

Regenerate from this directory (fixest is not in the default R library; see below):

```bash
Rscript data_checksums.R        > data_checksums.csv
Rscript observed_sd_reference.R > observed_sd_reference.csv
Rscript weights_reference.R     > weights_reference.csv
Rscript offsets_reference.R     > offsets_reference.csv
R_LIBS=<private lib with fixest> Rscript fixest_reference.R > fixest_reference.csv
R_LIBS=<private lib with fixest> Rscript rank_deficient_reference.R > rank_deficient_reference.csv
```

Versions: R 4.6.1, marginaleffects 1.0.0, fixest 0.14.2 (with its dependencies dreamerr,
stringmagic, zoo, and sandwich 3.1.3, installed into a private library on `R_LIBS`, not the
default one). Julia side: Julia 1.13.0, GLM 1.9.5, FixedEffectModels 1.13.3, MixedModels
4.38.1, ForwardDiff 1.4.6. Every number in the fixtures has 17 significant digits.

## Numerical settings shared by every script

- **Tight fits.** The R GLMs use `glm.control(epsilon = 1e-14)`, and the Julia GLMs use
  `rtol = 1e-15, atol = 0`. At the default tolerances GLM.jl and R stop at fits that differ by
  up to 1e-5 relative (probit here), which would dominate every comparison. FixedEffectModels
  uses `tol = 1e-12` and fixest `fixef.tol = 1e-11` (its minimum is 2.2e-12). At the defaults
  the fixed effects agree only to about 4e-8.
- **marginaleffects numerics.** `numderiv = "richardson"` for the Jacobian and `eps = 1e-4` for
  the numerical slope. marginaleffects' slope is a central difference. At `eps = 1e-6` its
  rounding error, amplified by the numerical Jacobian, reaches 1e-3 relative in unit-level (MEM)
  SEs. At 1e-4 the truncation error is about 1e-11. Margins differentiates with ForwardDiff.

## Cases and observed differences

Max absolute / relative difference, Margins − R. "SE" for the Gaussian-weights and fixest cases
is after dividing by the convention ratio described below. Tolerances in the test are a few
times these.

### `at = :observed` second differences (32 rows)

R stacks two shifted copies of the data in one `newdata` (every row's `w` minus δ and plus δ,
labelled `shift_`). It then calls
`avg_slopes(m, variables = "x", newdata = nd, by = "shift_", hypothesis = "(b2 - b1) / (2δ) = 0")`,
or `avg_comparisons` with one contrast of `g` at a time. The SE comes from the joint Jacobian of
the two averages. They are almost perfectly correlated, so an SE built from two independent
AMEs would be about 60 times too large here. δ = 0.01·sd(w), Margins' `delta = :auto`. The test
checks `delta_used` against it. `groups = :g` restricts `newdata` to each stratum's rows, with
the same fit.

Models: `lm` and logit/probit `yc|yb ~ x * w + z + g * w`, an additive logit (no interaction,
response scale only), and a weighted logit (`sw`, Margins `weights = :model`, R `wts = "sw"`).
Each has focal `x` (dy/dx, plus eyex on the response scale for logit and wlogit) and `g` (b − a,
c − a), on the response and link scales.

| Model | rows | estimate abs / rel | SE abs / rel |
|---|---|---|---|
| lm | 3 | 1.4e-11 / 3.5e-11 | 4.8e-12 / 6.7e-11 |
| logit (incl. 6 `groups` rows) | 13 | 1.0e-10 / 8.5e-10 | 4.1e-6 / 1.5e-5 (eyex; dy/dx ≤ 2.3e-9) |
| probit | 6 | 2.2e-11 / 5.4e-10 | 1.0e-10 / 2.3e-9 |
| logit, no interaction | 3 | 1.0e-11 / 2.7e-10 | 9.7e-12 / 1.0e-9 |
| weighted logit | 7 | 1.2e-10 / 4.6e-10 | 4.7e-7 / 2.0e-6 (eyex) |

Test tolerances: estimate rtol 1e-7, SE rtol 1e-6, and 1e-4 for eyex, whose nested numerical
derivative divides by ŷ. Exact references with no R involved: the link-scale `at = :observed`
second difference of `x` across `w` equals the `x & w` coefficient, and its SE equals that
coefficient's SE, to 1e-9.

### Model weights (216 rows)

`lm`, Gaussian `glm`, logit and Poisson, `yc|yb|cnt ~ x * z + g`, each with `fw` (1–4) and
`sw` (0.5–2.5). Margins uses the fit's weights (`weights = :model`), R uses `wts =` the same
column. Quantities: AME (all variables, and `x`, `z` by `g`), AAP (overall and by `g`), and MEM
of `x`, `z` at fixed values and at the weighted means of `x`, `z` (both with `g = "b"`).

| | estimate rel | SE rel |
|---|---|---|
| logit, Poisson (fw, sw) | ≤ 2.1e-10 | ≤ 5.0e-10 |
| lm, Gaussian glm (fw, sw) | ≤ 1.2e-11 | ≤ 1.5e-11 after the ratio |

**Convention (characterized, not a tolerance).** GLM.jl treats `wts` as frequency weights, so a
model with an estimated dispersion has σ̂² = Σwᵢrᵢ²/(Σw − p). R's `lm`/`glm(weights =)` treat
them as precision weights, σ̂² = Σwᵢrᵢ²/(n − p). Every SE of the lm and Gaussian fits therefore
differs by exactly sqrt((n − p)/(Σw − p)), with n = 400 and p = 6:

| weights | Σw | predicted Margins/R SE ratio | observed ratio (all 36 rows of each model) |
|---|---|---|---|
| fw | 1000 | 0.629585792103 | 0.629585792098 to 0.629585792112 |
| sw | 601.2 | 0.813610696581 | 0.813610696578 to 0.813610696587 |

The test requires the ratio to equal the prediction to 1e-8. The Gaussian fits' residual df is
Σw − p, so Margins' t inference uses df = 994 for `fw`, and a non-integer 595.2 for `sw`.
marginaleffects uses z, so p-values and CIs are not compared. For logit and Poisson the
dispersion is fixed, the two conventions coincide, and the SEs agree.

### Offsets (156 rows)

`cnt ~ x * z + g` with `offset(log(expo))`, unweighted and with `sw`. Quantities: AAP (overall
and by `g`), AME (`x`, `z`, `g`; by `g`) on the response and link scales, and, at a grid with
the exposure given, MEM of `x`, `z`, `g` and the adjusted predictions. There are two grids, one
row with `expo = 3` and a 2 × 2 grid of `x` ∈ {1, 2} and `expo` ∈ {1, 5}. Margins takes the
offset as `offset = log(expo)`.

| | estimate rel | SE rel |
|---|---|---|
| AAP, APM | ≤ 3.1e-15 | ≤ 1.3e-9 |
| AME | ≤ 2.2e-11 | ≤ 1.2e-6 |
| MEM | ≤ 3.1e-11 | ≤ 2.6e-5 |

The unweighted AAP equals `mean(cnt)` in both languages (log link with an intercept). The
test also checks that a grid `offset` column gives the same result as the `offset =` keyword,
and that a grid with neither is an `ArgumentError`.

How marginaleffects handles the offset, which the R script has to work around:

- With `offset(log(expo))` in the formula, `expo` is an ordinary variable, so `newdata` must
  carry it. `datagrid()` fills a variable it is not given with its mean, so a grid without
  `expo` silently uses log(mean(expo)), which is not the mean offset. Margins raises an error
  instead.
- With `glm(..., offset = log(expo))`, a grid without `expo` is an error.
- `slopes()` also differentiates with respect to `expo` unless `variables` excludes it.
- `predictions()` on a GLM defaults to `type = "invlink(link)"`, which has no SE, so
  `type = "response"` is used.
- A categorical variable fixed in the grid: marginaleffects reports every contrast at the row
  (both b − a and c − a at `g = "b"`), while Margins reports that row's own contrast. R's
  c − a is therefore compared with Margins' row at `g = "c"`.

### FixedEffectModels vs fixest (129 rows)

`y ~ x * z + fe(st) + fe(yr)` on a 12-state × 11-year panel. 12 and 11 are coprime, so every
state-year pair occurs and the fixed-effects graph is connected. Three singleton states and a
row with a missing `z` are outside the estimation sample: 300 of 304 rows are used, via
`drop_singletons` in Julia and `fixef.rm = "singleton"` in R. The comparison uses three
covariance estimators: `Vcov.simple()`/`"iid"`, `Vcov.robust()`/`"hetero"`, and
`Vcov.cluster(:st)`/`~st`. Quantities:

- the coefficients;
- the AAP, overall and by state, which includes the estimated fixed effects (it equals the mean
  of `fixest::predict`);
- the AME of `x` and `z`, overall and by state;
- the prediction at a grid row (`st = s3, yr = 2004`), with α and γ looked up from the fit. It
  matches `predict(m, newdata = …)`.

marginaleffects' default `newdata` is the whole data frame. It gives NA predictions for the
dropped rows and NaN averages for the singleton states, so the estimation sample is passed
explicitly. Like Margins, it treats the fixed effects as known.

| | estimate rel | SE rel after the ratio |
|---|---|---|
| coefficients | ≤ 3.9e-14 | ≤ 2.4e-12 |
| AAP, prediction at the grid row | ≤ 2.0e-13 | ≤ 2.9e-10 |
| AME | ≤ 4.1e-12 | ≤ 9.3e-7 |

**Convention (characterized).** The packages count the absorbed degrees of freedom
differently. fixest uses K = 3 + 12 + 11 − 1 = 25, so n − K = 275. FixedEffectModels uses
`dof_fes` = 12 + 11 = 23, so `dof_residual` = 274: it does not subtract the redundant level of
the second fixed effect. With state clusters, fixest drops the state FE as nested in the
cluster and uses n − K = 286, against FixedEffectModels' 285. Every SE, coefficients
included, therefore differs by the same constant:

| vcov | predicted FEM/fixest SE ratio | observed (coefficients; all quantities) |
|---|---|---|
| simple, robust | sqrt(275/274) = 1.0018231556 | 1.0018231556; within 1e-6 |
| cluster | sqrt(286/285) = 1.0017528497 | 1.0017528497; within 1e-6 |

Margins' delta method carries the model's own convention unchanged. Its df is FixedEffectModels'
`dof_residual`, 274 here, or G − 1 = 11 when clustered. marginaleffects reports z.

### Rank-deficient fits (90 rows)

`v = x + z` makes `z` redundant in `lm(yc ~ x * w + g + v + z)` and the logit
`yb ~ x + w + g + v + z` (GLM.jl's pivoted Cholesky drops `z`; R drops the last aliased column, so `z`
is last in R's formula), and `c` (a value per level of `g`) is collinear with `fe(g)` in
`feols(yc ~ x * w + z + c | g)` / `reg(... + fe(g))`. Quantities: AME of every variable, AME by `g`,
AAP overall and by `g`, and the effects at one grid row, on the response scale (and the link scale for
the logit). Each row also has `refit_estimate` / `refit_std_error`, the same quantity from R on the
refit without the redundant term.

R's glm detects the collinear column only with `epsilon = 1e-11` (its QR tolerance is
min(1e-7, epsilon/1000)), so the logit is fit less tightly than in the other scripts.

| | estimate rel | SE rel |
|---|---|---|
| vs R on the rank-deficient fit | ≤ 2.0e-10 | ≤ 1.9e-5 |
| vs R on the refit | ≤ 2.0e-10 | ≤ 1.6e-6 |

marginaleffects' numerical Jacobian is noisier on the rank-deficient `lm`: the MEM SE of `x` is
1.9e-5 from the refit's (exact for a linear model, and equal to Margins' to 1e-12), 2.6% with the
default `numderiv`. The effect of the dropped variable (`z`, `c`; 20 rows) is 0 with SE `NA` in R and
0 with SE NaN in Margins (not identified).

## Related Julia-only checks

`test/validation/test_second_differences_mixed_fem.jl` covers the model-level
`second_differences` on an LMM, a Bernoulli GLMM and a FixedEffectModels fit that drops rows.
It runs `at = :observed`, `at = :mean`, `levels`, a binary modifier, `groups =` and the
result-level form, and checks them against hand computations (interaction coefficients, a
fixed-effects-predictor finite-difference oracle for the GLMM) and AD ≈ FD.
