# R vs Julia Comparison Pipeline

This folder contains a reproducible pipeline to compare Margins.jl with R's `marginaleffects` on standard datasets.

What it does
- Prepares datasets in R and exports CSVs
- Fits models in R and Julia (LM/GLM; multinomial is R-only)
- Computes AME/MEM and AAP/APM in both languages
- Ensures covariance/scale parity for delta-method SEs
- Compares estimates and SEs with strict tolerances

## Layout

- `r/r_pipeline.R`: Prepare datasets, estimate models, compute AME/MEM and AAP/APM, export CSVs.
- `jl/jl_pipeline.jl`: Load CSVs, fit Julia models, compute Margins.jl outputs, export CSVs, compare.
- `jl/compare.jl`: Helper to align and compare results.
- `data/`: CSV datasets exported by R.
- `results_r/`: CSV results from R.
- `results_julia/`: CSV results from Julia + `compare_summary.csv`.
- `run.sh`: Convenience wrapper to run both steps with common flags.

## Requirements

- R packages: `marginaleffects`, `nnet`, `MASS`, `sandwich`.
  - Install: `install.packages(c("marginaleffects","nnet","MASS","sandwich"))`
- Julia packages: `Margins`, `GLM`, `DataFrames`, `CSV`, `CategoricalArrays`, `CovarianceMatrices`.
  - Install (Julia REPL):
    - `using Pkg`
    - `Pkg.add.( ["Margins","GLM","DataFrames","CSV","CategoricalArrays","CovarianceMatrices"] )`

Note: The scripts do not install packages automatically.

## Run

Option A — one-shot wrapper
- `bash R_compare/run.sh`
- Uses robust HC3 covariance and response scale by default.

Option B — step-by-step
1) Export data and R results
- `Rscript R_compare/r/r_pipeline.R --vcov HC3 --type response`

2) Compute Julia results and compare
- `julia --project -t auto R_compare/jl/jl_pipeline.jl --cov HC3 --scale response`

Outputs
- R results → `R_compare/results_r/`
- Julia results + comparison → `R_compare/results_julia/`
- Comparison summary → `R_compare/results_julia/compare_summary.csv`

## Flags and Parity Controls

- Covariance: `--vcov` (R) / `--cov` (Julia)
  - Supported: `HC3` (recommended), `HC1`, `HC0` (Julia); R accepts string for `vcov=`.
  - Omit to use model-based covariance in both.
- Scale: `--type` (R) / `--scale` (Julia)
  - `response` (default) or `link`.

## Datasets and Models

- Cross-package: `mtcars` (LM), `ToothGrowth` (LM), `Titanic` (GLM Binomial)
- R-only: `iris` (multinomial logistic via `nnet::multinom`)
- 5 models per dataset; results written for AME, MEM, AAP, APM.

## Interpreting Results

- `compare_summary.csv` reports, per dataset/model/type:
  - `n_match`: matched terms across outputs
  - `n_est_ok`, `n_se_ok`: counts within tolerances (estimates 1e-8, SE 1e-6)
  - `status`: `ok` if all matched terms agree within tolerance, else `diff`

## Notes (statistical correctness)

- Covariance parity: If robust in R (e.g., `HC3`), set the same in Julia. Do not mix robust/model-based across languages.
- Scale parity: Match `--type` (R) to `--scale` (Julia) explicitly.
- Profiles at means: Margins.jl `means_grid` uses frequency-weighted categoricals; `marginaleffects` typical rows use a single mode level. Treat differences as methodological unless switching Julia to a single-level `cartesian_grid`.
- No silent fallbacks: Julia uses Margins.jl defaults (AD backend). If AD isn’t available, error rather than approximate.

## Troubleshooting

- R packages missing: Install via `install.packages(...)` and re-run.
- Julia packages missing: Use `Pkg.add` (see Requirements) and re-run. If using a custom project, provide `--project` appropriately.
- `compare_summary.csv` shows `no_match`: Column naming mismatched; inspect CSVs and adjust mapping if needed.
- Large diffs on MEM/APM only: Likely due to profiles-at-means conventions (mixture vs single-level). Consider switching Julia to a single-level grid for strict parity.
