# Testing Framework

## Quick Start

```julia
julia --project=. test/runtests.jl
```

## Run time and slow runs

The full suite takes about 16–17 minutes with `--check-bounds=yes --threads=4` on a 14-core Apple
Silicon laptop; "Statistical Correctness" takes about 2.5 minutes of that. `runtests.jl` prints a
`Test environment:` header (Julia and GC threads, BLAS library and thread count,
`OPENBLAS_NUM_THREADS`) and, after each top-level testset, a line such as

```
[clock] Statistical Correctness: wall 168.6s, CPU 205.8s, machine asleep 0.0s
```

If a run is much slower than usual, check this line first. When the wall time far exceeds the CPU
time and "machine asleep" accounts for the difference, the OS suspended the process and none of
the tests was slow. On a laptop this happens when the lid is closed: clamshell sleep is not
prevented by `caffeinate`, Amphetamine or other idle-sleep assertions (only an external display,
power and keyboard, or Amphetamine's closed-display mode keep the machine awake). The process runs
only during the short maintenance "DarkWake" periods, so a 2.5-minute testset can take 30
minutes. `pmset -g log | grep -E " Sleep | Wake | DarkWake "` shows when the machine slept. The
three slow runs seen while preparing 3.0.0 (a 21-minute concurrency test, a 38-minute
`core/test_weights.jl`, a 30-minute "Statistical Correctness") each coincided with a clamshell
sleep; the 43-minute full suite that contained the last one used the same CPU time (about 2200 s)
as a normal 16-minute run.

BLAS threads: OpenBLAS uses its own thread pool (by default one thread per performance core) on
top of Julia's `--threads`. After each multithreaded BLAS call its idle workers spin for a short
time before sleeping, which shows up as system CPU time, but it does not change the suite's wall
time measurably: "Statistical Correctness", `core/test_weights.jl` and
`validation/test_concurrent_calls.jl` take the same time with `BLAS.set_num_threads(1)` as with
the default. The suite leaves the BLAS thread count at its default.

To get comparable timings for release gates: keep the machine awake with the lid open (or on
external power with a display), run gate suites one at a time rather than several full suites in
parallel, and read the `[clock]` lines rather than the total wall time.

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

- `backend_consistency.jl`: AD vs FD agreement, including near the log/sqrt domain boundary
- `incompatible_formula_se_validation.jl`: SEs for formulas that need special handling
- `robust_se_validation.jl` (run under Core): HC0–HC3 and CR0 SEs for `lm` and logit, checked against hand-computed sandwich matrices and sqrt(g'Vg)
- `function_composition_tests.jl`: nested and trigonometric formula terms checked against closed-form derivatives

## Statistical Validation Features

- **Analytical Verification**: Hand-calculated SE verification across linear and GLM models
- **Closed-form oracles**: `validation/test_feature_matrix_oracle.jl` and `validation/test_elasticity_unit_definitions.jl` check estimates and SEs exactly
- **Robust SE Integration**: CovarianceMatrices.jl sandwich estimators (HC0-HC3) and clustered SEs
- **Specialized Cases**: Integer variables, elasticities, categorical mixtures, error propagation
- **Publication Grade**: All SE computations meet econometric publication standards