# Margins.jl vs R marginaleffects Performance Comparison

Performance benchmarking comparing Julia's Margins.jl against R's **marginaleffects** package - the current state-of-the-art R implementation for marginal effects computation (~1000× faster than the older `margins` package).

## Overview

This directory contains a complete benchmark suite comparing Margins.jl performance against `marginaleffects`, the fastest and most modern R package for marginal effects analysis. Unlike the comparison in `../r_compare/` which benchmarks against the older `margins` package, this comparison tests against the true state of the art in R.

### Why This Matters

The `marginaleffects` package (Arel-Bundock et al., 2024):
- Is ~1000× faster than the older `margins` package
- Supports over 100 model types
- Implements sophisticated optimizations
- Represents the current best practice in R

Comparing against `marginaleffects` provides a fair and rigorous test of Margins.jl's zero-allocation architecture against R's best-in-class implementation.

## Prerequisites

### R Packages
```r
install.packages(c(
  "marginaleffects",  # State-of-the-art marginal effects
  "tidyverse",        # Data manipulation
  "microbenchmark",   # Timing
  "profmem"           # Memory profiling
))
```

### Julia Packages
All required packages are specified in `Project.toml`. Install with:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start

### Option 1: Makefile (Recommended)
```bash
# Run 5K dataset benchmark
make performance

# Run 500K dataset benchmark (may take 30+ minutes)
make performance-large

# Show all available targets
make help
```

### Option 2: Shell Scripts
```bash
# 5K dataset
bash run_performance_marginaleffects.sh

# 500K dataset (with confirmation prompt)
bash run_performance_marginaleffects_large.sh
```

### Option 3: Manual Steps
```bash
# Generate data
julia --project=. generate_data.jl

# Run Julia benchmarks
julia --project=. performance_benchmark.jl

# Run R marginaleffects benchmarks
Rscript r_marginaleffects_benchmarks.R

# Compare performance
julia --project=. compare_performance_marginaleffects.jl
```

## File Structure

### Core Benchmark Scripts

**R Scripts:**
- `r_marginaleffects_benchmarks.R` - Performance benchmarks using marginaleffects (5K dataset)
- `r_marginaleffects_benchmarks_large.R` - Large-scale benchmarks (500K dataset)

**Julia Scripts (symlinked from ../r_compare):**
- `generate_data.jl` - Creates reproducible test datasets
- `generate_data_large.jl` - Creates 500K observation dataset
- `performance_benchmark.jl` - Julia benchmarks (5K)
- `performance_benchmark_large.jl` - Julia benchmarks (500K)

**Comparison Scripts:**
- `compare_performance_marginaleffects.jl` - Analyzes and reports speedup ratios (5K)
- `compare_performance_marginaleffects_large.jl` - Large-scale comparison (500K)

### Automation
- `Makefile` - Make targets for automated workflows
- `run_performance_marginaleffects.sh` - Shell script for 5K benchmark
- `run_performance_marginaleffects_large.sh` - Shell script for 500K benchmark

### Output Files (Generated)
- `r_comparison_data.csv` / `r_comparison_data_large.csv` - Shared datasets
- `julia_benchmarks.csv` / `julia_benchmarks_large.csv` - Julia timing/memory
- `r_marginaleffects_benchmarks.rds` / `r_marginaleffects_benchmarks_large.rds` - R results
- `performance_comparison_marginaleffects.csv` / `performance_comparison_marginaleffects_large.csv` - Final comparison

## Operations Benchmarked

All benchmarks compare identical operations across Julia and R:

1. **APM** (Adjusted Predictions at Profiles)
   - Julia: `profile_margins(..., type=:predictions)`
   - R: `predictions(model, newdata = datagrid(...))`
   - O(1) complexity - independent of dataset size

2. **MEM** (Marginal Effects at Profiles)
   - Julia: `profile_margins(..., type=:effects)`
   - R: `slopes(model, newdata = datagrid(...), variables = ...)`
   - O(1) complexity

3. **AAP** (Average Adjusted Predictions)
   - Julia: `population_margins(..., type=:predictions)`
   - R: `predictions(model)`
   - O(n) complexity

4. **AME (all)** (Average Marginal Effects - all variables)
   - Julia: `population_margins(..., type=:effects)`
   - R: `slopes(model)`
   - O(n) complexity

5. **AME (single)** (Single variable marginal effect)
   - Julia: `population_margins(..., vars=[:age_h])`
   - R: `slopes(model, variables = "age_h")`
   - O(n) complexity

6. **AME (scenario)** (Marginal effects at counterfactual)
   - Julia: `population_margins(..., scenarios=(x=val,))`
   - R: `slopes(model, newdata = datagrid(x = val))`
   - O(n) complexity

## marginaleffects API Translation

Key differences from the `margins` package (used in `../r_compare/`):

### Marginal Effects
```r
# margins (old):
margins(model)

# marginaleffects (new):
slopes(model)
```

### Predictions
```r
# margins (old):
prediction(model)

# marginaleffects (new):
predictions(model)
```

### Reference Grids/Profiles
```r
# margins (old):
prediction(model, at = list(x = c(1, 2), y = c(TRUE, FALSE)))

# marginaleffects (new):
predictions(model, newdata = datagrid(x = c(1, 2), y = c(TRUE, FALSE)))
```

### Marginal Effects at Profiles
```r
# emmeans (used in r_compare):
emtrends(model, ~ x, var = "z", at = list(x = c(1, 2)))

# marginaleffects (new):
slopes(model, variables = "z", newdata = datagrid(x = c(1, 2)))
```

## Expected Performance Characteristics

### Hypothesis
`marginaleffects` is highly optimized but still fundamentally O(n) constrained by design matrix materialization:

**Profile Operations (APM, MEM):**
- Expected: Modest Julia speedup (2-5×)
- Rationale: Both are O(1), but Julia's compiled evaluators avoid overhead

**Population Operations (AME, AAP):**
- Expected: Moderate Julia speedup (2-10×)
- Rationale: Both scale O(n), but zero-allocation gives edge
- Memory: Julia should show dramatic advantage (10-100×)

**Key Architectural Difference:**
- `marginaleffects`: Optimized O(n) design matrix approach
- Margins.jl: Zero-allocation O(1) compiled evaluators
- Scaling: Julia's advantage grows with dataset size

## Interpreting Results

The comparison script provides detailed analysis with interpretation categories:

- **EXCEPTIONAL** (≥10× speedup): Major performance advantage
- **EXCELLENT** (5-10× speedup): Substantial gains
- **STRONG** (2-5× speedup): Notable improvement
- **COMPETITIVE** (1.2-2× speedup): Solid performance
- **COMPARABLE** (<1.2× speedup): Similar performance

### What to Look For

1. **Speed Ratios**: How much faster is Julia?
2. **Memory Ratios**: How much less memory does Julia use?
3. **Scaling**: Do advantages grow with dataset size (5K → 500K)?
4. **Operation Type**: Are O(1) operations (APM, MEM) faster than O(n) (AME, AAP)?

## Model Specification

Benchmarks use a realistic logistic regression model with:
- **65 parameters** across multiple model types
- **Binary outcome** (binomial family, logit link)
- **Interactions**: Binary×continuous, categorical×continuous
- **Categorical variables**: Relation (5 levels), Religion (4 levels)
- **Continuous variables**: ~20 predictors including transformations

This complexity ensures benchmarks test realistic statistical workflows, not toy examples.

## Troubleshooting

### R Package Installation Issues
```r
# If marginaleffects installation fails:
install.packages("marginaleffects", dependencies = TRUE)

# Check version (should be ≥0.18.0 for best performance)
packageVersion("marginaleffects")
```

### Julia Package Issues
```bash
# Reset Julia environment
rm -rf Manifest.toml
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Memory Issues on Large Dataset
The 500K dataset benchmark can be memory-intensive for R:
- Recommended: 16GB+ RAM
- Monitor R memory usage: `pryr::mem_used()`
- If R crashes: Reduce dataset size in `generate_data_large.jl`

### Long Benchmark Times
Expected times (approximate, M1/M2 Mac or similar):

**5K Dataset:**
- Julia: <1 minute
- R marginaleffects: 2-5 minutes
- Total: ~5-10 minutes

**500K Dataset:**
- Julia: 2-5 minutes
- R marginaleffects: 20-60 minutes
- Total: 30-90 minutes

## Paper Integration

After running benchmarks, update the paper with results:

### Add to `paper/notes/motivation.md`

After line 17 (the 617MB marginaleffects claim), add:

```markdown
Direct benchmarks comparing Margins.jl against marginaleffects on a 500,000-observation
logistic model with 65 parameters show [X]× speedup for average marginal effects
computation and [Y]× memory reduction, demonstrating that even state-of-the-art R
implementations remain constrained by O(n) design matrix materialization overhead.
```

### Create Comparison Table

| Operation | vs margins | vs marginaleffects | Architecture |
|-----------|------------|-------------------|--------------|
| AME (500K obs) | 293× faster | [X]× faster | O(1) vs O(n) |
| Memory (AME) | 99.97% reduction | [Y]% reduction | Zero-alloc vs materialization |

Fill in [X] and [Y] with actual benchmark results.

## References

- Arel-Bundock, V., Greifer, N., & Heiss, A. (2024). "How to Interpret Statistical Models Using marginaleffects for R and Python." *Journal of Statistical Software*, 111(9), 1-32.
- Leeper, T. J. (2021). "Interpreting Regression Results using Average Marginal Effects with R's margins." *Available at SSRN*.

## Notes

- This comparison focuses on **performance** (speed and memory)
- Correctness validation is secondary (both packages compute identical results from same glm model)
- The key insight: marginaleffects is fast, but Julia's zero-allocation architecture provides fundamental advantages
- For full correctness validation workflow, see `../r_compare/` directory

## Clean Up

Remove generated files:
```bash
make clean         # Remove 5K dataset files
make clean-large   # Remove 500K dataset files
```
