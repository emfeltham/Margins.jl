# R Comparison Study

Validation study comparing Margins.jl against R's `margins` package.

## Quick Start

### Correctness Validation (5K observations)

**Option 1: Automated Script (Recommended)**

From the Margins.jl root directory:

```bash
# Interactive version (prompts to verify factor levels)
bash test/r_compare/run_comparison.sh

# Or fully automated (assumes factor levels already correct)
bash test/r_compare/run_comparison_auto.sh
```

**Option 2: Using Make**

From the `test/r_compare/` directory:

```bash
cd test/r_compare
make all          # Complete correctness validation workflow
make performance  # Julia vs R performance comparison (5K dataset)
make help         # See all available targets
```

### Large-Scale Performance Testing (500K observations)

For production-scale performance comparison (Julia vs R):

```bash
cd test/r_compare
make performance-large  # Generate 500K dataset and benchmark Julia vs R
```

**Note**: R benchmarks on 500K observations may take significant time and memory. The workflow automatically runs both Julia and R for direct comparison at scale.

### Option 3: Manual Step-by-Step

From the Margins.jl root directory:

```bash
# 1. Generate shared data
julia --project=. test/r_compare/generate_data.jl

# 2. Fit model in Julia
julia --project=. test/r_compare/julia_model.jl

# 3. Update R factor levels in r_model.R based on output from step 1

# 4. Fit model in R
Rscript test/r_compare/r_model.R

# 5. Compare results
julia --project=. test/r_compare/compare_results.jl
```

## Files

### Correctness Validation Scripts:
- **`generate_data.jl`** - Generates 5K observation dataset for correctness validation
- **`julia_model.jl`** - Julia model estimation and marginal effects computation
- **`r_model.R`** - R model estimation and marginal effects computation
- **`compare_results.jl`** - Automated validation of coefficient and SE agreement
- **`performance_benchmark.jl`** - Julia performance benchmarks (5K dataset)
- **`r_benchmarks.R`** - R performance benchmarks (5K dataset)
- **`compare_performance.jl`** - Julia vs R performance comparison analysis (5K dataset)

### Large-Scale Performance Scripts:
- **`generate_data_large.jl`** - Generates 500K observation dataset
- **`performance_benchmark_large.jl`** - Julia performance benchmarks (500K dataset)
- **`r_benchmarks_large.R`** - R performance benchmarks (500K dataset)
- **`compare_performance_large.jl`** - Julia vs R performance comparison analysis (500K dataset)

### Automation:
- **`run_comparison.sh`** - Interactive bash script to run complete workflow
- **`run_comparison_auto.sh`** - Non-interactive bash script (fully automated)
- **`Makefile`** - Make targets for all workflows

### Data:
- **`r_comparison_data.csv`** - 5K dataset for correctness validation (Julia + R)
- **`r_comparison_data_large.csv`** - 500K dataset for performance testing (Julia + R)

## Key Points

1. **Identical Data**: Both languages load from the same CSV file
2. **Identical Processing**: Data type conversions mirror each other
3. **Critical**: Categorical factor levels MUST match exactly between Julia and R
4. **Model Complexity**: ~100+ terms with interactions and categorical variables

## Expected Outputs

### Julia Results:
- `julia_coefficients.csv` - Model coefficients and SEs
- `julia_apm.csv` - Adjusted predictions at profiles
- `julia_mem.csv` - Marginal effects at profiles (pairwise contrasts)
- `julia_aap.csv` - Average adjusted predictions
- `julia_ame.csv` - Average marginal effects (all variables)
- `julia_ame_age.csv` - AME for single variable
- `julia_ame_scenario.csv` - AME with scenario

### R Results (Correctness Validation):
- `r_coefficients.csv` - Model coefficients and SEs
- `r_apm.csv` - Adjusted predictions at profiles
- `r_mem.csv` - Marginal effects at profiles
- `r_aap.csv` - Average adjusted predictions
- `r_ame.csv` - Average marginal effects (all variables)
- `r_ame_age.csv` - AME for single variable
- `r_ame_scenario.csv` - AME with scenario
- `r_model_fit.rds` - R model object
- `r_results_complete.rds` - Complete results bundle

### Performance Benchmark Outputs:

**5K Dataset (`make performance`):**
- `julia_benchmarks.csv` - Julia timing and memory usage
- `r_benchmarks.rds` - R timing data (microbenchmark format)
- `performance_comparison.csv` - Speedup ratios and comparison metrics

**500K Dataset (`make performance-large`):**
- `julia_benchmarks_large.csv` - Julia timing and memory usage
- `r_benchmarks_large.rds` - R timing data (microbenchmark format)
- `performance_comparison_large.csv` - Speedup ratios, comparison metrics, and scaling analysis

## Automated Validation

The `compare_results.jl` script automatically validates:
- Coefficient count matches (65 terms in both models ✓)
- Term names standardized (handles Julia vs R naming conventions)
- Estimates agree within realistic numerical precision tolerances
- Standard errors agree within expected cross-platform differences

**Validation Criteria:**
- SUCCESS: Max coefficient relative error < 0.01%, Max SE relative error < 1%
- PASS: Max coefficient relative error < 0.1%, Max SE relative error < 5%

**Current Results (5,000 observation dataset):**
- ✓✓✓ SUCCESS achieved
- Coefficient agreement: 0.0028% max relative error
- Standard error agreement: 0.04% max relative error
- Statistical equivalence confirmed

The script provides detailed quality metrics, worst-case mismatches, and a clear verdict.

## Performance Testing

### Two Performance Workflows

**1. Comparative Benchmark (5K observations)**
- Compares Julia vs R on identical 5K dataset
- Run: `make performance`
- Outputs: `julia_benchmarks.csv`, `r_benchmarks.rds`, `performance_comparison.csv`
- Demonstrates relative speedup between implementations

**2. Large-Scale Benchmark (500K observations)**
- Compares Julia vs R on production-scale dataset
- Run: `make performance-large`
- Outputs: `julia_benchmarks_large.csv`, `r_benchmarks_large.rds`, `performance_comparison_large.csv`
- Demonstrates O(1) profile margins scaling, O(n) population margins scaling
- Compares speedup ratios across dataset sizes (5K vs 500K)
- Analyzes whether Julia's advantage increases/decreases/maintains at scale
- **Note**: R benchmarks may take considerable time on large dataset (expect 5-30 minutes depending on hardware)

### Performance Comparison Metrics

The `compare_performance.jl` and `compare_performance_large.jl` scripts provide:

**Per-Operation Analysis:**
- Speedup ratio (R time / Julia time)
- Memory usage ratio
- Time improvement percentage
- Detailed breakdown for each operation (APM, MEM, AAP, AME, etc.)

**Summary Statistics:**
- Average speedup across all operations
- Median speedup
- Range (min to max speedup)
- Overall verdict (Exceptional/Excellent/Strong/Competitive)

**Scaling Analysis (500K only):**
- Cross-scale comparison (5K vs 500K results)
- Whether speedup increases/decreases/maintains at larger N
- Identification of operations where Julia's advantage grows

### Expected Performance Characteristics

**Profile margins** (APM, MEM):
- O(1) complexity - independent of dataset size
- Reference grid evaluation only
- Constant time regardless of N
- Julia speedup typically consistent across scales

**Population margins** (AAP, AME):
- O(n) complexity - linear scaling with dataset size
- Zero-allocation per-row computation in Julia
- Efficient memory usage even at large N
- Julia speedup may increase at larger N due to better memory efficiency

## Manual Validation Checks

1. **Marginal effects comparison**: Compare corresponding CSV files
   - Point estimates should agree within 0.0001% relative error
   - Standard errors should agree within 0.001% relative error

2. **Performance**: Julia should show superior scaling on large datasets

## Troubleshooting

**Problem**: Coefficients don't match
- **Solution**: Check that categorical factor levels match exactly. Run `generate_data.jl` and verify the printed levels match the `factor()` calls in `r_model.R`.

**Problem**: Different number of coefficients
- **Solution**: Verify that both formulas expand to the same terms. Check for missing interaction terms or categorical levels.

**Problem**: R crashes or runs out of memory on large dataset
- **Solution**: The 500K dataset benchmarks are designed to work with R, but may require significant RAM (16GB+ recommended). If R fails, you can still run Julia benchmarks only using `julia --project=. performance_benchmark_large.jl`.
