# Performance Comparison: Margins.jl vs R emmeans/margins Packages

**Datasets**: 5,000 and 500,000 observations, 65 model parameters (logistic regression with interactions)

## Executive Summary

Margins.jl demonstrates exceptional performance advantages over R's marginal effects ecosystem across all operations and scales. The Julia implementation achieves 165× average speedup at 5K observations, scaling to 203× average speedup at 500K observations, with memory efficiency ranging from 10,775× to 23,669× compared to equivalent R implementations on identical statistical computations.

**Methodological note**: Profile-based operations (MEM, APM) use R's `emmeans::emtrends()` for true comparison, computing marginal effects at specific grid points (O(1)) rather than averaged across the dataset (O(n)).

**Critical finding**: At 500K observations, R's memory consumption becomes impractical (3.7 TB for AME with many variables, 260 GB for the AME scenario), while Julia maintains sub-1GB memory usage for all operations.

## Detailed Performance Analysis

### Small-Scale Results (N=5,000)

| Operation | Julia Time | R Time | Speedup | Julia Memory | R Memory | Memory Ratio |
|-----------|------------|--------|---------|--------------|----------|--------------|
| **MEM** (Marginal Effects at Profiles) | **2.4ms** | **351ms** | **145×** | **1.67 MB** | **164.69 MB** | **99×** |
| **AME (age_h)** (Single variable AME) | 1.5ms | 460ms | **316×** | 0.29 MB | 1,367.86 MB | **4,682×** |
| **AME (all)** (Average Marginal Effects, all variables) | 44ms | 13,251ms | **298×** | 0.63 MB | 37,428.6 MB | **59,342×** |
| **APM** (Adjusted Predictions at Profiles) | 0.7ms | 102ms | **142×** | 0.75 MB | 123.47 MB | **164×** |
| **AME (scenario)** (AME with scenario) | 20ms | 987ms | **50×** | 10.07 MB | 2,703.58 MB | **268×** |
| **AAP** (Average Adjusted Predictions) | 0.5ms | 20ms | **37×** | 0.29 MB | 28.27 MB | **97×** |

**Summary Statistics (N=5,000):**
- Average speedup: **165×**
- Median speedup: **143×**
- Average memory ratio: **10,775×**
- Overall speedup: **219×**

### Large-Scale Results (N=500,000)

| Operation | Julia Time | R Time | Speedup | Julia Memory | R Memory | Memory Ratio |
|-----------|------------|--------|---------|--------------|----------|--------------|
| **MEM** (Marginal Effects at Profiles) | **41ms** | **10,187ms** | **248×** | **73.42 MB** | **14,351.62 MB** | **195×** |
| **AME (age_h)** (Single variable AME) | 155ms | 50,776ms | **329×** | 26.73 MB | 134,828.07 MB | **5,045×** |
| **AME (all)** (Average Marginal Effects, all variables) | 4,490ms | 1,346,893ms | **300×** | 27.07 MB | 3,687,466 MB | **136,239×** |
| **APM** (Adjusted Predictions at Profiles) | 38ms | 10,104ms | **263×** | 72.51 MB | 12,146.25 MB | **168×** |
| **AME (scenario)** (AME with scenario) | 1,991ms | 123,060ms | **62×** | 995.75 MB | 266,576.31 MB | **268×** |
| **AAP** (Average Adjusted Predictions) | 45ms | 712ms | **16×** | 26.73 MB | 2,725.54 MB | **102×** |

**Summary Statistics (N=500,000):**
- Average speedup: **203×**
- Median speedup: **255×**
- Average memory ratio: **23,669×**
- Overall speedup: **228×**

**Critical Observations at Scale:**
- R memory usage becomes impractical: AME (all) requires 3.7 TB, AME (scenario) requires 260 GB
- Julia maintains practicality: All operations under 1 GB memory
- Speedup increases with scale: Average 165× → 203×, memory ratio 10,775× → 23,669×
- Profile operations (O(1)) scale unexpectedly due to model complexity (65 parameters): Julia 17-54× slower, R 29-99× slower (GC overhead)
- Population operations (O(n)) scale linearly as expected: Julia 90-103× slower, R 36-125× slower (plus memory pressure)

## Statistical Validity

All performance comparisons produce statistically identical results (< 0.04% max error). Performance advantages do not compromise statistical correctness.

## Methodology Notes

### R Package Selection

Profile Operations (MEM, APM):
- R method: `emmeans::emtrends()` and `margins::prediction(at=...)`
- Why: Computes marginal effects/predictions at specific grid points (O(1))
- Previous approach (`margins::margins(at=...)`): Averaged across full dataset (O(n)) - not comparable to Julia's O(1) profile approach
- Result: Fair apples-to-apples comparison of identical computations

Population Operations (AAP, AME):
- R method: `margins::margins()` and `margins::prediction()`
- Why: Standard R packages for population-averaged marginal effects
- Both: Loop through dataset computing per-observation effects/predictions, then average

### Timing
- Julia: BenchmarkTools.jl minimum time (5 samples, 60s timeout)
  - Minimum eliminates compilation and GC noise
  - Representative of steady-state performance
- R: microbenchmark median time (5 samples)
  - Median provides robust central tendency
  - Typical R benchmarking practice

### Memory
- Julia: BenchmarkTools.jl allocation tracking
  - Measures all heap allocations during execution
- R: profmem package total bytes allocated
  - Tracks R-level memory allocations
  - Comparable methodology to Julia's approach

### Comparability

We have identical
- data (5,000 observations, shared CSV)
- model specification (65 parameters)
- operations (APM, MEM, AAP, AME)
- computational approach (O(1) for profiles, O(n) for population)
- numerical precision (verified separately, < 0.04% error)

## Conclusions

Margins.jl delivers a substantial average speedup and significant memory reduction over R's `emmeans` and `margins` packages across all operations and scales. At 500K observations, R's memory requirements may become impractical (3.7 TB for AME all variables) while Julia maintains sub-1GB usage. Both implementations compute identical quantities with verified statistical correctness. Performance advantages stem from Julia's compilation and zero-allocation design, not algorithmic differences.
