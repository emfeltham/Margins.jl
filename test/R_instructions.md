# R Validation Study: Margins.jl vs margins Package

## Objective

Conduct a comprehensive validation study comparing Margins.jl against the R `margins` package to verify:
1. **Statistical accuracy**: Point estimates and standard errors match within numerical precision
2. **Performance**: Computational efficiency across dataset sizes
3. **Feature parity**: Coverage of key marginal effects specifications

## Data Preparation

### 1. Generate Shared Data File

**File**: `test/r_compare/generate_data.jl`

Generate synthetic data once and export to CSV. Both Julia and R will load from this identical CSV file:

```bash
julia --project=. test/r_compare/generate_data.jl
```

This script:
- Generates 310K rows of synthetic data (seed = 08540)
- Exports to `test/r_compare/r_comparison_data.csv`
- Documents categorical variable levels for R factor conversion
- Lists all data types for verification

**CRITICAL**: Record the categorical variable levels printed by this script. You will need them to update the R script's factor conversions.

### 2. Fit Model in Julia

**File**: `test/r_compare/julia_model.jl`

Loads data from CSV and fits the R-compatible complex model:

```bash
julia --project=. test/r_compare/julia_model.jl
```

This script:
- Loads `r_comparison_data.csv` (same file R uses)
- Converts data types (Bool, CategoricalArray, Int) to match Julia expectations
- Fits the complex logistic regression model
- Computes all marginal effects (APM, MEM, AAP, AME variants)
- Exports results to CSV for comparison

### 3. Fit Model in R

**File**: `test/r_compare/r_model.R`

Loads the same CSV and fits the identical model:

```bash
Rscript test/r_compare/r_model.R
```

This script:
- Loads `r_comparison_data.csv` (same file Julia uses)
- Converts data types (logical, factor, integer) with **identical processing** to Julia
- **IMPORTANT**: Update factor levels based on `generate_data.jl` output before running
- Fits the same logistic regression model
- Computes matching marginal effects
- Exports results and benchmarks for comparison

**Export to CSV** with proper handling of:
- **Categorical variables**: Export as strings with level information (`perceiver`, `village_code`, `relation`, `religion_c_p`, `man_x`, `religion_c_x`, `isindigenous_x`)
  - Use `describe(df)` and `levels(df[!, variable])` in Julia to document reference levels
  - Must ensure R factors use identical reference levels for coefficient matching
- **Boolean variables**: Export as integers (0/1) for cross-language compatibility (`response`, `kin431`, `same_building`, `socio4`, `isindigenous_p`, `man_p`, `coffee_cultivation`, `maj_catholic`, `maj_indigenous`)
- **Continuous variables**: Full precision for numerical accuracy
- **Inverse distance variables**: Pre-computed (`dists_a_inv`, `dists_p_inv`, `are_related_dists_a_inv`)

Save to: `test/r_compare/r_comparison_data.csv`

## Model Specification

### R-Compatible Complex Logistic Regression

**Response**: `response` (binary, Bernoulli with logit link)

**Julia Formula** (from `test/r_compare/julia_model.jl`):
```julia
@formula(response ~
    socio4 + dists_p_inv + dists_a_inv + are_related_dists_a_inv +
    socio4 & (dists_p_inv + are_related_dists_a_inv + dists_a_inv) +
    (age_p + wealth_d1_4_p + schoolyears_p + man_p +
     same_building + population + hhi_religion + hhi_indigenous +
     coffee_cultivation + market) & (1 + socio4 + are_related_dists_a_inv) +
    relation + religion_c_p +
    relation & socio4 +
    religion_c_p & are_related_dists_a_inv +
    degree_a_mean + degree_h + age_a_mean + schoolyears_a_mean + wealth_d1_4_a_mean +
    age_h & age_h_nb_1_socio +
    schoolyears_h & schoolyears_h_nb_1_socio +
    wealth_d1_4_h & wealth_d1_4_h_nb_1_socio +
    (degree_a_mean + degree_h + age_a_mean + schoolyears_a_mean +
     wealth_d1_4_a_mean) & socio4 +
    hhi_religion & are_related_dists_a_inv +
    hhi_indigenous & are_related_dists_a_inv
)
```

**R Formula Translation**:

**Key Translation Rules**:
- Julia `&` → R `:` (interaction only)
- Julia `& (1 + x)` → R `* x` (main effects + interaction, since `*` expands to `+ + :`)
- Julia `1` in interaction context represents the intercept/main effect
- R will deduplicate any repeated main effects automatically

```r
response ~
  # Base effects
  socio4 + dists_p_inv + dists_a_inv + are_related_dists_a_inv +

  # Binary × continuous interactions (Julia: socio4 & (dists_p_inv + ...))
  socio4:(dists_p_inv + are_related_dists_a_inv + dists_a_inv) +

  # Individual-level variables with interactions
  # Julia: (age_p + ...) & (1 + socio4 + are_related_dists_a_inv)
  # R: Use * operator which expands to main effects + interactions
  (age_p + wealth_d1_4_p + schoolyears_p + man_p +
   same_building + population + hhi_religion + hhi_indigenous +
   coffee_cultivation + market) * (socio4 + are_related_dists_a_inv) +

  # Categorical variables
  relation + religion_c_p +
  relation:socio4 +
  religion_c_p:are_related_dists_a_inv +

  # Tie-level homophily measures (main effects)
  degree_a_mean + degree_h + age_a_mean + schoolyears_a_mean + wealth_d1_4_a_mean +

  # Continuous × continuous interactions (Julia uses &, R uses :)
  age_h:age_h_nb_1_socio +
  schoolyears_h:schoolyears_h_nb_1_socio +
  wealth_d1_4_h:wealth_d1_4_h_nb_1_socio +

  # Tie-level interactions with socio4
  # Julia: (degree_a_mean + ...) & socio4
  (degree_a_mean + degree_h + age_a_mean + schoolyears_a_mean +
   wealth_d1_4_a_mean):socio4 +

  # Composition effects (Julia uses &, R uses :)
  hhi_religion:are_related_dists_a_inv +
  hhi_indigenous:are_related_dists_a_inv
```

**Formula Equivalence Check**:
The R formula above should produce the same model matrix as the Julia formula. R's `*` operator will create duplicate main effects for `socio4` and `are_related_dists_a_inv`, but R automatically deduplicates them. The resulting model will have identical terms and coefficients.

**Model Complexity**:
- **50+ main effects** (including categorical expansions)
- **100+ total terms** when categorical variables are expanded
- **Multiple interaction types**: binary × continuous, continuous × continuous, categorical × continuous
- **R-compatible**: No Julia-specific operators that cannot be translated

## R Script Structure

### File: `test/r_compare/r_model.R`

#### Section 1: Environment Setup
```r
library(tidyverse)
library(margins)
library(broom)
library(microbenchmark)

# Load data
df <- read_csv("test/r_compare/r_comparison_data.csv")

# Convert to proper data types
df <- df %>%
  mutate(
    # Binary variables as logical
    response = as.logical(response),
    socio4 = as.logical(socio4),
    same_building = as.logical(same_building),
    kin431 = as.logical(kin431),
    coffee_cultivation = as.logical(coffee_cultivation),
    isindigenous_p = as.logical(isindigenous_p),
    man_p = as.logical(man_p),
    maj_catholic = as.logical(maj_catholic),
    maj_indigenous = as.logical(maj_indigenous),

    # Integer count variables
    num_common_nbs = as.integer(num_common_nbs),
    age_p = as.integer(age_p),
    schoolyears_p = as.integer(schoolyears_p),
    population = as.integer(population),
    degree_h = as.integer(degree_h),
    degree_p = as.integer(degree_p)
  )

# CRITICAL: Convert categorical variables to factors with EXACT reference levels as Julia
# Reference levels must match Julia's CategoricalArray levels (first level is reference)
# Use describe(df) and levels(df[!, :variable]) in Julia to determine these

# NOTE: Adjust factor levels based on actual data - these are placeholders
# Run in Julia first: levels(df[!, :relation]), levels(df[!, :religion_c_p]), etc.
df <- df %>%
  mutate(
    # perceiver: Keep as-is (unique IDs, reference doesn't matter for interpretation)
    perceiver = as.factor(perceiver),

    # village_code: Numeric levels, first observed becomes reference
    village_code = as.factor(village_code),

    # relation: MUST match Julia order exactly
    # Example: if Julia has ["family", "free_time", "friend", "neighbor", "work"]
    # Then reference level is "family"
    relation = factor(relation, levels = c("family", "free_time", "friend", "neighbor", "work")),

    # religion_c_p: MUST match Julia order exactly
    # Example: if Julia has ["Catholic", "No Religion", "Other", "Protestant"]
    # Then reference level is "Catholic"
    religion_c_p = factor(religion_c_p, levels = c("Catholic", "No Religion", "Other", "Protestant")),

    # religion_c_x: Mixed categorical (may contain "Catholic, Protestant" style values)
    religion_c_x = as.factor(religion_c_x),

    # man_x: Mixed categorical ("true", "false", "true, false")
    man_x = as.factor(man_x),

    # isindigenous_x: Mixed categorical
    isindigenous_x = as.factor(isindigenous_x)
  )

# VERIFICATION: Print factor levels to ensure they match Julia
cat("R Factor Levels:\n")
cat("relation:", levels(df$relation), "\n")
cat("religion_c_p:", levels(df$religion_c_p), "\n")
cat("village_code:", levels(df$village_code), "\n")

# Compare these to Julia output from:
# julia> levels(df[!, :relation])
# julia> levels(df[!, :religion_c_p])
# julia> levels(df[!, :village_code])
```

#### Section 2: Model Estimation
```r
# Fit logistic regression model matching Julia specification
model_r <- glm(
  response ~
    # Base effects
    socio4 + dists_p_inv + dists_a_inv + are_related_dists_a_inv +

    # Binary × continuous interactions
    socio4:(dists_p_inv + are_related_dists_a_inv + dists_a_inv) +

    # Individual-level variables with interactions
    (age_p + wealth_d1_4_p + schoolyears_p + man_p +
     same_building + population + hhi_religion + hhi_indigenous +
     coffee_cultivation + market) * (socio4 + are_related_dists_a_inv) +

    # Categorical variables
    relation + religion_c_p +
    relation:socio4 +
    religion_c_p:are_related_dists_a_inv +

    # Tie-level homophily measures
    degree_a_mean + degree_h + age_a_mean + schoolyears_a_mean + wealth_d1_4_a_mean +

    # Continuous × continuous interactions
    age_h:age_h_nb_1_socio +
    schoolyears_h:schoolyears_h_nb_1_socio +
    wealth_d1_4_h:wealth_d1_4_h_nb_1_socio +

    # Tie-level interactions with socio4
    (degree_a_mean + degree_h + age_a_mean + schoolyears_a_mean +
     wealth_d1_4_a_mean):socio4 +

    # Composition effects
    hhi_religion:are_related_dists_a_inv +
    hhi_indigenous:are_related_dists_a_inv,
  data = df,
  family = binomial(link = "logit")
)

# Verify coefficient count matches Julia model
cat("Number of coefficients:", length(coef(model_r)), "\n")

# Save model summary
summary(model_r)
write_rds(model_r, "test/r_compare/r_model_fit.rds")
```

#### Section 3: Marginal Effects Computation

Target specifications matching Julia implementation:

**3.1 Adjusted Predictions at Profiles (APM)**
```r
# Equivalent to: profile_margins(m, df, cg; type=:predictions)
# Reference grid: cartesian_grid(socio4 = [false, true], are_related_dists_a_inv = [1, 1/6])

margins_apm <- margins(
  model_r,
  at = list(
    socio4 = c(FALSE, TRUE),
    are_related_dists_a_inv = c(1.0, 1/6)
  ),
  type = "response"  # Predictions on response scale
)
summary(margins_apm)
```

**3.2 Marginal Effects at Profiles with Pairwise Contrasts (MEM)**
```r
# Equivalent to: profile_margins(m, df, cg; type=:effects, contrasts=:pairwise)
margins_mem <- margins(
  model_r,
  at = list(
    socio4 = c(FALSE, TRUE),
    are_related_dists_a_inv = c(1.0, 1/6)
  ),
  type = "link"  # Effects on link scale by default
)
summary(margins_mem)
```

**3.3 Average Adjusted Predictions (AAP)**
```r
# Equivalent to: population_margins(m, df; type=:predictions)
margins_aap <- margins(
  model_r,
  type = "response",
  variables = NULL  # All variables
)
summary(margins_aap)
```

**3.4 Average Marginal Effects (AME) - All Variables**
```r
# Equivalent to: population_margins(m, df; type=:effects)
margins_ame <- margins(
  model_r,
  type = "link"
)
summary(margins_ame)
```

**3.5 AME for Single Variable**
```r
# Equivalent to: population_margins(m, df; type=:effects, vars=[:age_h])
margins_ame_age <- margins(
  model_r,
  variables = "age_h",
  type = "link"
)
summary(margins_ame_age)
```

**3.6 AME with Scenarios**
```r
# Equivalent to: population_margins(m, df; type=:effects, vars=[:wealth_d1_4_p, :wealth_d1_4_h],
#                                     scenarios=(are_related_dists_a_inv=[1/6],))
margins_ame_scenario <- margins(
  model_r,
  variables = c("wealth_d1_4_p", "wealth_d1_4_h"),
  at = list(are_related_dists_a_inv = 1/6),
  type = "link"
)
summary(margins_ame_scenario)
```

#### Section 4: Performance Benchmarking
```r
# Benchmark key operations
benchmark_results <- microbenchmark(
  "GLM_fit" = {
    glm(response ~
          socio4 + dists_p_inv + dists_a_inv + are_related_dists_a_inv +
          socio4:(dists_p_inv + are_related_dists_a_inv + dists_a_inv) +
          (age_p + wealth_d1_4_p + schoolyears_p + man_p +
           same_building + population + hhi_religion + hhi_indigenous +
           coffee_cultivation + market) * (socio4 + are_related_dists_a_inv) +
          relation + religion_c_p + relation:socio4 + religion_c_p:are_related_dists_a_inv +
          degree_a_mean + degree_h + age_a_mean + schoolyears_a_mean + wealth_d1_4_a_mean +
          age_h:age_h_nb_1_socio + schoolyears_h:schoolyears_h_nb_1_socio +
          wealth_d1_4_h:wealth_d1_4_h_nb_1_socio +
          (degree_a_mean + degree_h + age_a_mean + schoolyears_a_mean +
           wealth_d1_4_a_mean):socio4 +
          hhi_religion:are_related_dists_a_inv + hhi_indigenous:are_related_dists_a_inv,
        data = df, family = binomial(link = "logit"))
  },
  "AME_all" = {
    margins(model_r, type = "link")
  },
  "AME_single" = {
    margins(model_r, variables = "age_h", type = "link")
  },
  "APM_profiles" = {
    margins(model_r, at = list(socio4 = c(FALSE, TRUE), are_related_dists_a_inv = c(1.0, 1/6)),
            type = "response")
  },
  times = 10
)

print(benchmark_results)
write_rds(benchmark_results, "test/r_compare/r_benchmarks.rds")
```

#### Section 5: Export Results
```r
# Export for cross-validation with Julia
results_list <- list(
  apm = summary(margins_apm),
  mem = summary(margins_mem),
  aap = summary(margins_aap),
  ame = summary(margins_ame),
  ame_age = summary(margins_ame_age),
  ame_scenario = summary(margins_ame_scenario),
  benchmarks = benchmark_results,
  model_summary = summary(model_r)
)

write_rds(results_list, "test/r_compare/r_results_complete.rds")

# Export to CSV for easy inspection
write_csv(as.data.frame(summary(margins_apm)), "test/r_compare/r_apm.csv")
write_csv(as.data.frame(summary(margins_mem)), "test/r_compare/r_mem.csv")
write_csv(as.data.frame(summary(margins_aap)), "test/r_compare/r_aap.csv")
write_csv(as.data.frame(summary(margins_ame)), "test/r_compare/r_ame.csv")
```

## Julia Comparison Script

### File: `test/r_compare/julia_comparison_study.jl`

```julia
using Margins, GLM, DataFrames, CSV, Statistics

# Load same data
df = CSV.read("test/r_compare/r_comparison_data.csv", DataFrame)

# Convert data types to match Julia expectations
# (Boolean columns from 0/1 to Bool, categorical strings to CategoricalArray, etc.)

# Fit identical model
fx = @formula(response ~ <JULIA_FORMULA>)
m = fit(GeneralizedLinearModel, fx, df, Bernoulli(), LogitLink())

# Compute marginal effects matching R specifications
cg = cartesian_grid(socio4 = [false, true], are_related_dists_a_inv = [1, 1/6])

apm_result = @time profile_margins(m, df, cg; type=:predictions)
mem_result = @time profile_margins(m, df, cg; type=:effects, contrasts=:pairwise)
aap_result = @time population_margins(m, df; type=:predictions)
ame_result = @time population_margins(m, df; type=:effects)
ame_age_result = @time population_margins(m, df; type=:effects, vars=[:age_h])
ame_scenario_result = @time population_margins(m, df; type=:effects,
                                                 vars=[:wealth_d1_4_p, :wealth_d1_4_h],
                                                 scenarios=(are_related_dists_a_inv=[1/6],))

# Export results
CSV.write("test/r_compare/julia_apm.csv", DataFrame(apm_result))
CSV.write("test/r_compare/julia_mem.csv", DataFrame(mem_result))
CSV.write("test/r_compare/julia_aap.csv", DataFrame(aap_result))
CSV.write("test/r_compare/julia_ame.csv", DataFrame(ame_result))
```

## Validation Analysis

### File: `test/r_compare/comparison_report.jl`

Generate comparative report:
1. **Coefficient comparison**: Model parameters match within tolerance (< 1e-8)
2. **Point estimate comparison**: Marginal effects estimates match (< 1e-6)
3. **Standard error comparison**: Delta-method SEs match (< 1e-6)
4. **Performance comparison**: Timing tables and speedup factors
5. **Diagnostic plots**: Scatter plots of Julia vs R estimates

## Workflow

### Option 1: Automated (Recommended)

Run the entire workflow with one command:

```bash
# From Margins.jl root directory
bash test/r_compare/run_comparison.sh
```

Or using Make:

```bash
cd test/r_compare
make all  > make.txt 2>&1
```

### Option 2: Manual Step-by-Step

1. **Generate data**: `julia --project=. test/r_compare/generate_data.jl`
   - Creates shared CSV file
   - Prints categorical levels (record these!)

2. **Run Julia analysis**: `julia --project=. test/r_compare/julia_model.jl`
   - Loads CSV, fits model, computes marginal effects
   - Exports Julia results

3. **Update R factor levels**: Edit `test/r_compare/r_model.R`
   - Update `relation` and `religion_c_p` factor levels based on step 1 output
   - Ensure levels match exactly

4. **Run R analysis**: `Rscript test/r_compare/r_model.R`
   - Loads same CSV, fits identical model
   - Exports R results and benchmarks

5. **Compare results**: `julia --project=. test/r_compare/compare_results.jl`
   - Automated verification of coefficient and SE agreement
   - Clear SUCCESS/FAILURE verdict with diagnostics

## Deliverables

All files in `test/r_compare/` directory:

### Scripts:
1. **Data generation**: `generate_data.jl` - creates shared CSV
2. **Julia model script**: `julia_model.jl` - fits model and computes marginal effects
3. **R model script**: `r_model.R` - fits identical model in R
4. **Comparison script**: `compare_results.jl` - automated validation of coefficient agreement
5. **Automation**: `run_comparison.sh` (interactive), `run_comparison_auto.sh` (non-interactive), `Makefile`

### Data and Results:
6. **Data file**: `r_comparison_data.csv` (~310K rows, used by both Julia and R)
7. **Julia results**: `julia_*.csv` files (coefficients, APM, MEM, AAP, AME variants)
8. **R results**: `r_*.csv` and `*.rds` files (coefficients, marginal effects, benchmarks)

## Success Criteria

- **Statistical validity**: Point estimates agree within 0.0001% relative error
- **Standard errors**: Delta-method SEs agree within 0.001% relative error
- **Performance**: Julia implementation demonstrates superior scaling on large datasets
- **Documentation**: Comprehensive comparison report suitable for package documentation
