# R Comparison Study: Margins.jl vs marginaleffects

Planning file.

## Objective

Conduct a statistically rigorous comparison between Margins.jl and R's marginaleffects package using standard datasets. The goal is to validate agreement in estimates and delta-method standard errors under matched assumptions (model, scale, data, and covariance specification), and to document any systematic, theoretically justified differences.

## Dataset Selection

### 1. mtcars (Motor Trend Car Road Tests)
- Variables: 32 observations, 11 variables
- Continuous: mpg, disp, hp, drat, wt, qsec
- Categorical: cyl (4,6,8), vs (0,1), am (0,1), gear (3,4,5), carb (1,2,3,4,6,8)
- Rationale: Classic dataset with mixed variable types

### 2. iris (Edgar Anderson's Iris Data)
- Variables: 150 observations, 5 variables
- Continuous: Sepal.Length, Sepal.Width, Petal.Length, Petal.Width
- Categorical: Species (setosa, versicolor, virginica)
- Rationale: Multinomial outcomes
- Status: Skip for Margins.jl baseline comparison (multinomial not supported via GLM.jl). Keep for R-side checks if desired.

### 3. ToothGrowth (Tooth Growth in Guinea Pigs)
- Variables: 60 observations, 3 variables
- Continuous: len (tooth length)
- Categorical: supp (VC, OJ), dose (0.5, 1.0, 2.0; ordered factor or numeric)
- Rationale: Simple factorial design, interactions

### 4. Titanic (Survival on the Titanic)
- Variables: 2201 observations, 4 variables (expanded from table)
- Binary outcome: Survived (Yes/No)
- Categorical: Class (1st, 2nd, 3rd, Crew), Sex (Male, Female), Age (Child, Adult)
- Rationale: Logistic regression with multiple categoricals

## Model Scenarios (5 per dataset)

### mtcars Scenarios
1. mpg ~ wt + hp
2. mpg ~ factor(cyl) + wt
3. mpg ~ wt * factor(am) + hp
4. mpg ~ factor(cyl) + factor(vs) + factor(am)
5. mpg ~ wt * hp + factor(cyl) * factor(am)

### iris Scenarios (R-only, multinomial logistic)
1. Species ~ Sepal.Length + Sepal.Width
2. Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width
3. Species ~ Sepal.Length * Sepal.Width + Petal.Length
4. Species ~ Sepal.Length + I(Sepal.Length^2) + Petal.Width
5. Species ~ (Sepal.Length + Petal.Length) * (Sepal.Width + Petal.Width)

### ToothGrowth Scenarios
1. len ~ supp + dose_factor
2. len ~ supp + dose (treat dose as numeric)
3. len ~ supp * dose_factor
4. len ~ supp + dose_ordered (ordered factor)
5. len ~ supp + dose + I(dose^2)

### Titanic Scenarios (Logistic regression)
1. Survived ~ Sex + Class
2. Survived ~ Class + Sex + Age
3. Survived ~ Class * Sex + Age
4. Survived ~ Class * Sex * Age
5. Survived ~ (Class + Age) * Sex

## R Implementation Workflow

### Step 1: Data Preparation
```r
# Load required packages
library(marginaleffects)
library(nnet)    # For multinomial models (R-only)
library(MASS)    # Additional model functions
library(sandwich) # For robust vcov, if desired

# Prepare datasets with proper factor coding
prepare_datasets <- function() {
  # mtcars
  mtcars$cyl <- factor(mtcars$cyl)
  mtcars$vs  <- factor(mtcars$vs, labels = c("V", "Straight"))
  mtcars$am  <- factor(mtcars$am, labels = c("Automatic", "Manual"))
  mtcars$gear <- factor(mtcars$gear)
  mtcars$carb <- factor(mtcars$carb)

  # ToothGrowth
  ToothGrowth$supp <- factor(ToothGrowth$supp)
  ToothGrowth$dose_factor  <- factor(ToothGrowth$dose)
  ToothGrowth$dose_ordered <- ordered(ToothGrowth$dose)

  # Titanic (expand from table to individual observations)
  titanic_df <- as.data.frame(Titanic)
  titanic_expanded <- titanic_df[rep(seq_len(nrow(titanic_df)), titanic_df$Freq), 1:4]
  titanic_expanded$Survived <- factor(titanic_expanded$Survived)
  titanic_expanded$Class    <- factor(titanic_expanded$Class)
  titanic_expanded$Sex      <- factor(titanic_expanded$Sex)
  titanic_expanded$Age      <- factor(titanic_expanded$Age)

  return(list(
    mtcars = mtcars,
    iris = iris,
    toothgrowth = ToothGrowth,
    titanic = titanic_expanded
  ))
}
```

### Step 2: Model Estimation
```r
estimate_models <- function(datasets) {
  models <- list()

  # mtcars models (linear regression)
  models$mtcars <- list(
    m1 = lm(mpg ~ wt + hp, data = datasets$mtcars),
    m2 = lm(mpg ~ cyl + wt, data = datasets$mtcars),
    m3 = lm(mpg ~ wt * am + hp, data = datasets$mtcars),
    m4 = lm(mpg ~ cyl + vs + am, data = datasets$mtcars),
    m5 = lm(mpg ~ wt * hp + cyl * am, data = datasets$mtcars)
  )

  # iris models (multinomial, R-only)
  models$iris <- list(
    m1 = multinom(Species ~ Sepal.Length + Sepal.Width, data = iris, trace = FALSE),
    m2 = multinom(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris, trace = FALSE),
    m3 = multinom(Species ~ Sepal.Length * Sepal.Width + Petal.Length, data = iris, trace = FALSE),
    m4 = multinom(Species ~ Sepal.Length + I(Sepal.Length^2) + Petal.Width, data = iris, trace = FALSE),
    m5 = multinom(Species ~ (Sepal.Length + Petal.Length) * (Sepal.Width + Petal.Width), data = iris, trace = FALSE)
  )

  # toothgrowth models (linear regression)
  models$toothgrowth <- list(
    m1 = lm(len ~ supp + dose_factor, data = datasets$toothgrowth),
    m2 = lm(len ~ supp + dose, data = datasets$toothgrowth),
    m3 = lm(len ~ supp * dose_factor, data = datasets$toothgrowth),
    m4 = lm(len ~ supp + dose_ordered, data = datasets$toothgrowth),
    m5 = lm(len ~ supp + dose + I(dose^2), data = datasets$toothgrowth)
  )

  # titanic models (logistic regression)
  models$titanic <- list(
    m1 = glm(Survived ~ Sex + Class, data = datasets$titanic, family = binomial),
    m2 = glm(Survived ~ Class + Sex + Age, data = datasets$titanic, family = binomial),
    m3 = glm(Survived ~ Class * Sex + Age, data = datasets$titanic, family = binomial),
    m4 = glm(Survived ~ Class * Sex * Age, data = datasets$titanic, family = binomial),
    m5 = glm(Survived ~ (Class + Age) * Sex, data = datasets$titanic, family = binomial)
  )

  return(models)
}
```

### Step 3: Marginal Effects and Predictions
```r
# vcov_spec: NULL (model-based), string like "HC3", or a function(model) -> vcov matrix
compute_marginal_effects <- function(models, datasets, vcov_spec = NULL, type = "response") {
  vcov_arg <- vcov_spec
  results <- list()

  for (dataset_name in names(models)) {
    results[[dataset_name]] <- list()
    dataset <- datasets[[dataset_name]]

    for (model_name in names(models[[dataset_name]])) {
      model <- models[[dataset_name]][[model_name]]

      # Average Marginal Effects (AME)
      ame <- avg_slopes(model, vcov = vcov_arg, type = type)

      # Profile at means (APM/MEM): numeric means, factor mode (single row)
      means_data <- dataset[1, , drop = FALSE]
      for (col in names(dataset)) {
        if (is.numeric(dataset[[col]])) {
          means_data[[col]] <- mean(dataset[[col]], na.rm = TRUE)
        } else if (is.factor(dataset[[col]])) {
          means_data[[col]] <- names(sort(table(dataset[[col]]), decreasing = TRUE))[1]
        }
      }

      # MEM (slopes at means row)
      mem <- slopes(model, newdata = means_data, vcov = vcov_arg, type = type)

      # AAP/APM predictions
      aap <- avg_predictions(model, vcov = vcov_arg, type = type)
      apm <- predictions(model, newdata = means_data, vcov = vcov_arg, type = type)

      results[[dataset_name]][[model_name]] <- list(
        ame = ame,
        mem = mem,
        aap = aap,
        apm = apm,
        model_summary = summary(model)
      )
    }
  }

  return(results)
}
```

### Step 4: Export to CSV
```r
export_results <- function(results, datasets) {
  # Export datasets
  for (dataset_name in names(datasets)) {
    write.csv(datasets[[dataset_name]], file = paste0("data_", dataset_name, ".csv"), row.names = FALSE)
  }

  # Export marginal effects results
  for (dataset_name in names(results)) {
    for (model_name in names(results[[dataset_name]])) {
      result <- results[[dataset_name]][[model_name]]

      write.csv(as.data.frame(result$ame), file = paste0("r_results_", dataset_name, "_", model_name, "_ame.csv"), row.names = FALSE)
      write.csv(as.data.frame(result$mem), file = paste0("r_results_", dataset_name, "_", model_name, "_mem.csv"), row.names = FALSE)
      write.csv(as.data.frame(result$aap), file = paste0("r_results_", dataset_name, "_", model_name, "_aap.csv"), row.names = FALSE)
      write.csv(as.data.frame(result$apm), file = paste0("r_results_", dataset_name, "_", model_name, "_apm.csv"), row.names = FALSE)
    }
  }
}
```

## Julia Implementation Workflow

### Step 1: Data Loading and Preparation
```julia
using Margins, GLM, StatsModels, DataFrames, CSV, CategoricalArrays

function prepare_julia_datasets()
    datasets = Dict()

    # Load CSV files exported from R
    datasets["mtcars"] = CSV.read("data_mtcars.csv", DataFrame)
    datasets["iris"] = CSV.read("data_iris.csv", DataFrame)
    datasets["toothgrowth"] = CSV.read("data_toothgrowth.csv", DataFrame)
    datasets["titanic"] = CSV.read("data_titanic.csv", DataFrame)

    # Convert to proper Julia types
    # mtcars
    datasets["mtcars"].cyl = categorical(datasets["mtcars"].cyl)
    datasets["mtcars"].vs  = categorical(datasets["mtcars"].vs)
    datasets["mtcars"].am  = categorical(datasets["mtcars"].am)
    datasets["mtcars"].gear = categorical(datasets["mtcars"].gear)
    datasets["mtcars"].carb = categorical(datasets["mtcars"].carb)

    # iris
    datasets["iris"].Species = categorical(datasets["iris"].Species)

    # toothgrowth
    datasets["toothgrowth"].supp = categorical(datasets["toothgrowth"].supp)
    datasets["toothgrowth"].dose_factor  = categorical(datasets["toothgrowth"].dose_factor)
    datasets["toothgrowth"].dose_ordered = categorical(datasets["toothgrowth"].dose_ordered, ordered=true)

    # titanic
    datasets["titanic"].Survived = categorical(datasets["titanic"].Survived)
    datasets["titanic"].Class    = categorical(datasets["titanic"].Class)
    datasets["titanic"].Sex      = categorical(datasets["titanic"].Sex)
    datasets["titanic"].Age      = categorical(datasets["titanic"].Age)

    return datasets
end
```

### Step 2: Model Estimation
```julia
function estimate_julia_models(datasets)
    models = Dict()

    # mtcars models
    models["mtcars"] = Dict(
        "m1" => lm(@formula(mpg ~ wt + hp), datasets["mtcars"]),
        "m2" => lm(@formula(mpg ~ cyl + wt), datasets["mtcars"]),
        "m3" => lm(@formula(mpg ~ wt * am + hp), datasets["mtcars"]),
        "m4" => lm(@formula(mpg ~ cyl + vs + am), datasets["mtcars"]),
        "m5" => lm(@formula(mpg ~ wt * hp + cyl * am), datasets["mtcars"])
    )

    # iris models (R-only for now)
    # Julia GLM.jl lacks multinomial; skip for cross-package comparison.

    # toothgrowth models
    models["toothgrowth"] = Dict(
        "m1" => lm(@formula(len ~ supp + dose_factor), datasets["toothgrowth"]),
        "m2" => lm(@formula(len ~ supp + dose), datasets["toothgrowth"]),
        "m3" => lm(@formula(len ~ supp * dose_factor), datasets["toothgrowth"]),
        "m4" => lm(@formula(len ~ supp + dose_ordered), datasets["toothgrowth"]),
        "m5" => lm(@formula(len ~ supp + dose + dose^2), datasets["toothgrowth"])
    )

    # titanic models
    models["titanic"] = Dict(
        "m1" => glm(@formula(Survived ~ Sex + Class), datasets["titanic"], Binomial()),
        "m2" => glm(@formula(Survived ~ Class + Sex + Age), datasets["titanic"], Binomial()),
        "m3" => glm(@formula(Survived ~ Class * Sex + Age), datasets["titanic"], Binomial()),
        "m4" => glm(@formula(Survived ~ Class * Sex * Age), datasets["titanic"], Binomial()),
        "m5" => glm(@formula(Survived ~ (Class + Age) * Sex), datasets["titanic"], Binomial())
    )

    return models
end
```

### Step 3: Marginal Effects and Predictions
```julia
using CovarianceMatrices

function compute_julia_marginal_effects(models, datasets; cov::Union{Nothing,AbstractMatrix}=nothing)
    results = Dict()

    for (dataset_name, dataset_models) in models
        results[dataset_name] = Dict()
        dataset = datasets[dataset_name]

        for (model_name, model) in dataset_models
            # Average Marginal Effects (population)
            ame = population_margins(model, dataset; type=:effects, cov=cov)

            # Marginal Effects at Means (profile)
            # Note: means_grid uses sample means and frequency-weighted categoricals.
            # To exactly mimic R's single-row mode/mean approach, use cartesian_grid
            # with explicit baseline levels instead of means_grid.
            mem = profile_margins(model, dataset, means_grid(dataset); type=:effects, cov=cov)

            # Average Adjusted Predictions (response scale)
            aap = population_margins(model, dataset; type=:predictions, scale=:response, cov=cov)

            # Adjusted Predictions at Means (response scale)
            apm = profile_margins(model, dataset, means_grid(dataset); type=:predictions, scale=:response, cov=cov)

            results[dataset_name][model_name] = Dict(
                "ame" => DataFrame(ame),
                "mem" => DataFrame(mem),
                "aap" => DataFrame(aap),
                "apm" => DataFrame(apm),
            )
        end
    end

    return results
end

"""
compute_covariance(model) -> AbstractMatrix or nothing
Return a covariance matrix (e.g., HC3) compatible with Margins.jl, or nothing for model-based.
"""
function compute_covariance(model)
    try
        return vcov(HC3(), model)
    catch
        return nothing
    end
end
```

### Step 4: Export Julia Results
```julia
function export_julia_results(results)
    for (dataset_name, dataset_results) in results
        for (model_name, model_results) in dataset_results
            for (result_type, result_df) in model_results
                filename = "julia_results_$(dataset_name)_$(model_name)_$(result_type).csv"
                CSV.write(filename, result_df)
            end
        end
    end
end
```

## Comparison Analysis

### Step 1: Automated Comparison
```julia
function compare_results(; datasets = ["mtcars", "toothgrowth", "titanic"],
                        models = ["m1", "m2", "m3", "m4", "m5"],
                        result_types = ["ame", "mem", "aap", "apm"],
                        tol_est = 1e-8, tol_se = 1e-6)
    comparisons = Dict()

    for dataset in datasets
        comparisons[dataset] = Dict()

        for model in models
            comparisons[dataset][model] = Dict()

            for result_type in result_types
                # Load R and Julia results
                r_file = "r_results_$(dataset)_$(model)_$(result_type).csv"
                julia_file = "julia_results_$(dataset)_$(model)_$(result_type).csv"

                if isfile(r_file) && isfile(julia_file)
                    r_results = CSV.read(r_file, DataFrame)
                    julia_results = CSV.read(julia_file, DataFrame)

                    comparison = compare_estimates(r_results, julia_results; tol_est, tol_se)
                    comparisons[dataset][model][result_type] = comparison
                end
            end
        end
    end

    return comparisons
end

function compare_estimates(r_df, julia_df; tol_est = 1e-8, tol_se = 1e-6)
    # Heuristic column detection
    term_r = hasproperty(r_df, :term) ? :term : (hasproperty(r_df, :variable) ? :variable : nothing)
    term_j = hasproperty(julia_df, :term) ? :term : (hasproperty(julia_df, :variable) ? :variable : nothing)
    est_r = hasproperty(r_df, :estimate) ? :estimate : (hasproperty(r_df, :dydx) ? :dydx : nothing)
    se_r  = hasproperty(r_df, Symbol("std.error")) ? Symbol("std.error") : (hasproperty(r_df, :std_error) ? :std_error : nothing)
    est_j = hasproperty(julia_df, :estimate) ? :estimate : (hasproperty(julia_df, :dydx) ? :dydx : nothing)
    se_j  = hasproperty(julia_df, :se) ? :se : (hasproperty(julia_df, :std_error) ? :std_error : nothing)

    if term_r === nothing || term_j === nothing || est_r === nothing || est_j === nothing || se_r === nothing || se_j === nothing
        return Dict("error" => "unrecognized column structure")
    end

    left = rename!(select(r_df, term_r, est_r, se_r), [ :term, :estimate, :se ])
    right = rename!(select(julia_df, term_j, est_j, se_j), [ :term, :estimate_j, :se_j ])
    merged = innerjoin(left, right, on = :term)

    diffs = DataFrame(
        term = merged.term,
        est_diff = abs.(merged.estimate .- merged.estimate_j),
        se_diff = abs.(merged.se .- merged.se_j),
        est_ok = abs.(merged.estimate .- merged.estimate_j) .<= tol_est,
        se_ok = abs.(merged.se .- merged.se_j) .<= tol_se,
    )

    return Dict(
        "n_r" => nrow(r_df),
        "n_julia" => nrow(julia_df),
        "n_match" => nrow(merged),
        "n_est_ok" => count(identity, diffs.est_ok),
        "n_se_ok" => count(identity, diffs.se_ok),
        "details" => diffs,
    )
end
```

## Key Challenges and Solutions

### 1. Factor Level Ordering
- Challenge: R and Julia may order factor levels differently.
- Solution: Explicitly set factor levels in both environments to match.
- Validation: Check that contrasts use the same baseline categories.

### 2. Boolean Variable Handling
- Challenge: Margins.jl treats Bool as categorical (discrete contrast), whereas R can treat 0/1 as numeric unless factored.
- Solution: In R, coerce booleans to factor when a discrete contrast is intended. Verify both packages compute discrete contrasts for boolean variables.
- Testing: Include models with boolean predictors and verify contrast interpretation.

### 3. Categorical Mixture Handling
- Challenge: Margins.jl supports frequency-weighted/fractional categorical specifications in profiles; marginaleffects does not natively support fractional level mixtures in a single row.
- Solution: For apples-to-apples comparisons of MEM/APM, either: (a) use the R typical row (mode for factors) and, in Julia, a matching cartesian_grid reflecting the same single level; or (b) document the conceptual difference when using means_grid in Julia (mixtures) vs single-level in R.
- Extension: Test Margins.jl's mixture features separately and document methodological differences.

### 4. Missing Data
- Challenge: Different handling of missing values.
- Solution: Use complete cases only for initial comparisons.
- Validation: Ensure both packages use identical observations.

### 5. Numerical Precision
- Challenge: Small differences in numerical computation across libraries.
- Solution: Use strict tolerances; start with 1e-8 (estimates) and 1e-6 (SEs). Investigate any exceedance; do not mask via looser thresholds.
- Reporting: Flag differences above threshold for investigation with model matrix and gradient checks.

### 6. Model Specification Differences
- Challenge: Formula syntax differences between R and Julia.
- Solution: Carefully verify model matrices are identical (coefficient equality within tolerance) before comparing marginal effects.
- Validation: Compare coefficient estimates; if coefficients differ materially, halt comparison and diagnose.

### 7. Covariance Matrix Parity (SEs)
- Challenge: Standard errors must be computed with the same covariance estimator.
- Solution: In R, pass vcov = "HC3" or a sandwich::vcovHC function; in Julia, compute cov = vcov(HC3(), model) with CovarianceMatrices.jl and pass cov into Margins.jl calls. Never mix model-based and robust SEs between languages.
- Validation: Verify equality of the supplied covariance matrices' diagonals mapped to parameter order.

## Expected Deliverables

### Phase 1: Basic Comparison
1. CSV datasets: 4 prepared datasets with proper factor coding
2. R results: 60 CSV files (3 datasets x 5 models x 4 result types) — skipping iris for cross-package
3. Julia results: Corresponding CSV files from Margins.jl
4. Comparison report: Statistical summary of differences found

### Phase 2: Advanced Features
1. Elasticity comparison: Test Margins.jl's elasticity features against R (where applicable)
2. Categorical mixtures: Test Margins.jl's categorical mixture capabilities (R may lack parity)
3. Performance benchmarks: Compare computation speed across packages

### Phase 3: Documentation
1. Methodological differences: Document any systematic differences found
2. Recommendations: Guidelines for users migrating between packages
3. Validation report: Statistical validation of Margins.jl against established R package

## Success Criteria

1. Estimates Agreement: Estimates agree within 1e-8 absolute/relative tolerance for simple models
2. SE Agreement: Standard errors agree within 1e-6 when using matched covariance estimators
3. Categorical Handling: Boolean and factor variables produce identical contrasts under matched level choices
4. Scale Consistency: Link/response scale results are matched explicitly (type in R; scale in Julia)
5. Edge Case Robustness: Both packages handle unusual specifications consistently; if not, document and error rather than approximate

## Alignment Notes (Critical)

- Backend policy: Margins.jl uses AD by default at the API. Do not silently fall back to FD; if AD is unavailable, error out.
- Delta-method SEs: Both environments must use full covariance matrices with no independence assumptions. Supply cov/vcov explicitly to ensure parity when using robust estimators.
- Profiles at means: Margins.jl means_grid uses sample means and frequency-weighted categoricals; R typical rows use single category modes. Decide which convention to test and document it in the report.
- Scale: For GLMs, set type = "response" in R to match scale = :response in Julia when comparing response-scale effects/predictions. On link scale, set type = "link" and scale = :link.

