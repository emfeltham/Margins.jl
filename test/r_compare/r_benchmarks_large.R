# r_benchmarks_large.R
# R performance benchmarks on large dataset (500K observations)
# WARNING: This may take significant time and memory

library(tidyverse)
library(margins)
library(emmeans)
library(microbenchmark)
library(profmem)

cat("================================================================================\n")
cat("R LARGE-SCALE PERFORMANCE BENCHMARK (500K observations)\n")
cat("================================================================================\n\n")

cat("WARNING: This benchmark uses a 500K observation dataset.\n")
cat("R may require significant time and memory.\n\n")

# Load large dataset
cat("Loading large dataset from CSV...\n")
df <- read_csv("r_comparison_data_large.csv", show_col_types = FALSE)
N <- nrow(df)
cat("  N =", N, "observations\n\n")

# Convert to proper data types
cat("Converting data types...\n")

# Boolean variables (from 0/1 to logical)
df <- df %>%
  mutate(
    response = as.logical(response),
    socio4 = as.logical(socio4),
    same_building = as.logical(same_building),
    kin431 = as.logical(kin431),
    coffee_cultivation = as.logical(coffee_cultivation),
    isindigenous_p = as.logical(isindigenous_p),
    man_p = as.logical(man_p),
    maj_catholic = as.logical(maj_catholic),
    maj_indigenous = as.logical(maj_indigenous)
  )

# Integer variables
df <- df %>%
  mutate(
    num_common_nbs = as.integer(num_common_nbs),
    age_p = as.integer(age_p),
    schoolyears_p = as.integer(schoolyears_p),
    population = as.integer(population),
    degree_h = as.integer(degree_h),
    degree_p = as.integer(degree_p)
  )

# Categorical variables
df <- df %>%
  mutate(
    perceiver = factor(perceiver),
    village_code = factor(village_code),
    relation = factor(relation),
    religion_c_p = factor(religion_c_p),
    religion_c_x = factor(religion_c_x),
    man_x = factor(man_x),
    isindigenous_x = factor(isindigenous_x)
  )

cat("  ✓ Done\n\n")

# Fit model
cat("Fitting model...\n")
cat("--------------------------------------------------------------------------------\n")
model_time <- system.time({
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
})
cat("Model fitting:", round(model_time["elapsed"], 3), "s\n")
cat("  K =", length(coef(model_r)), "parameters\n\n")

# Helper function to measure memory allocation
measure_memory <- function(expr) {
  p <- profmem(expr)
  total_bytes <- sum(p$bytes, na.rm = TRUE)
  total_bytes / 1024^2  # Convert to MB
}

# Performance Benchmarks
cat("================================================================================\n")
cat("RUNNING BENCHMARKS\n")
cat("================================================================================\n\n")

cat("NOTE: Using 3 samples instead of 5 due to large dataset size\n\n")

# 1. APM - Adjusted Predictions at Profiles
cat("1. APM (Adjusted Predictions at Profiles)\n")
apm <- microbenchmark(
  prediction(
    model_r,
    at = list(
      socio4 = c(FALSE, TRUE),
      are_related_dists_a_inv = c(1.0, 1/6)
    )
  ),
  times = 3
)
apm_mem <- measure_memory({
  prediction(
    model_r,
    at = list(
      socio4 = c(FALSE, TRUE),
      are_related_dists_a_inv = c(1.0, 1/6)
    )
  )
})
cat("   Time:", round(median(apm$time) / 1e9, 4), "s (median)\n")
cat("   Memory:", round(apm_mem, 2), "MB\n\n")

# 2. MEM - Marginal Effects at Profiles
# Use emmeans/emtrends for O(1) computation at specific grid points (matches Julia)
cat("2. MEM (Marginal Effects at Profiles - emtrends)\n")

# Reference grid specification
ref_grid_spec <- list(
  socio4 = c(FALSE, TRUE),
  are_related_dists_a_inv = c(1.0, 1/6),
  relation = "family",
  religion_c_p = "Catholic",
  same_building = FALSE,
  coffee_cultivation = FALSE,
  man_p = FALSE
)

# List of continuous variables to compute derivatives for
continuous_vars <- c("dists_p_inv", "dists_a_inv", "are_related_dists_a_inv",
                    "age_p", "wealth_d1_4_p", "schoolyears_p", "population",
                    "hhi_religion", "hhi_indigenous", "market",
                    "age_h", "age_h_nb_1_socio", "schoolyears_h",
                    "schoolyears_h_nb_1_socio", "wealth_d1_4_h",
                    "wealth_d1_4_h_nb_1_socio", "degree_a_mean", "degree_h",
                    "age_a_mean", "schoolyears_a_mean", "wealth_d1_4_a_mean")

mem <- microbenchmark(
  {
    # Compute derivative for each continuous variable at the reference grid
    for (var in continuous_vars) {
      emtrends(model_r, ~ socio4 + are_related_dists_a_inv,
               var = var,
               at = ref_grid_spec,
               type = "response")
    }
  },
  times = 3
)
mem_mem <- measure_memory({
  for (var in continuous_vars) {
    emtrends(model_r, ~ socio4 + are_related_dists_a_inv,
             var = var,
             at = ref_grid_spec,
             type = "response")
  }
})
cat("   Time:", round(median(mem$time) / 1e9, 4), "s (median)\n")
cat("   Memory:", round(mem_mem, 2), "MB\n")
cat("   Note: emtrends computes derivatives at grid points (O(1)), matching Julia\n\n")

# 3. AAP - Average Adjusted Predictions
cat("3. AAP (Average Adjusted Predictions)\n")
aap <- microbenchmark(
  prediction(model_r),
  times = 3
)
aap_mem <- measure_memory({
  prediction(model_r)
})
cat("   Time:", round(median(aap$time) / 1e9, 4), "s (median)\n")
cat("   Memory:", round(aap_mem, 2), "MB\n\n")

# 4. AME - Average Marginal Effects (all variables)
cat("4. AME (Average Marginal Effects - all variables)\n")
ame_all <- microbenchmark(
  margins(model_r),
  times = 3
)
ame_all_mem <- measure_memory({
  margins(model_r)
})
cat("   Time:", round(median(ame_all$time) / 1e9, 4), "s (median)\n")
cat("   Memory:", round(ame_all_mem, 2), "MB\n\n")

# 5. AME - Single variable
cat("5. AME (single variable: age_h)\n")
ame_age <- microbenchmark(
  margins(model_r, variables = "age_h"),
  times = 3
)
ame_age_mem <- measure_memory({
  margins(model_r, variables = "age_h")
})
cat("   Time:", round(median(ame_age$time) / 1e9, 4), "s (median)\n")
cat("   Memory:", round(ame_age_mem, 2), "MB\n\n")

# 6. AME with scenario
cat("6. AME (with scenario - wealth at are_related_dists_a_inv=1/6)\n")
ame_scenario <- microbenchmark(
  margins(
    model_r,
    variables = c("wealth_d1_4_p", "wealth_d1_4_h"),
    at = list(are_related_dists_a_inv = 1/6)
  ),
  times = 3
)
ame_scenario_mem <- measure_memory({
  margins(
    model_r,
    variables = c("wealth_d1_4_p", "wealth_d1_4_h"),
    at = list(are_related_dists_a_inv = 1/6)
  )
})
cat("   Time:", round(median(ame_scenario$time) / 1e9, 4), "s (median)\n")
cat("   Memory:", round(ame_scenario_mem, 2), "MB\n\n")

# Save benchmark results
benchmarks <- list(
  apm = apm,
  mem = mem,
  aap = aap,
  ame_all = ame_all,
  ame_age = ame_age,
  ame_scenario = ame_scenario,
  apm_mem = apm_mem,
  mem_mem = mem_mem,
  aap_mem = aap_mem,
  ame_all_mem = ame_all_mem,
  ame_age_mem = ame_age_mem,
  ame_scenario_mem = ame_scenario_mem
)
saveRDS(benchmarks, "r_benchmarks_large.rds")

cat("================================================================================\n")
cat("RESULTS SAVED\n")
cat("================================================================================\n\n")
cat("✓ r_benchmarks_large.rds\n\n")

# Summary
cat("================================================================================\n")
cat("SUMMARY\n")
cat("================================================================================\n\n")
cat("Dataset: N =", N, "observations\n")
cat("Total operation time:",
    round((median(apm$time) + median(mem$time) + median(aap$time) +
           median(ame_all$time) + median(ame_age$time) + median(ame_scenario$time)) / 1e9, 2),
    "s (R)\n\n")
cat("Next: Compare with Julia using compare_performance_large.jl\n\n")
