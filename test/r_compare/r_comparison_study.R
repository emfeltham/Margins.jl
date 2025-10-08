# r_comparison_study.R
# R validation study comparing margins package to Margins.jl
# Loads data from Julia, fits identical model, computes marginal effects

# ==============================================================================
# Section 1: Environment Setup
# ==============================================================================

library(tidyverse)
library(margins)
library(broom)
library(microbenchmark)

cat("R Validation Study: Margins.jl vs margins package\n")
cat("==================================================\n\n")

# Load data
cat("Loading data from Julia export...\n")
df <- read_csv("test/r_compare/r_comparison_data.csv", show_col_types = FALSE)
cat("Loaded:", nrow(df), "rows,", ncol(df), "columns\n\n")

# Convert to proper data types
cat("Converting data types...\n")
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

# NOTE: Run the Julia script first and use the printed levels to update these factor() calls
# The levels below are placeholders - adjust based on Julia output
cat("Converting categorical variables to factors...\n")
cat("IMPORTANT: Verify these match Julia levels from r_compatible_model.jl output\n\n")

df <- df %>%
  mutate(
    # perceiver: Keep as-is (unique IDs, reference doesn't matter for interpretation)
    perceiver = as.factor(perceiver),

    # village_code: Numeric levels, first observed becomes reference
    village_code = as.factor(village_code),

    # relation: MUST match Julia order exactly
    # Update after running Julia script
    relation = factor(relation, levels = c("free_time", "work", "family", "neighbor", "friend")),

    # religion_c_p: MUST match Julia order exactly
    # Update after running Julia script
    religion_c_p = factor(religion_c_p, levels = c("Catholic", "Protestant", "No Religion", "Other")),

    # religion_c_x: Mixed categorical (may contain "Catholic, Protestant" style values)
    religion_c_x = as.factor(religion_c_x),

    # man_x: Mixed categorical ("true", "false", "true, false")
    man_x = as.factor(man_x),

    # isindigenous_x: Mixed categorical
    isindigenous_x = as.factor(isindigenous_x)
  )

# VERIFICATION: Print factor levels to ensure they match Julia
cat("R Factor Levels (verify these match Julia output):\n")
cat("relation:", levels(df$relation), "\n")
cat("religion_c_p:", levels(df$religion_c_p), "\n")
cat("village_code:", levels(df$village_code), "\n")
cat("\n")

# ==============================================================================
# Section 2: Model Estimation
# ==============================================================================

cat("Fitting logistic regression model...\n")
cat("This may take several minutes with 310K observations...\n\n")

model_fit_time <- system.time({
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

cat("Model fitted in", round(model_fit_time["elapsed"], 2), "seconds\n")

# Verify coefficient count matches Julia model
cat("Number of coefficients:", length(coef(model_r)), "\n\n")

# Save model summary
model_summary <- summary(model_r)
write_rds(model_r, "test/r_compare/r_model_fit.rds")
write_rds(model_summary, "test/r_compare/r_model_summary.rds")

# ==============================================================================
# Section 3: Marginal Effects Computation
# ==============================================================================

cat("Computing marginal effects...\n\n")

# 3.1 Adjusted Predictions at Profiles (APM)
cat("APM (Adjusted Predictions at Profiles)...\n")
apm_time <- system.time({
  margins_apm <- margins(
    model_r,
    at = list(
      socio4 = c(FALSE, TRUE),
      are_related_dists_a_inv = c(1.0, 1/6)
    ),
    type = "response"  # Predictions on response scale
  )
})
cat("  Completed in", round(apm_time["elapsed"], 2), "seconds\n")
apm_summary <- summary(margins_apm)

# 3.2 Marginal Effects at Profiles with Pairwise Contrasts (MEM)
cat("MEM (Marginal Effects at Profiles, pairwise contrasts)...\n")
mem_time <- system.time({
  margins_mem <- margins(
    model_r,
    at = list(
      socio4 = c(FALSE, TRUE),
      are_related_dists_a_inv = c(1.0, 1/6)
    ),
    type = "link"  # Effects on link scale
  )
})
cat("  Completed in", round(mem_time["elapsed"], 2), "seconds\n")
mem_summary <- summary(margins_mem)

# 3.3 Average Adjusted Predictions (AAP)
cat("AAP (Average Adjusted Predictions)...\n")
aap_time <- system.time({
  margins_aap <- margins(
    model_r,
    type = "response",
    variables = NULL  # All variables
  )
})
cat("  Completed in", round(aap_time["elapsed"], 2), "seconds\n")
aap_summary <- summary(margins_aap)

# 3.4 Average Marginal Effects (AME) - All Variables
cat("AME (Average Marginal Effects - all variables)...\n")
ame_time <- system.time({
  margins_ame <- margins(
    model_r,
    type = "link"
  )
})
cat("  Completed in", round(ame_time["elapsed"], 2), "seconds\n")
ame_summary <- summary(margins_ame)

# 3.5 AME for Single Variable
cat("AME (single variable - age_h)...\n")
ame_age_time <- system.time({
  margins_ame_age <- margins(
    model_r,
    variables = "age_h",
    type = "link"
  )
})
cat("  Completed in", round(ame_age_time["elapsed"], 2), "seconds\n")
ame_age_summary <- summary(margins_ame_age)

# 3.6 AME with Scenarios
cat("AME (with scenario - wealth variables at are_related_dists_a_inv=1/6)...\n")
ame_scenario_time <- system.time({
  margins_ame_scenario <- margins(
    model_r,
    variables = c("wealth_d1_4_p", "wealth_d1_4_h"),
    at = list(are_related_dists_a_inv = 1/6),
    type = "link"
  )
})
cat("  Completed in", round(ame_scenario_time["elapsed"], 2), "seconds\n")
ame_scenario_summary <- summary(margins_ame_scenario)

cat("\nAll marginal effects computed successfully!\n\n")

# ==============================================================================
# Section 4: Performance Benchmarking
# ==============================================================================

cat("Running performance benchmarks (10 iterations each)...\n")
cat("This will take several minutes...\n\n")

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

# ==============================================================================
# Section 5: Export Results
# ==============================================================================

cat("\nExporting results...\n")

# Export for cross-validation with Julia
results_list <- list(
  apm = apm_summary,
  mem = mem_summary,
  aap = aap_summary,
  ame = ame_summary,
  ame_age = ame_age_summary,
  ame_scenario = ame_scenario_summary,
  benchmarks = benchmark_results,
  model_summary = model_summary,
  timings = list(
    model_fit = model_fit_time["elapsed"],
    apm = apm_time["elapsed"],
    mem = mem_time["elapsed"],
    aap = aap_time["elapsed"],
    ame = ame_time["elapsed"],
    ame_age = ame_age_time["elapsed"],
    ame_scenario = ame_scenario_time["elapsed"]
  )
)

write_rds(results_list, "test/r_compare/r_results_complete.rds")

# Export to CSV for easy inspection
write_csv(as.data.frame(apm_summary), "test/r_compare/r_apm.csv")
write_csv(as.data.frame(mem_summary), "test/r_compare/r_mem.csv")
write_csv(as.data.frame(aap_summary), "test/r_compare/r_aap.csv")
write_csv(as.data.frame(ame_summary), "test/r_compare/r_ame.csv")
write_csv(as.data.frame(ame_age_summary), "test/r_compare/r_ame_age.csv")
write_csv(as.data.frame(ame_scenario_summary), "test/r_compare/r_ame_scenario.csv")

# Export coefficient comparison data
coef_df <- data.frame(
  term = names(coef(model_r)),
  estimate = coef(model_r),
  std_error = summary(model_r)$coefficients[, "Std. Error"]
)
write_csv(coef_df, "test/r_compare/r_coefficients.csv")

cat("\nAll results exported to test/r_compare/\n")
cat("\n✓ R analysis complete! Ready for comparison with Julia results.\n")
