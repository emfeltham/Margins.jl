# r_model.R
# R model estimation for R comparison study
# Loads data from CSV (same file as Julia script)

library(tidyverse)
library(margins)
library(emmeans)  # For reference grid predictions
library(broom)
library(microbenchmark)

cat("================================================================================\n")
cat("R Model Estimation - R Comparison Study\n")
cat("================================================================================\n\n")

# Load data from CSV (same file Julia uses)
cat("Loading data from CSV...\n")
df <- read_csv("r_comparison_data.csv", show_col_types = FALSE)
cat("Loaded:", nrow(df), "rows,", ncol(df), "columns\n\n")

# Convert to proper data types (MUST match Julia processing exactly)
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

# Integer variables (ensure proper integer type)
df <- df %>%
  mutate(
    num_common_nbs = as.integer(num_common_nbs),
    age_p = as.integer(age_p),
    schoolyears_p = as.integer(schoolyears_p),
    population = as.integer(population),
    degree_h = as.integer(degree_h),
    degree_p = as.integer(degree_p)
  )

# CRITICAL: Categorical variables must use EXACT same levels and order as Julia
# Get these from running generate_data.jl first!
cat("\nConverting categorical variables to factors...\n")
cat("NOTE: Update these factor levels based on generate_data.jl output!\n\n")

df <- df %>%
  mutate(
    # perceiver: Unique IDs - convert to factor (order doesn't matter for unique IDs)
    perceiver = factor(perceiver),

    # village_code: Convert to factor (numeric codes, alphabetical is fine)
    village_code = factor(village_code),

    # relation: CRITICAL - must match Julia order exactly
    # TODO: Update with actual levels from generate_data.jl output
    # Example: relation = factor(relation, levels = c("free_time", "work", "family", "neighbor", "friend"))
    relation = factor(relation),

    # religion_c_p: CRITICAL - must match Julia order exactly
    # TODO: Update with actual levels from generate_data.jl output
    # Example: religion_c_p = factor(religion_c_p, levels = c("Catholic", "Protestant", "No Religion", "Other"))
    religion_c_p = factor(religion_c_p),

    # Mixed categorical variables - order from data generation
    religion_c_x = factor(religion_c_x),
    man_x = factor(man_x),
    isindigenous_x = factor(isindigenous_x)
  )

cat("Data types converted\n")

# VERIFICATION: Print factor levels
cat("\n================================================================================\n")
cat("Categorical variable levels (verify against Julia output):\n")
cat("================================================================================\n")
cat("relation:", levels(df$relation), "\n")
cat("religion_c_p:", levels(df$religion_c_p), "\n")
cat("village_code:", levels(df$village_code), "\n\n")

# ==============================================================================
# Model Estimation
# ==============================================================================

cat("================================================================================\n")
cat("Fitting logistic regression model...\n")
cat("================================================================================\n")

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

cat("\nModel fitted in", round(model_fit_time["elapsed"], 2), "seconds\n")
cat("Number of coefficients:", length(coef(model_r)), "\n\n")

# Export model coefficients for comparison with Julia
cat("Exporting model coefficients...\n")
coef_df <- data.frame(
  term = names(coef(model_r)),
  estimate = coef(model_r),
  std_error = summary(model_r)$coefficients[, "Std. Error"]
)
write_csv(coef_df, "r_coefficients.csv")

# Save model objects
write_rds(model_r, "r_model_fit.rds")
write_rds(summary(model_r), "r_model_summary.rds")

# ==============================================================================
# Marginal Effects Computation
# ==============================================================================

cat("\n================================================================================\n")
cat("Computing marginal effects...\n")
cat("================================================================================\n\n")

# APM - Adjusted Predictions at Profiles
# Julia: profile_margins uses delta method for proper SEs
# R: Use margins package at specific profiles for predictions with delta-method SEs
cat("APM (Adjusted Predictions at Profiles)...\n")
apm_time <- system.time({
  # Use margins package which computes predictions with proper delta-method SEs
  # prediction(at=...) gives predictions at specific covariate values averaged over data
  apm_margins <- prediction(
    model_r,
    at = list(
      socio4 = c(FALSE, TRUE),
      are_related_dists_a_inv = c(1.0, 1/6)
    )
  )
  # Extract the summary which has proper SEs
  margins_apm <- summary(apm_margins)
})
cat("  Completed in", round(apm_time["elapsed"], 2), "seconds\n")

# MEM - Marginal Effects at Profiles (with pairwise contrasts)
# Julia: profile_margins with contrasts=:pairwise
# R: margins computes effects, we need to extract pairwise contrasts manually
cat("MEM (Marginal Effects at Profiles, pairwise contrasts)...\n")
mem_time <- system.time({
  margins_mem <- margins(
    model_r,
    at = list(
      socio4 = c(FALSE, TRUE),
      are_related_dists_a_inv = c(1.0, 1/6)
    )
    # Note: R margins computes effects on response scale by default
    # Julia default is also response scale
  )
  # TODO: Extract pairwise contrasts if needed for comparison
  # R margins package doesn't have built-in pairwise contrasts like Julia
})
cat("  Completed in", round(mem_time["elapsed"], 2), "seconds\n")

# AAP - Average Adjusted Predictions
# Julia: population_margins uses delta method for proper SE
# R: Use margins package prediction() function for delta-method SE
cat("AAP (Average Adjusted Predictions)...\n")
aap_time <- system.time({
  # Use prediction() which computes delta-method standard errors
  aap_pred <- prediction(model_r)
  margins_aap <- summary(aap_pred)
})
cat("  Completed in", round(aap_time["elapsed"], 2), "seconds\n")

# AME - Average Marginal Effects (all variables)
cat("AME (Average Marginal Effects - all variables)...\n")
ame_time <- system.time({
  margins_ame <- margins(model_r)
  # Note: R margins computes effects on response scale by default
  # Julia default is also response scale
})
cat("  Completed in", round(ame_time["elapsed"], 2), "seconds\n")

# AME - Single variable
cat("AME (single variable - age_h)...\n")
ame_age_time <- system.time({
  margins_ame_age <- margins(model_r, variables = "age_h")
})
cat("  Completed in", round(ame_age_time["elapsed"], 2), "seconds\n")

# AME - With scenario
cat("AME (with scenario - wealth variables at are_related_dists_a_inv=1/6)...\n")
ame_scenario_time <- system.time({
  margins_ame_scenario <- margins(
    model_r,
    variables = c("wealth_d1_4_p", "wealth_d1_4_h"),
    at = list(are_related_dists_a_inv = 1/6)
  )
})
cat("  Completed in", round(ame_scenario_time["elapsed"], 2), "seconds\n\n")

# ==============================================================================
# Performance Benchmarking
# ==============================================================================
# Skipped - not needed for validation

# ==============================================================================
# Export Results
# ==============================================================================

cat("\n================================================================================\n")
cat("Exporting results...\n")
cat("================================================================================\n")

# Export marginal effects to CSV
# APM and AAP are already data frames, just export directly
write_csv(margins_apm, "r_apm.csv")
write_csv(margins_aap, "r_aap.csv")

# MEM and AME are margins objects, use summary() to get the data frame
write_csv(as.data.frame(summary(margins_mem)), "r_mem.csv")
write_csv(as.data.frame(summary(margins_ame)), "r_ame.csv")
write_csv(as.data.frame(summary(margins_ame_age)), "r_ame_age.csv")
write_csv(as.data.frame(summary(margins_ame_scenario)), "r_ame_scenario.csv")

# Save complete results object
results_list <- list(
  apm = margins_apm,  # Already a data frame
  mem = summary(margins_mem),
  aap = margins_aap,  # Already a data frame
  ame = summary(margins_ame),
  ame_age = summary(margins_ame_age),
  ame_scenario = summary(margins_ame_scenario),
  model_summary = summary(model_r),
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
write_rds(results_list, "r_results_complete.rds")

cat("\nAll results exported to current directory\n")
cat("\n✓ R analysis complete!\n")
cat("\nNext step:\n")
cat("Run: julia --project=. compare_results.jl\n")
