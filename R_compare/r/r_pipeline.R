#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(marginaleffects)
  library(nnet)
  library(MASS)
  library(sandwich)
})

args <- commandArgs(trailingOnly = TRUE)

# Defaults
vcov_spec <- NULL  # model-based by default
type <- "response" # response vs link

if (length(args) > 0) {
  for (i in seq(1, length(args), by = 2)) {
    key <- args[i]
    val <- ifelse(i + 1 <= length(args), args[i + 1], NA)
    if (key %in% c("--vcov", "-v")) {
      if (!is.na(val) && nzchar(val)) vcov_spec <- val
    }
    if (key %in% c("--type", "-t")) {
      if (!is.na(val) && nzchar(val)) type <- val
    }
  }
}

message(sprintf("Using vcov=%s, type=%s", ifelse(is.null(vcov_spec), "model", vcov_spec), type))

dir.create("data", showWarnings = FALSE, recursive = TRUE)
dir.create("results_r", showWarnings = FALSE, recursive = TRUE)

prepare_datasets <- function() {
  # mtcars - keep numeric variables numeric, only factor true categoricals
  # cyl, gear, carb stay numeric (they're counts)
  # vs, am become logical (TRUE/FALSE)
  mtcars$vs <- as.logical(mtcars$vs)  # TRUE = Straight, FALSE = V
  mtcars$am <- as.logical(mtcars$am)  # TRUE = Manual, FALSE = Automatic

  # ToothGrowth - supp is naturally categorical, dose is naturally numeric
  ToothGrowth$supp <- factor(ToothGrowth$supp, levels = c("OJ","VC"))
  # dose stays numeric (0.5, 1, 2 mg)

  # iris - Species is naturally categorical (keep as-is)

  # Titanic (expand from table) to individual observations  
  titanic_df <- as.data.frame(Titanic)
  titanic_expanded <- titanic_df[rep(seq_len(nrow(titanic_df)), titanic_df$Freq), 1:4]
  # Survived becomes logical
  titanic_expanded$Survived <- titanic_expanded$Survived == "Yes"
  # Class, Sex, Age are naturally categorical
  titanic_expanded$Class <- factor(titanic_expanded$Class, levels = c("1st","2nd","3rd","Crew"))
  titanic_expanded$Sex   <- factor(titanic_expanded$Sex, levels = c("Female","Male"))
  titanic_expanded$Age   <- factor(titanic_expanded$Age, levels = c("Adult","Child"))

  return(list(
    mtcars = mtcars,
    iris = iris,
    toothgrowth = ToothGrowth,
    titanic = titanic_expanded
  ))
}

estimate_models <- function(datasets) {
  # Ensure all categorical variables are proper factors before model fitting
  datasets$toothgrowth$supp <- factor(datasets$toothgrowth$supp, levels = c("OJ","VC"))
  datasets$iris$Species <- factor(datasets$iris$Species, levels = c("setosa","versicolor","virginica"))
  datasets$titanic$Class <- factor(datasets$titanic$Class, levels = c("1st","2nd","3rd","Crew"))
  datasets$titanic$Sex <- factor(datasets$titanic$Sex, levels = c("Female","Male"))
  datasets$titanic$Age <- factor(datasets$titanic$Age, levels = c("Adult","Child"))
  
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
    m1 = lm(len ~ supp + dose, data = datasets$toothgrowth),
    m2 = lm(len ~ supp * dose, data = datasets$toothgrowth),
    m3 = lm(len ~ supp + dose + I(dose^2), data = datasets$toothgrowth),
    m4 = lm(len ~ supp + I(dose - 1), data = datasets$toothgrowth),
    m5 = lm(len ~ supp * I(dose^2), data = datasets$toothgrowth)
  )

  # titanic models (logistic regression)
  models$titanic <- list(
    m1 = glm(Survived ~ Sex + Class, data = datasets$titanic, family = binomial),
    m2 = glm(Survived ~ Class + Sex + Age, data = datasets$titanic, family = binomial),
    m3 = glm(Survived ~ Class * Sex + Age, data = datasets$titanic, family = binomial),
    # m4 simplified to avoid empty cells in Class:Age (e.g., Crew:Child)
    m4 = glm(Survived ~ Class + Sex * Age, data = datasets$titanic, family = binomial),
    m5 = glm(Survived ~ (Class + Age) * Sex, data = datasets$titanic, family = binomial)
  )

  return(models)
}

check_rank_deficiency <- function(models) {
  message("[rank-check] Inspecting model matrices for rank deficiency...")
  for (dataset_name in names(models)) {
    for (model_name in names(models[[dataset_name]])) {
      model <- models[[dataset_name]][[model_name]]
      mm <- tryCatch(stats::model.matrix(model), error = function(e) NULL)
      if (is.null(mm)) {
        message(sprintf("[rank-check] %s:%s model.matrix unavailable", dataset_name, model_name))
        next
      }
      r <- suppressWarnings(tryCatch(qr(mm)$rank, error = function(e) NA_integer_))
      p <- ncol(mm)
      aliased <- suppressWarnings(tryCatch(summary(model)$aliased, error = function(e) NULL))
      is_def <- (!is.na(r) && r < p) || (!is.null(aliased) && any(aliased, na.rm = TRUE))
      if (is_def) {
        aliased_coefs <- if (!is.null(aliased)) names(aliased)[which(aliased)] else character(0)
        message(sprintf("[rank-deficient] %s:%s rank=%s < cols=%s; aliased=%s",
                        dataset_name, model_name, as.character(r), as.character(p),
                        if (length(aliased_coefs)) paste(aliased_coefs, collapse = ", ") else "none"))
      } else {
        message(sprintf("[full-rank] %s:%s rank=%s cols=%s",
                        dataset_name, model_name, as.character(r), as.character(p)))
      }
    }
  }
}

compute_marginal_effects <- function(models, datasets, vcov_spec = NULL, type = "response") {
  results <- list()

  # Ensure categorical variables remain proper factors (in case CSV export/import changed them)
  datasets$toothgrowth$supp <- factor(datasets$toothgrowth$supp, levels = c("OJ","VC"))
  datasets$iris$Species <- factor(datasets$iris$Species, levels = c("setosa","versicolor","virginica")) 
  datasets$titanic$Class <- factor(datasets$titanic$Class, levels = c("1st","2nd","3rd","Crew"))
  datasets$titanic$Sex <- factor(datasets$titanic$Sex, levels = c("Female","Male"))
  datasets$titanic$Age <- factor(datasets$titanic$Age, levels = c("Adult","Child"))

  for (dataset_name in names(models)) {
    results[[dataset_name]] <- list()
    dataset <- datasets[[dataset_name]]

    for (model_name in names(models[[dataset_name]])) {
      model <- models[[dataset_name]][[model_name]]

      # Determine appropriate type per model
      type_here <- type
      if (inherits(model, "nnet") || inherits(model, "multinom")) {
        # multinomial models use 'probs' or 'latent'
        type_here <- "probs"
      }

      # AME
      ame <- avg_slopes(model, vcov = vcov_spec, type = type_here)

      # Typical row for MEM/APM (aligned with Julia):
      # - numeric: mean
      # - factor: baseline (first level)
      # - logical: mode (most frequent)
      means_data <- dataset[1, , drop = FALSE]
      for (col in names(dataset)) {
        x <- dataset[[col]]
        if (is.numeric(x)) {
          means_data[[col]] <- mean(x, na.rm = TRUE)
        } else if (is.factor(x)) {
          means_data[[col]] <- levels(x)[1]
        } else if (is.logical(x)) {
          tab <- table(x)
          # choose most frequent logical value; ties break by TRUE first
          which_max <- names(sort(tab, decreasing = TRUE))[1]
          means_data[[col]] <- which_max == "TRUE"
        }
      }

      mem <- slopes(model, newdata = means_data, vcov = vcov_spec, type = type_here)
      aap <- avg_predictions(model, vcov = vcov_spec, type = type_here)
      apm <- predictions(model, newdata = means_data, vcov = vcov_spec, type = type_here)

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

export_datasets <- function(datasets) {
  write.csv(datasets$mtcars, file = "data/data_mtcars.csv", row.names = FALSE)
  write.csv(datasets$iris, file = "data/data_iris.csv", row.names = FALSE)
  write.csv(datasets$toothgrowth, file = "data/data_toothgrowth.csv", row.names = FALSE)
  write.csv(datasets$titanic, file = "data/data_titanic.csv", row.names = FALSE)
}

export_results <- function(results) {
  for (dataset_name in names(results)) {
    for (model_name in names(results[[dataset_name]])) {
      result <- results[[dataset_name]][[model_name]]
      write.csv(as.data.frame(result$ame), file = sprintf("results_r/r_results_%s_%s_ame.csv", dataset_name, model_name), row.names = FALSE)
      write.csv(as.data.frame(result$mem), file = sprintf("results_r/r_results_%s_%s_mem.csv", dataset_name, model_name), row.names = FALSE)
      write.csv(as.data.frame(result$aap), file = sprintf("results_r/r_results_%s_%s_aap.csv", dataset_name, model_name), row.names = FALSE)
      write.csv(as.data.frame(result$apm), file = sprintf("results_r/r_results_%s_%s_apm.csv", dataset_name, model_name), row.names = FALSE)
    }
  }
}

main <- function() {
  datasets <- prepare_datasets()
  export_datasets(datasets)
  models <- estimate_models(datasets)
  check_rank_deficiency(models)
  results <- compute_marginal_effects(models, datasets, vcov_spec = vcov_spec, type = type)
  export_results(results)
  message("R pipeline completed: data and results written under data/ and results_r/")
}

main()
