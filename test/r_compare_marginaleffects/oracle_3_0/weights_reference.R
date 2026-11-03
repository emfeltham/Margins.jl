# Reference values for test/validation/test_oracle_3_0.jl, "model weights".
#
# Fits with estimation weights, integer (`fw`, 1..4, frequency-type) and non-integer (`sw`), and
# the weighted AME, AAP, MEM and `groups = :g` (`by = "g"`) from marginaleffects with `wts =` the
# same column (Margins uses the fit's weights by default, `weights = :model`). MEM is at fixed
# values and at the weighted means of x and z (computed here with weighted.mean; the Julia test
# computes them with Σwx/Σw).
#
#   Rscript weights_reference.R > weights_reference.csv
#
# Convention: GLM.jl treats `wts` as frequency weights, so a model with an estimated dispersion
# (lm, Gaussian glm) has σ̂² = Σwᵢrᵢ²/(Σw − p); R's lm / glm treat `weights` as precision weights,
# σ̂² = Σwᵢrᵢ²/(n − p). Estimates agree; SEs of the Gaussian fits differ by the constant
# sqrt((n − p)/(Σw − p)), which the Julia test checks exactly. Logit and Poisson have a fixed
# dispersion, and their SEs agree. marginaleffects reports z statistics; Margins uses
# t(dof_residual) for the linear models (only estimates and SEs are compared).
#
# marginaleffects 1.0.0 with `numderiv = "richardson"` (Jacobian) and `eps = 1e-4` (the step of the
# numerical slope: at 1e-6 its rounding error, amplified by the numerical Jacobian, reaches 1e-3
# relative in the unit-level MEM SEs; at 1e-4 the central-difference truncation error is ~1e-11).
# Every number with 17 significant digits.
suppressPackageStartupMessages(library(marginaleffects))
options(marginaleffects_safe = FALSE)
source("oracle_data.R")
d <- oracle_data()
ctl <- glm.control(epsilon = 1e-14, maxit = 100)

fits <- list()
for (wv in c("fw", "sw")) {
  fits[[paste0("lm_", wv)]]      <- lm(as.formula("yc ~ x * z + g"), data = d, weights = d[[wv]])
  fits[[paste0("gauss_", wv)]]   <- glm(yc ~ x * z + g, data = d, family = gaussian(), weights = d[[wv]], control = ctl)
  fits[[paste0("logit_", wv)]]   <- suppressWarnings(glm(yb ~ x * z + g, data = d, family = binomial(), weights = d[[wv]], control = ctl))
  fits[[paste0("poisson_", wv)]] <- glm(cnt ~ x * z + g, data = d, family = poisson(), weights = d[[wv]], control = ctl)
}

out <- list()
add <- function(model, quantity, term, contrast, by, type, r) {
  out[[length(out) + 1]] <<- data.frame(model = model, quantity = quantity, term = term,
    contrast = contrast, by = by, type = type,
    estimate = sprintf("%.17g", r$estimate), std_error = sprintf("%.17g", r$std.error))
}
for (mn in names(fits)) {
  m <- fits[[mn]]
  wv <- sub(".*_", "", mn)
  types <- if (startsWith(mn, "lm") || startsWith(mn, "gauss")) "response" else c("response", "link")
  wm <- function(v) weighted.mean(d[[v]], d[[wv]])
  for (tp in types) {
    common <- list(m, type = tp, eps = 1e-4, numderiv = "richardson")
    # AME, every variable
    r <- do.call(avg_slopes, c(common, list(wts = wv)))
    for (k in seq_len(nrow(r))) add(mn, "ame", r$term[k], r$contrast[k], "all", tp, r[k, ])
    # AME within g
    r <- do.call(avg_slopes, c(common, list(variables = c("x", "z"), by = "g", wts = wv)))
    for (k in seq_len(nrow(r))) add(mn, "ame", r$term[k], r$contrast[k], as.character(r$g[k]), tp, r[k, ])
    # AAP, overall and within g
    r <- do.call(avg_predictions, c(common[-3], list(wts = wv)))
    add(mn, "aap", "", "", "all", tp, r)
    r <- do.call(avg_predictions, c(common[-3], list(by = "g", wts = wv)))
    for (k in seq_len(nrow(r))) add(mn, "aap", "", "", as.character(r$g[k]), tp, r[k, ])
    # MEM at fixed values and at the weighted means (g = "b"); the weights do not enter except
    # through the fit
    grids <- list(fixed = datagrid(model = m, x = 1.5, z = 0.25, g = "b"),
                  wmeans = datagrid(model = m, x = wm("x"), z = wm("z"), g = "b"))
    for (gn in names(grids)) {
      r <- do.call(slopes, c(common, list(variables = c("x", "z"), newdata = grids[[gn]])))
      for (k in seq_len(nrow(r))) add(mn, paste0("mem_", gn), r$term[k], r$contrast[k], "all", tp, r[k, ])
    }
  }
}
res <- do.call(rbind, out)
write.csv(res, stdout(), row.names = FALSE, quote = FALSE)
