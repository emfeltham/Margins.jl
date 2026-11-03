# Reference values for test/validation/test_oracle_3_0.jl, "FixedEffectModels vs fixest".
#
# A two-way fixed-effects linear model, feols(y ~ x * z | st + yr), on a deterministic panel (no
# RNG; `_fem_oracle_data()` in the Julia test builds the same rows). Rows 301-303 are singletons of
# their state (dropped by FixedEffectModels' default `drop_singletons = true`, and here by
# `fixef.rm = "singleton"`) and row 304 has a missing z, so the estimation sample (`esample`) is
# 300 of 304 rows. 12 states and 11 years (coprime) make every state-year pair occur, so the
# fixed-effects graph is connected and α_state + γ_year is identified for every pair (with 9 years,
# i mod 3 would split the panel into three components, and the prediction at an unobserved pair
# would depend on the normalization of the fixed effects).
#
# Compared: the coefficients and their SEs under iid ("simple"), heteroskedasticity-robust and
# state-clustered covariance; the average adjusted prediction (overall and by state) and the
# prediction at a grid row, which include the estimated fixed effects (fixest `predict`;
# marginaleffects treats the fixed effects as known, as Margins does, so the SE comes from the
# slopes' covariance only); the AME of x and z (overall and by state), via marginaleffects.
#
#   Rscript fixest_reference.R > fixest_reference.csv
#
# fixest and sandwich are not in the default library; they were installed into a private library,
# put first on .libPaths() through R_LIBS:
#   R_LIBS=<lib> Rscript fixest_reference.R > fixest_reference.csv
# The fixed effects are solved tightly (`fixef.tol = 1e-11`, its minimum is 2.2e-12; FixedEffectModels `tol = 1e-12` in the test):
# at the defaults the two packages' FE agree only to ~4e-8 relative.
# fixest 0.14.2, marginaleffects 1.0.0 (`eps = 1e-4`, `numderiv = "richardson"`); 17 digits.
suppressPackageStartupMessages({library(fixest); library(marginaleffects)})
options(marginaleffects_safe = FALSE)
i <- 1:300
d <- data.frame(st = paste0("s", (i %% 12) + 1), yr = 2000 + ((i * 7) %% 11),
                x = 0.5 + ((i * 37) %% 101) / 40, z = cos(i * 0.7))
d$y <- 1 + 0.5 * d$x - 0.3 * d$z + 0.4 * d$x * d$z + 0.2 * (i %% 12) + 0.1 * ((i * 7) %% 11) + sin(i * 2.3)
extra <- data.frame(st = c("t1", "t2", "t3", "s1"), yr = c(2001, 2003, 2005, 2002),
                    x = c(1.1, 2.2, 0.7, 1.4), z = c(0.3, -0.2, 0.5, NA), y = c(3.1, 2.4, 4.0, 2.9))
d <- rbind(d, extra)

out <- list()
add <- function(vcov, quantity, term, by, r_est, r_se) {
  out[[length(out) + 1]] <<- data.frame(vcov = vcov, quantity = quantity, term = term, by = by,
    estimate = sprintf("%.17g", r_est), std_error = sprintf("%.17g", r_se))
}
vcovs <- list(simple = "iid", robust = "hetero", cluster = ~st)
for (vn in names(vcovs)) {
  m <- feols(y ~ x * z | st + yr, data = d, vcov = vcovs[[vn]], fixef.rm = "singleton",
             fixef.tol = 1e-11, fixef.iter = 1e5)
  stopifnot(nobs(m) == 300)
  for (k in names(coef(m))) add(vn, "coef", k, "all", coef(m)[[k]], se(m)[[k]])
  # marginaleffects' default `newdata` is the whole data frame (304 rows, NA predictions for the
  # dropped rows, NaN averages for the singleton states); Margins computes over the estimation
  # sample, so that is passed explicitly
  es <- d[obs(m), ]
  common <- list(m, numderiv = "richardson", newdata = es)
  r <- do.call(avg_predictions, common)
  add(vn, "aap", "", "all", r$estimate, r$std.error)
  # the overall AAP is the mean of fixest's fitted values (which include the fixed effects)
  stopifnot(abs(r$estimate - mean(predict(m))) < 1e-12)
  r <- do.call(avg_predictions, c(common, list(by = "st")))
  for (k in seq_len(nrow(r))) add(vn, "aap", "", as.character(r$st[k]), r$estimate[k], r$std.error[k])
  r <- do.call(avg_slopes, c(common, list(variables = c("x", "z"), eps = 1e-4)))
  for (k in seq_len(nrow(r))) add(vn, "ame", r$term[k], "all", r$estimate[k], r$std.error[k])
  r <- do.call(avg_slopes, c(common, list(variables = c("x", "z"), by = "st", eps = 1e-4)))
  for (k in seq_len(nrow(r))) add(vn, "ame", r$term[k], as.character(r$st[k]), r$estimate[k], r$std.error[k])
  # prediction at one grid row (state s3, year 2004): fixest predict and marginaleffects agree
  nd <- data.frame(st = "s3", yr = 2004, x = 1.5, z = 0.25)
  r <- do.call(predictions, list(m, numderiv = "richardson", newdata = nd))
  stopifnot(abs(r$estimate - predict(m, newdata = nd)) < 1e-12)
  add(vn, "apm", "", "s3_2004", r$estimate, r$std.error)
}
res <- do.call(rbind, out)
write.csv(res, stdout(), row.names = FALSE, quote = FALSE)
