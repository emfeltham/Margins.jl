# Reference values for test/validation/test_oracle_3_0.jl, "rank-deficient fits".
#
# Fits with a regressor that is exactly collinear with others, so the fit drops a coefficient:
# R's lm/glm report it as NA (and drop it from vcov), fixest removes it. marginaleffects treats the
# dropped coefficient as fixed at 0 (it is not in the Jacobian), and reports a quantity whose
# standard error is then 0 (one that depends only on the dropped coefficient) with SE NA.
#
#   v = 2x + 3z         (lm, logit: `z` is dropped; GLM.jl's pivoted Cholesky drops `z` too, and
#                        R drops the last aliased column, so `z` is placed last in the formula;
#                        with v = x + z, x and z tie in GLM.jl's pivoting and the BLAS decides)
#   c = level value of g (feols with `| g`: `c` is collinear with the fixed effect and removed)
#
# Quantities: AME of every variable (the dropped `z` / `c` included: estimate 0, SE NA), AME within
# g, AAP overall and within g, and the effects at one grid row (MEM). `refit_estimate` and
# `refit_std_error` are the same quantities from the refit without the redundant term (NA for the
# dropped variable). On the rank-deficient lm, marginaleffects' numerical Jacobian is noisy (MEM SE
# of x 1.9e-5 relative from the refit's, which is exact for a linear model; 2.6% with the default
# `numderiv`), so the Julia test compares Margins tightly with the refit columns and loosely with
# the rank-deficient ones.
#
#   R_LIBS=<private lib with fixest> Rscript rank_deficient_reference.R > rank_deficient_reference.csv
#
# R 4.6.1, marginaleffects 1.0.0 (`numderiv = "richardson"`, `eps = 1e-4`), fixest 0.14.2; 17 digits.
suppressPackageStartupMessages({library(marginaleffects); library(fixest)})
options(marginaleffects_safe = FALSE, warn = -1)   # the rank-deficiency warnings are expected
source("oracle_data.R")
d <- oracle_data()
d$v <- 2 * d$x + 3 * d$z
d$c <- c(a = 0.7, b = 1.9, c = -0.4)[as.character(d$g)]
# glm's QR rank tolerance is min(1e-7, epsilon / 1000): at the 1e-14 of the other scripts (1e-17) the
# collinear column is not detected, so epsilon is 1e-11 here (QR tolerance 1e-14; the IRLS iterate is
# converged far below this, as the coefficient comparison in the Julia test shows)
ctl <- glm.control(epsilon = 1e-11, maxit = 100)

fits <- list(
  lm    = lm(yc ~ x * w + g + v + z, data = d),
  logit = glm(yb ~ x + w + g + v + z, data = d, family = binomial(), control = ctl),
  fem   = feols(yc ~ x * w + z + c | g, data = d, fixef.tol = 1e-11, fixef.iter = 1e5)
)
stopifnot(is.na(coef(fits$lm)[["z"]]), is.na(coef(fits$logit)[["z"]]), !("c" %in% names(coef(fits$fem))))
# the same models without the redundant term
refits <- list(
  lm    = lm(yc ~ x * w + g + v, data = d),
  logit = glm(yb ~ x + w + g + v, data = d, family = binomial(), control = ctl),
  fem   = feols(yc ~ x * w + z | g, data = d, fixef.tol = 1e-11, fixef.iter = 1e5)
)
dropped <- c(lm = "z", logit = "z", fem = "c")

quantities <- function(m, mn, tp, drop) {
  out <- list()
  add <- function(quantity, r, by) {
    out[[length(out) + 1]] <<- data.frame(model = mn, quantity = quantity,
      term = if (is.null(r$term)) "" else r$term, contrast = if (is.null(r$contrast)) "" else r$contrast,
      by = by, type = tp, estimate = r$estimate, std_error = r$std.error)
  }
  keep <- function(v) setdiff(v, drop)
  common <- list(m, type = tp, numderiv = "richardson", newdata = d)
  r <- do.call(avg_slopes, c(common, list(eps = 1e-4)))
  add("ame", r, "all")
  byvars <- if (mn == "fem") c("x", "z", "c") else c("x", "v", "z")
  r <- do.call(avg_slopes, c(common, list(eps = 1e-4, variables = keep(byvars), by = "g")))
  add("ame", r, as.character(r$g))
  r <- do.call(avg_predictions, common)
  add("aap", r, "all")
  r <- do.call(avg_predictions, c(common, list(by = "g")))
  add("aap", r, as.character(r$g))
  # effects at one grid row (every variable given, so nothing is filled in)
  nd <- data.frame(x = 1.5, w = 1.2, z = 0.1, v = 3.3, g = factor("b", levels = levels(d$g)), c = 1.9)
  vars <- if (mn == "fem") c("x", "w", "z", "c") else c("x", "w", "v", "z")
  r <- do.call(slopes, list(m, type = tp, numderiv = "richardson", eps = 1e-4, variables = keep(vars), newdata = nd))
  add("mem", r, "all")
  do.call(rbind, out)
}

res <- list()
for (mn in names(fits)) {
  for (tp in if (mn == "logit") c("response", "link") else "response") {
    a <- quantities(fits[[mn]], mn, tp, character(0))
    b <- quantities(refits[[mn]], mn, tp, dropped[[mn]])
    names(b)[names(b) == "estimate"] <- "refit_estimate"
    names(b)[names(b) == "std_error"] <- "refit_std_error"
    ab <- merge(a, b, all.x = TRUE, sort = FALSE)
    stopifnot(nrow(ab) == nrow(a), sum(is.na(ab$refit_estimate)) == sum(a$term == dropped[[mn]]))
    res[[length(res) + 1]] <- ab
  }
}
res <- do.call(rbind, res)
for (k in c("estimate", "std_error", "refit_estimate", "refit_std_error")) res[[k]] <- sprintf("%.17g", res[[k]])
write.csv(res, stdout(), row.names = FALSE, quote = FALSE)
