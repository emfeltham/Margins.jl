# Reference values for test/validation/test_oracle_3_0.jl, "at = :observed second differences".
#
# The default second difference of a continuous modifier w (`second_differences(m, d, :x, :w)`,
# `at = :observed`) is the average over the rows of the cross-derivative ∂²ŷᵢ/∂x∂w, computed as
#   (AME(x | every row's w shifted by +δ) − AME(x | every row's w shifted by −δ)) / (2δ)
# with δ = 0.01·sd(w) (Margins' `delta = :auto`, reported as `delta_used`). Here: the two shifted
# copies of the data stacked in one `newdata`, labelled by `shift_`, then
#   avg_slopes(m, variables = "x", newdata = nd, by = "shift_", hypothesis = "(b2 - b1) / (2δ) = 0")
# whose delta-method SE uses the joint Jacobian of both averages (they are strongly correlated; an
# SE from two independent AMEs would be far too large). A categorical focal variable uses
# avg_comparisons with one contrast at a time; a `groups = :g` stratum restricts `newdata` to the
# stratum's rows (the same Σ).
#
#   Rscript observed_sd_reference.R > observed_sd_reference.csv
#
# marginaleffects 1.0.0; `eps = 1e-4` (central-difference slope; a smaller step's rounding error is
# amplified by the numerical Jacobian and by the division by 2δ) and `numderiv = "richardson"` make
# the slope and the Jacobian accurate enough to compare SEs tightly.
# The fixture stores every number with 17 significant digits.
suppressPackageStartupMessages(library(marginaleffects))
options(marginaleffects_safe = FALSE)
source("oracle_data.R")
d <- oracle_data()

# Tight IRLS convergence (Julia fits with rtol = 1e-15): at the default tolerances GLM.jl and R stop
# at fits about 1e-5 apart (probit), which would dominate every comparison below
ctl <- glm.control(epsilon = 1e-14, maxit = 100)
models <- list(
  lm        = lm(yc ~ x * w + z + g * w, data = d),
  logit     = glm(yb ~ x * w + z + g * w, data = d, family = binomial(), control = ctl),
  probit    = glm(yb ~ x * w + z + g * w, data = d, family = binomial(link = "probit"), control = ctl),
  logit_add = glm(yb ~ x + w + z + g, data = d, family = binomial(), control = ctl),       # no interaction
  wlogit    = suppressWarnings(glm(yb ~ x * w + z + g * w, data = d, family = binomial(), weights = sw, control = ctl))
)

dl <- 0.01 * sd(d$w)
nd <- rbind(transform(d, w = w - dl, shift_ = "a_minus"), transform(d, w = w + dl, shift_ = "b_plus"))
hyp <- sprintf("(b2 - b1) / %.17g = 0", 2 * dl)

out <- list()
add <- function(model, focal, contrast, measure, type, stratum, r) {
  stopifnot(nrow(r) == 1)
  out[[length(out) + 1]] <<- data.frame(model = model, focal = focal, contrast = contrast,
    measure = measure, type = type, stratum = stratum, delta = sprintf("%.17g", dl),
    estimate = sprintf("%.17g", r$estimate), std_error = sprintf("%.17g", r$std.error))
}

for (mn in names(models)) {
  m <- models[[mn]]
  types <- if (mn == "lm") "response" else if (mn == "logit_add") "response" else c("response", "link")
  wargs <- if (mn == "wlogit") list(wts = "sw") else list()
  for (tp in types) {
    common <- c(list(m, type = tp, eps = 1e-4, numderiv = "richardson", by = "shift_", hypothesis = hyp), wargs)
    # eyex on the link scale divides by η, which crosses 0: response scale only
    slopes_x <- c("dydx", if (mn %in% c("logit", "wlogit") && tp == "response") "eyex")
    for (s in slopes_x) {
      r <- do.call(avg_slopes, c(common, list(variables = "x", slope = s, newdata = nd)))
      add(mn, "x", "dy/dx", s, tp, "all", r)
    }
    for (lv in c("b", "c")) {
      r <- do.call(avg_comparisons, c(common, list(variables = list(g = c("a", lv)), newdata = nd)))
      add(mn, "g", paste(lv, "- a"), "dydx", tp, "all", r)
    }
    # groups = :g on the logit: each stratum's rows, the same fit
    if (mn == "logit") for (st in levels(d$g)) {
      r <- do.call(avg_slopes, c(common, list(variables = "x", newdata = nd[nd$g == st, ])))
      add(mn, "x", "dy/dx", "dydx", tp, st, r)
    }
  }
}
res <- do.call(rbind, out)
write.csv(res, stdout(), row.names = FALSE, quote = FALSE)
