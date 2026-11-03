# Reference values for test/validation/test_second_differences_r_oracle.jl
#
# A second difference is the change in an average marginal effect when the modifier is set to
# different values for everyone (a counterfactual grid):
#
#   avg_slopes(m, variables = "x", newdata = datagrid(t = c(0, 1), grid_type = "counterfactual"),
#              by = "t", hypothesis = "b2 - b1 = 0")
#
# The delta-method SE comes from the joint Jacobian of both averages, so it includes their
# covariance. The deterministic dataset (no RNG) is the one built in the Julia test.
#
#   Rscript second_differences_reference.R > second_differences_reference.csv
#
# Produced with marginaleffects 1.0.0; `numderiv = "richardson"` makes the Jacobian accurate
# enough to compare SEs tightly.
suppressPackageStartupMessages(library(marginaleffects))

i <- 1:300
d <- data.frame(
  x = 0.5 + ((i * 37) %% 101) / 40,
  z = cos(i * 0.7),
  g = factor(c("a", "b", "c")[(i %% 3) + 1]),
  w = 0.5 + ((i * 13) %% 17) / 10,
  t = as.numeric(i %% 2),
  sw = 0.5 + ((i * 7) %% 11) / 5   # weights (a separate column: `w` is a modifier below)
)
d$yb  <- as.numeric(sin(i * 1.3) + 0.8 * d$x - 1.5 + 0.5 * d$z + 0.4 * d$x * d$t > 0)
d$cnt <- floor(exp(0.3 + 0.4 * d$x - 0.3 * d$z + 0.2 * d$x * d$t) + 2 * (sin(i * 2.1) + 1))

models <- list(
  logit   = glm(yb ~ x * t + x * w + z + g, data = d, family = binomial()),
  poisson = glm(cnt ~ x * t + x * w + z + g, data = d, family = poisson())
)

w0 <- 1.2; dl <- 0.01          # local derivative: AME at w0 - dl vs w0 + dl
a <- 0.8;  b <- 1.8            # slope between two modifier values

hyp_diff <- "b2 - b1 = 0"
out <- list()
add <- function(model, kind, measure, weights, type, r) {
  out[[length(out) + 1]] <<- data.frame(model = model, kind = kind, measure = measure, weights = weights,
                                        type = type, estimate = r$estimate, std_error = r$std.error)
}

for (mn in names(models)) {
  m <- models[[mn]]
  for (tp in c("response", "link")) for (wt in c("none", "sw")) {
    wargs <- if (wt == "sw") list(wts = "sw") else list()
    common <- c(list(m, type = tp, eps = 1e-6, numderiv = "richardson"), wargs)

    # binary modifier t, continuous focal x, every measure
    nd <- datagrid(newdata = d, t = c(0, 1), grid_type = "counterfactual")
    for (s in c("dydx", "eyex", "dyex", "eydx")) {
      r <- do.call(avg_slopes, c(common, list(variables = "x", slope = s, newdata = nd, by = "t",
                                              hypothesis = hyp_diff)))
      add(mn, "binary_x", s, wt, tp, r)
    }
    # binary modifier t, categorical focal g (one contrast at a time)
    for (lv in c("b", "c")) {
      r <- do.call(avg_comparisons, c(common, list(variables = list(g = c("a", lv)), newdata = nd, by = "t",
                                                   hypothesis = hyp_diff)))
      add(mn, paste0("binary_g_", lv), "dydx", wt, tp, r)
    }
    # continuous modifier w: local derivative (symmetric difference of the AME)
    nl <- datagrid(newdata = d, w = c(w0 - dl, w0 + dl), grid_type = "counterfactual")
    for (s in c("dydx", "eyex")) {
      r <- do.call(avg_slopes, c(common, list(variables = "x", slope = s, newdata = nl, by = "w",
                                              hypothesis = sprintf("(b2 - b1) / %.10g = 0", 2 * dl))))
      add(mn, "local_w", s, wt, tp, r)
    }
    # slope of the AME between two values of w
    ns <- datagrid(newdata = d, w = c(a, b), grid_type = "counterfactual")
    r <- do.call(avg_slopes, c(common, list(variables = "x", newdata = ns, by = "w",
                                            hypothesis = sprintf("(b2 - b1) / %.10g = 0", b - a))))
    add(mn, "slope_w", "dydx", wt, tp, r)
  }
}
res <- do.call(rbind, out)
write.csv(res, stdout(), row.names = FALSE, quote = FALSE)
