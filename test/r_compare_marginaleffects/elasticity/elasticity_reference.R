# Reference values for test/validation/test_elasticity_unit_definitions.jl
#
# Generates the deterministic dataset used by that test (no RNG, so Julia and R build
# identical data) and records R marginaleffects avg_slopes() elasticities.
#
#   Rscript elasticity_reference.R > elasticity_reference.csv
#
# Produced with marginaleffects 1.0.0. `numderiv = "richardson"` makes the delta-method
# Jacobian accurate enough to compare SEs tightly (the default forward difference is
# accurate to only ~1e-2 relative here).
suppressPackageStartupMessages(library(marginaleffects))

i <- 1:300
d <- data.frame(
  x = 0.5 + ((i * 37) %% 101) / 40,
  z = cos(i * 0.7),
  g = factor(c("a", "b", "c")[(i %% 3) + 1]),
  w = 0.5 + ((i * 13) %% 17) / 10
)
d$yb <- as.numeric(sin(i * 1.3) + 0.8 * d$x - 1.5 + 0.5 * d$z > 0)
d$cnt <- floor(exp(0.3 + 0.4 * d$x - 0.3 * d$z) + 2 * (sin(i * 2.1) + 1))
d$ylin <- 6 + 1.5 * d$x - 0.7 * d$z + 2 * sin(i * 0.9)

models <- list(
  logit   = glm(yb ~ x * z + g, data = d, family = binomial()),
  poisson = glm(cnt ~ x + z + g, data = d, family = poisson()),
  lm      = lm(ylin ~ x * z + g, data = d)
)

out <- list()
for (mn in names(models)) {
  m <- models[[mn]]
  for (s in c("eyex", "dyex", "eydx")) {
    for (wt in c("none", "w")) {
      for (by in c("none", "g")) {
        for (tp in c("response", "link")) {
          if (mn == "lm" && tp == "link") next
          args <- list(m, variables = "x", slope = s, type = tp, eps = 1e-6,
                       numderiv = "richardson")
          if (wt == "w") args$wts <- "w"
          if (by == "g") args$by <- "g"
          r <- do.call(avg_slopes, args)
          grp <- if (by == "g") as.character(r$g) else rep("all", nrow(r))
          out[[length(out) + 1]] <- data.frame(model = mn, slope = s, weights = wt, by = by,
                                               type = tp, group = grp,
                                               estimate = r$estimate, std_error = r$std.error)
        }
      }
    }
  }
}
res <- do.call(rbind, out)
write.csv(res, stdout(), row.names = FALSE, quote = FALSE)
