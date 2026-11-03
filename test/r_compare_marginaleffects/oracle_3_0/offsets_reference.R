# Reference values for test/validation/test_oracle_3_0.jl, "offsets".
#
# Poisson with an exposure offset, η = x'β + log(expo), unweighted and with the non-integer weights
# `sw`: AAP (overall and by g), AME (response and link scale, overall and by g), and at a reference
# grid with the exposure given explicitly the MEM and the adjusted predictions (APM).
#
#   Rscript offsets_reference.R > offsets_reference.csv
#
# How marginaleffects 1.0.0 handles the offset (the Julia side passes it as `offset=` or an
# `offset` grid column, on the offset's scale, log(expo)):
#   - With `offset(log(expo))` in the formula, `expo` is an ordinary variable of the model data, so
#     `newdata` must carry it. `datagrid()` fills a variable it is not given with its mean, so a grid
#     without `expo` silently uses offset log(mean(expo)), not the mean offset; it is always given
#     here. With `glm(..., offset = log(expo))` instead, a grid without `expo` is an error.
#   - `slopes()` also differentiates with respect to `expo` (dY/dexpo) unless `variables` excludes it.
#   - `predictions()` on a GLM defaults to type = "invlink(link)", which has no SE; type = "response"
#     is used throughout.
# With a log link and an intercept the unweighted AAP equals mean(cnt) (score equation), as the
# Julia test also checks.
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

fits <- list(
  poisson_off   = glm(cnt ~ x * z + g + offset(log(expo)), data = d, family = poisson(), control = ctl),
  poisson_off_w = glm(cnt ~ x * z + g + offset(log(expo)), data = d, family = poisson(), weights = sw, control = ctl)
)
vars <- c("x", "z", "g")

out <- list()
add <- function(model, quantity, term, contrast, by, type, gx, gexpo, r) {
  out[[length(out) + 1]] <<- data.frame(model = model, quantity = quantity, term = term,
    contrast = contrast, by = by, type = type, grid_x = gx, grid_expo = gexpo,
    estimate = sprintf("%.17g", r$estimate), std_error = sprintf("%.17g", r$std.error))
}
for (mn in names(fits)) {
  m <- fits[[mn]]
  wargs <- if (mn == "poisson_off_w") list(wts = "sw") else list()
  for (tp in c("response", "link")) {
    common <- list(m, type = tp, numderiv = "richardson")
    r <- do.call(avg_predictions, c(common, wargs))
    add(mn, "aap", "", "", "all", tp, NA, NA, r)
    r <- do.call(avg_predictions, c(common, wargs, list(by = "g")))
    for (k in seq_len(nrow(r))) add(mn, "aap", "", "", as.character(r$g[k]), tp, NA, NA, r[k, ])
    r <- do.call(avg_slopes, c(common, wargs, list(variables = vars, eps = 1e-4)))
    for (k in seq_len(nrow(r))) add(mn, "ame", r$term[k], r$contrast[k], "all", tp, NA, NA, r[k, ])
    r <- do.call(avg_slopes, c(common, wargs, list(variables = c("x", "z"), by = "g", eps = 1e-4)))
    for (k in seq_len(nrow(r))) add(mn, "ame", r$term[k], r$contrast[k], as.character(r$g[k]), tp, NA, NA, r[k, ])
    # Reference grids: one exposure for the grid, and one per row
    grids <- list(datagrid(model = m, x = 1.5, z = 0.25, g = "b", expo = 3),
                  datagrid(model = m, x = c(1, 2), z = 0.25, g = "b", expo = c(1, 5)))
    for (gr in grids) {
      r <- do.call(slopes, c(common, list(variables = vars, newdata = gr, eps = 1e-4)))
      for (k in seq_len(nrow(r))) add(mn, "mem", r$term[k], r$contrast[k], "all", tp, r$x[k], r$expo[k], r[k, ])
      r <- do.call(predictions, c(common, list(newdata = gr)))
      for (k in seq_len(nrow(r))) add(mn, "apm", "", "", "all", tp, r$x[k], r$expo[k], r[k, ])
    }
  }
}
res <- do.call(rbind, out)
write.csv(res, stdout(), row.names = FALSE, quote = FALSE)
