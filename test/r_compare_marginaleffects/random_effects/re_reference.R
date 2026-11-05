# Reference values for test/validation/test_oracle_random_effects.jl (`random_effects` for mixed
# models). R is not a test-time dependency: this script writes `re_data.csv` (the data, from R's
# RNG) and `re_reference.csv` (the values), which are committed.
#
#   R_LIBS=<private lib with lme4> Rscript re_reference.R
#
# Models (lme4, tight optimizer tolerances so the fits agree with MixedModels.jl to ~1e-7):
#   lmm:  y  ~ x + w + (1 + w | rater) + (1 | village), ML
#   glmm: yb ~ x + w + (1 + w | rater) + (1 | village), binomial logit, Laplace
# Quantities:
#   zero         marginaleffects avg_predictions / avg_slopes with re.form = NA (random effects 0);
#                SEs on β (vcov(m))
#   cond_fixed   the same with re.form = NULL (each row's predicted random effects); marginaleffects'
#                SEs are on β only, = Margins' re_se = :fixed
#   cond_joint   the AAP with the SE from Henderson's prediction-error covariance of (β̂, b̂ − b),
#                built here from lme4's X, Z, Λ and working weights: Cov(u − û, β̂ − β) = s²A⁻¹,
#                A = [Λ'Z'WZΛ + I, Λ'Z'WX; X'WZΛ, X'WX], b = Λu
#   marginal     (glmm) the population-averaged AAP and AME of x and w, E[plogis(η + τZ)] with
#                τᵢ² = zᵢ'Σzᵢ, by R's adaptive `integrate` (independent of Margins' Gauss–Hermite
#                rule); AMEs by numDeriv (Richardson) in x and w, SEs by numDeriv Jacobians in β
# Covariance of β: MixedModels' `vcov` is the RX-based one (σ²(X'V⁻¹X)⁻¹ at the conditional modes),
# which for a GLMM is lme4's `vcov(m, use.hessian = FALSE)`; lme4's default for glmer fits
# (`use.hessian = TRUE`) inverts a finite-difference Hessian over (θ, β), and differs here by 2.3%.
# Every SE below uses the RX-based one (passed to marginaleffects as `vcov =`), and the script
# checks that the β block of Henderson's covariance equals it (to 1e-14).
# Every number with 17 significant digits.
suppressPackageStartupMessages({library(lme4); library(marginaleffects); library(numDeriv)})
options(marginaleffects_safe = FALSE)

set.seed(20261104)
n <- 600
rater <- sample(sprintf("r%02d", 1:20), n, replace = TRUE)
village <- sample(sprintf("v%02d", 1:10), n, replace = TRUE)
x <- round(rnorm(n), 6); w <- round(rnorm(n), 6)
ra <- setNames(0.7 * rnorm(20), sprintf("r%02d", 1:20)); rs <- setNames(0.4 * rnorm(20), sprintf("r%02d", 1:20))
va <- setNames(0.6 * rnorm(10), sprintf("v%02d", 1:10))
eta <- -0.2 + 0.8 * x + 0.3 * w + ra[rater] + rs[rater] * w + va[village]
y <- round(0.5 + x + 0.3 * w + ra[rater] + rs[rater] * w + va[village] + rnorm(n), 6)
yb <- as.numeric(runif(n) < plogis(eta))
d <- data.frame(rater, village, x, w, y, yb)
write.csv(d, "re_data.csv", row.names = FALSE)
d <- read.csv("re_data.csv")    # what Julia reads

ctl_l <- lmerControl(optimizer = "nloptwrap", optCtrl = list(xtol_rel = 1e-14, ftol_rel = 1e-15, ftol_abs = 1e-15, maxeval = 1e5))
ctl_g <- glmerControl(optimizer = "nloptwrap", optCtrl = list(xtol_rel = 1e-14, ftol_rel = 1e-15, ftol_abs = 1e-15, maxeval = 1e5),
                      tolPwrss = 1e-12)
fits <- list(
  lmm  = lmer(y ~ x + w + (1 + w | rater) + (1 | village), data = d, REML = FALSE, control = ctl_l),
  glmm = glmer(yb ~ x + w + (1 + w | rater) + (1 | village), data = d, family = binomial(), control = ctl_g)
)

out <- list()
add <- function(model, mode, quantity, term, estimate, se)
  out[[length(out) + 1]] <<- data.frame(model = model, mode = mode, quantity = quantity, term = term,
                                         estimate = estimate, std_error = se)

for (nm in names(fits)) {
  m <- fits[[nm]]
  Vm <- as.matrix(vcov(m, use.hessian = FALSE))    # MixedModels' vcov (see the header)
  for (mode in c("zero", "cond_fixed")) {
    rf <- if (mode == "zero") NA else NULL
    p <- avg_predictions(m, re.form = rf, type = "response", vcov = Vm, numderiv = "richardson")
    add(nm, mode, "aap", "", p$estimate, p$std.error)
    s <- avg_slopes(m, variables = c("x", "w"), re.form = rf, type = "response", vcov = Vm, eps = 1e-4,
                    numderiv = "richardson")
    for (i in seq_len(nrow(s))) add(nm, mode, "ame", s$term[i], s$estimate[i], s$std.error[i])
  }
  # joint SE of the conditional AAP: Henderson
  X <- getME(m, "X"); Z <- getME(m, "Z"); L <- getME(m, "Lambda")
  mu <- fitted(m)
  W <- if (nm == "lmm") rep(1, n) else mu * (1 - mu)      # logit: working weights = dμ/dη
  s2 <- if (nm == "lmm") sigma(m)^2 else 1
  ZL <- as.matrix(Z %*% L)
  A <- rbind(cbind(crossprod(ZL, W * ZL) + diag(ncol(ZL)), crossprod(ZL, W * X)),
             cbind(crossprod(X, W * ZL), crossprod(X, W * X)))
  Ai <- s2 * solve(A)
  q <- ncol(ZL); p <- ncol(X)
  T <- rbind(cbind(as.matrix(L), matrix(0, q, p)), cbind(matrix(0, p, q), diag(p)))  # (u, β) → (b, β)
  Cb <- T %*% Ai %*% t(T)
  dmu <- if (nm == "lmm") rep(1, n) else mu * (1 - mu)
  g <- c(colMeans(dmu * as.matrix(Z)), colMeans(dmu * X))
  add(nm, "cond_joint", "aap", "", mean(mu), sqrt(drop(t(g) %*% Cb %*% g)))
  # check: the β block of the joint covariance is vcov(m)
  Vb <- as.matrix(vcov(m, use.hessian = FALSE))
  rel <- max(abs(Cb[q + 1:p, q + 1:p] - Vb)) / max(abs(Vb))
  message(sprintf("%s: β block of the joint covariance vs vcov(m, use.hessian = FALSE), max rel. difference %.2g", nm, rel))
  for (k in seq_len(p)) add(nm, "fit", "beta", colnames(X)[k], fixef(m)[k], sqrt(Vm[k, k]))
}

# :marginal on the GLMM: τᵢ² = Σ₀₀ + 2wᵢΣ₀₁ + wᵢ²Σ₁₁ (rater) + σ²_village
m <- fits$glmm
vc <- VarCorr(m); Sr <- as.matrix(vc$rater); sv2 <- as.numeric(vc$village)
tau <- function(wv) sqrt(Sr[1, 1] + 2 * wv * Sr[1, 2] + wv^2 * Sr[2, 2] + sv2)
avg1 <- function(eta, t) integrate(function(z) plogis(eta + t * z) * dnorm(z), -Inf, Inf,
                                   rel.tol = 1e-13, abs.tol = 0, subdivisions = 1000L)$value
mubar <- function(b, xv, wv) mean(mapply(function(xi, wi) avg1(b[1] + b[2] * xi + b[3] * wi, tau(wi)), xv, wv))
b <- unname(fixef(m)); V <- as.matrix(vcov(m, use.hessian = FALSE))
aap <- function(b) mubar(b, d$x, d$w)
ga <- numDeriv::grad(aap, b)
add("glmm", "marginal", "aap", "", aap(b), sqrt(drop(t(ga) %*% V %*% ga)))
ame_x <- function(b) mean(mapply(function(xi, wi) numDeriv::grad(function(xx) avg1(b[1] + b[2] * xx + b[3] * wi, tau(wi)), xi), d$x, d$w))
ame_w <- function(b) mean(mapply(function(xi, wi) numDeriv::grad(function(ww) avg1(b[1] + b[2] * xi + b[3] * ww, tau(ww)), wi), d$x, d$w))
for (term in c("x", "w")) {
  f <- if (term == "x") ame_x else ame_w
  gr <- numDeriv::grad(f, b)
  add("glmm", "marginal", "ame", term, f(b), sqrt(drop(t(gr) %*% V %*% gr)))
}

res <- do.call(rbind, out)
res$estimate <- trimws(formatC(res$estimate, digits = 17, format = "g"))
res$std_error <- trimws(formatC(res$std_error, digits = 17, format = "g"))
write.csv(res, "re_reference.csv", row.names = FALSE, quote = FALSE)
cat(sprintf("R %s, lme4 %s, marginaleffects %s, numDeriv %s\n", getRversion(), packageVersion("lme4"),
            packageVersion("marginaleffects"), packageVersion("numDeriv")))
