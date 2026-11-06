# Mixed Models

*Using Margins.jl with MixedModels.jl*

## Overview

Margins.jl supports linear and generalized linear mixed models fit with
[MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl) through a package extension that
loads when both packages are imported. Every analysis of the other model classes works on them:
`population_margins`, `profile_margins`, scenarios, groups, contrasts and second differences.

A mixed model's predictions depend on the random effects as well as on β, so a marginal effect of
a mixed model is not one quantity. The keyword `random_effects` says which one you want:

| `random_effects` | the random effects are | the estimates describe |
|:--|:--|:--|
| `:zero` (default) | set to 0 | a typical group: one whose random effects are 0 |
| `:conditional` | each row's predicted random effects b̂ | the groups in the data |
| `:marginal` | averaged over their estimated distribution | the population, averaged over groups |
| a NamedTuple, e.g. `(village = :conditional, rater = :marginal)` | a mode per grouping factor | e.g. the villages in the data, with a new rater from the rater population |

Under an identity link (an LMM, or a GLMM with an identity link) `:marginal` equals `:zero`, and
average effects of variables without random slopes are the same in all three modes; the choice
matters for predictions (`:conditional` vs the others) and, under a nonlinear link, for everything.
Under a nonlinear link the effect at random effects 0 is not the population-averaged effect: the
average of g⁻¹(η + z′b) over b is flatter than g⁻¹(η), so `:zero` overstates population-averaged
effects on the probability scale.

## Example

Ratings by 30 raters in 12 villages (crossed), a binary outcome, a random intercept and a random
slope on `w` for raters, and a random intercept for villages:

```julia
using Random, DataFrames, CategoricalArrays, MixedModels, Margins
rng = Random.Xoshiro(2026)
nr, nv, per = 30, 12, 4
d = DataFrame(rater = repeat(string.("r", lpad.(1:nr, 2, '0')), inner = nv * per),
              village = repeat(repeat(string.("v", lpad.(1:nv, 2, '0')), inner = per), nr))
n = nrow(d)
d.x = randn(rng, n); d.w = randn(rng, n)
d.treat = categorical(rand(rng, ["control", "treated"], n))
a_r = randn(rng, nr); s_r = 0.5 .* randn(rng, nr); a_v = 0.8 .* randn(rng, nv)
ri = parse.(Int, chop.(d.rater; head = 1, tail = 0)); vi = parse.(Int, chop.(d.village; head = 1, tail = 0))
η = -0.3 .+ 0.8 .* d.x .+ 0.4 .* d.w .+ 0.6 .* (d.treat .== "treated") .+ a_r[ri] .+ s_r[ri] .* d.w .+ a_v[vi]
d.y = rand.(rng) .< 1 ./ (1 .+ exp.(-η))

m = fit(MixedModel, @formula(y ~ x + w + treat + (1 + w | rater) + (1 | village)), d, Bernoulli();
        progress = false)
```

The average marginal effects of `x` and `w` on the probability scale under each mode:

```julia
julia> population_margins(m, d; vars = [:x, :w])                     # random_effects = :zero
EffectsResult: 2 population effects (N=1440)
Random effects: 0 for rater, village (a typical group)
───────────────────────────────────────────────────────────────────────
Variable Contrast         dy/dx   Std. Err.  [95% Conf.   Interval]
───────────────────────────────────────────────────────────────────────
x        dy/dx          0.16591     0.01438     0.13774     0.19409
w        dy/dx          0.10831     0.01769     0.07364     0.14299
───────────────────────────────────────────────────────────────────────

julia> population_margins(m, d; vars = [:x, :w], random_effects = :conditional)
EffectsResult: 2 population effects (N=1440)
Random effects: predicted for rater, village
───────────────────────────────────────────────────────────────────────
Variable Contrast         dy/dx   Std. Err.  [95% Conf.   Interval]
───────────────────────────────────────────────────────────────────────
x        dy/dx            0.142     0.01057     0.12129     0.16272
w        dy/dx          0.08751     0.01063     0.06668     0.10834
───────────────────────────────────────────────────────────────────────

julia> population_margins(m, d; vars = [:x, :w], random_effects = :marginal)
EffectsResult: 2 population effects (N=1440)
Random effects: averaged over rater, village (population-averaged; SD 2.13)
───────────────────────────────────────────────────────────────────────
Variable Contrast         dy/dx   Std. Err.  [95% Conf.   Interval]
───────────────────────────────────────────────────────────────────────
x        dy/dx          0.13764     0.01115     0.11578      0.1595
w        dy/dx          0.08503     0.01495     0.05573     0.11433
───────────────────────────────────────────────────────────────────────
```

The typical-group effect of `x` (0.166) is about a fifth larger than both the effect averaged over
the raters and villages in the data (0.142) and the population-averaged one (0.138). The effect of
`w` includes each rater's predicted slope under `:conditional`, and the change in the spread of
z′b with `w` under `:marginal`.

## `random_effects = :conditional`

Each row's predicted random effects (the conditional modes b̂) enter the linear predictor, so a
row is predicted for its own rater and village. In a profile grid, set the grouping columns to
choose the groups; `re_terms` conditions on some grouping factors and sets the others to 0:

```julia
julia> profile_margins(m, d, DataFrame(x = 0.0, w = 0.0, treat = "control", village = ["v01", "v02"]);
                       type = :predictions, random_effects = :conditional, re_terms = :village)
PredictionsResult: 2 predictions at 2 profiles (N=1440)
Random effects: predicted for village; 0 for rater
──────────────────────────────────────────────────────────────────────────────────────
x       w       treat    village    Prediction   Std. Err.  [95% Conf.   Interval]
──────────────────────────────────────────────────────────────────────────────────────
0       0       control  v01           0.39221     0.06937     0.25623     0.52818
0       0       control  v02           0.48342     0.07322     0.33991     0.62692
──────────────────────────────────────────────────────────────────────────────────────
```

A level absent from the fit (a new village in a grid, or in new data) gets random effects 0 under
`re_new = :zero` (the default); `re_new = :error` rejects it.

A grid cell may hold a categorical mixture, as `means_grid` fills one into every categorical column
it is not given. Margins evaluates a mixture by setting the dummy columns to the shares (as Stata's
`atmeans` does), and the random effects follow the same rule: a mixture in a grouping column gives
the share-weighted average of the levels' b̂, and a mixture in a random-slope variable gives z at
the shares (for z′b̂, and for τ under `:marginal`). So `profile_margins(m, data, means_grid(data);
random_effects = :conditional)` predicts at the observed shares of every group. A nested factor's
grouping columns cannot hold mixtures (the shares of its cells are not determined by those of its
columns).

### Standard errors

A conditional quantity depends on b̂ as well as β̂, so its standard error should include the
prediction error of b̂. With `re_se = :joint` (the default) it does: the delta method runs over
(β, b) with the joint conditional covariance of (β̂ − β, b̂ − b) given the variance parameters
(Henderson's prediction-error covariance, whose β block is `vcov(model)`). Internally the random
effects become extra coefficients of the compiled formula (full dummy columns of each grouping
factor and their interactions with the slope variables), so the result's gradients and
`result.inference.vcov` are over (β, b), and `contrast`, groups, scenarios and second differences
of the result use the same joint covariance.

`re_se = :fixed` gives β-only standard errors, treating b̂ as known. These are not conservative:
they can be too small or far too large. An average over the observed groups is often known much
more precisely than x′β alone, since an error in β̂'s intercept is offset by the b̂ of the groups
in the data. In the example the average prediction is 0.583 with standard error 0.0108 under
`:joint` and 0.0499 under `:fixed`; for the AME of `x` they are 0.0106 and 0.0114. Use `:fixed`
when you need a covariance other than the model's own (a user `vcov`), or for a rank-deficient
fit; `:joint` requires the model's `vcov` and a full-rank X.

### Limits

The joint path holds a dense (p + q)² covariance, so it allows at most 2000 random-effect
coefficients (q). q counts every grouping factor, including those not conditioned on: the joint
covariance of β̂ and the conditioned b̂ involves all of them. Above that, `re_se = :joint` is an
error (use `re_se = :fixed`). `re_se = :fixed` above the cap uses a per-row offset instead, on
which two things are not supported: contrasts of a categorical variable with a random slope, and
population `scenarios` that set a variable with a random slope (a profile grid may set it).

## `random_effects = :marginal`

The estimates average the inverse link over the estimated random-effects distribution,
E[g⁻¹(η + z′b)] with b ~ N(0, Σ̂_b). z′b is normal with variance τ² = z′Σ̂_b z, so the average is
one-dimensional however many grouping factors are crossed or nested, and Margins computes it by
Gauss–Hermite quadrature. With random intercepts only, τ is the same for every row; with random
slopes it varies by row, and Margins computes τ from each evaluated row (z's columns are added to
the compiled formula with coefficient 0 and variance 0), so τ follows counterfactual rows too:
scenarios, contrasts of a categorical slope variable, and profile grid rows. The effect of a
random-slope variable includes the change in τ, from the formula's derivative, so any slope term
(`w^2` included) is exact.

```julia
julia> population_margins(m, d; type = :predictions, scenarios = (w = [-1.0, 1.0],),
                          random_effects = :marginal)
PredictionsResult: 2 population predictions (N=1440)
Scenarios: w
Random effects: averaged over rater, village (population-averaged; SD 2.13)
────────────────────────────────────────────────────────────
at_w      Prediction   Std. Err.  [95% Conf.   Interval]
────────────────────────────────────────────────────────────
-1           0.49584     0.05195     0.39403     0.59766
1            0.67216     0.04975     0.57465     0.76966
────────────────────────────────────────────────────────────
```

`re_nodes = :auto` (the default) starts from 16 nodes and doubles the rule until ŷ and its first
two derivatives agree with the next rule to 1e-10 at every η in [−20, 20], checked at the largest
τ over the rows the request evaluates; an integer fixes the rule. `metadata[:re_sd]` and
`metadata[:re_nodes]` record that τ and the rule (here 2.13 and 256 nodes). Very large τ (around
17 on the logit scale) does not converge within 1024 nodes, and the request says so.

Standard errors are delta-method on β, with the variance parameters held at their estimates.

## A mode per grouping factor

With crossed raters and villages, a natural question is about the villages in the data, each
rated by a new rater from the rater population:

```julia
julia> population_margins(m, d; vars = [:x], random_effects = (village = :conditional, rater = :marginal))
EffectsResult: 1 population effects (N=1440)
Random effects: predicted for village; averaged over rater (population-averaged; SD 1.99)
───────────────────────────────────────────────────────────────────────
Variable Contrast         dy/dx   Std. Err.  [95% Conf.   Interval]
───────────────────────────────────────────────────────────────────────
x        dy/dx           0.1393     0.01087       0.118      0.1606
───────────────────────────────────────────────────────────────────────
```

The villages' b̂ enter η (with joint standard errors under `re_se`), and the rater effects are
averaged over, so τ comes from the rater terms only. Unlisted factors are set to 0. The result
records `metadata[:random_effects] = :hybrid` and the averaged factors in
`metadata[:re_averaged]`. A NamedTuple with one mode is that mode with `re_terms` set to its
factors; `re_terms` cannot be combined with a NamedTuple. Village-level predictions follow with
`groups = :village`.

## What is held fixed

In every mode the variance parameters θ (the random-effect variances and correlations) are held at
their estimates, as in R's marginaleffects. With few groups θ is poorly
estimated, and the standard errors of `:marginal` quantities (which depend on θ through τ) and of
`:conditional` ones (through the shrinkage of b̂) are then too small.

## Recorded metadata

| key | value |
|:--|:--|
| `:random_effects` | `:zero`, `:conditional`, `:marginal` or `:hybrid` |
| `:re_terms` | the grouping factors used |
| `:re_averaged` | `:hybrid` only: the factors averaged over |
| `:re_se` | `:joint` or `:fixed`, when random effects are conditioned on |
| `:re_sd`, `:re_nodes` | when random effects are averaged over: the largest τ and the rule |

## Validation against R

`test/r_compare_marginaleffects/random_effects/` compares every mode against R, on an LMM and a
logit GLMM with `(1 + w | rater) + (1 | village)`: `:zero` and `:conditional` with
`re_se = :fixed` against marginaleffects 1.0.0 on lme4 2.1.0 fits (`re.form = NA` and
`re.form = NULL`), `re_se = :joint` against Henderson's equations built in R from lme4's X, Z, Λ
and working weights, and `:marginal` against R's adaptive `integrate` per row. Estimates and
standard errors agree to between 1e-15 and 3e-6 (the differences of the two fits).

!!! note "Standard errors from lme4"
    lme4's default `vcov` for a `glmer` fit (`use.hessian = TRUE`, a finite-difference Hessian
    over (θ, β)) differs from MixedModels' RX-based `vcov` (lme4's `use.hessian = FALSE`), by
    2.3% in the validation example. marginaleffects' default standard errors on a `glmer` fit
    therefore differ from Margins' by about that much; pass `vcov = vcov(m, use.hessian = FALSE)`
    in R to compare.

!!! note "Predicted random effects of GLMMs"
    For a GLMM, Margins takes b̂ = Λu from the fit (`m.u`, what `fitted(m)` uses), not
    `MixedModels.ranef`, which after a `fast = false` fit reads a stale working state
    (MixedModels 4.38.1).
