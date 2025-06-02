Below is a self‐contained Julia example that shows how to:

1. Simulate a small dataset with a binary outcome, continuous predictors `x` and `z`, and a grouping factor `group`.
2. Fit a mixed‐effects logistic model with `MixedModels.jl`.
3. Compute a single AME at `z = z₀` (via `ame_numeric`).
4. Compute the difference in AMEs between two `z`‐values (via `ame_contrast_numeric`).

You can copy–paste this into a REPL or a script file. Make sure you have the packages installed (`DataFrames`, `CategoricalArrays`, `MixedModels`, etc.).

```julia
using Random, DataFrames, CategoricalArrays, MixedModels, StatsModels, StatsBase, LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# (1) Simulate a tiny dataset
# ─────────────────────────────────────────────────────────────────────────────
Random.seed!(123)

n_groups = 10
n_per    = 50
N        = n_groups * n_per

# Continuous predictors
x = randn(N)            # focal variable
z = 2 .* rand(N) .- 1   # “moderator” in range [−1, +1]

# A grouping factor
group = repeat(string.("G", 1:n_groups), inner = n_per)
group = categorical(group)  # convert to CategoricalArray

# True fixed‐effect coefficients
β0 = -0.5    # intercept
β1 =  1.2    # main effect of x
β2 =  0.8    # main effect of z
β3 = -1.5    # interaction x*z

# Simulate a random intercept for each group
σ_re = 0.7
re   = σ_re .* randn(n_groups)
re_map = Dict( string("G", i) => re[i] for i in 1:n_groups )

η = similar(x)
for i in 1:N
    η[i] = β0 + β1*x[i] + β2*z[i] + β3*x[i]*z[i] + re_map[string(group[i])]
end

# Logistic link → probability
p = 1 ./(1 .+ exp.(-η))

# Simulate binary outcome y ~ Bernoulli(p)
y = rand.(Bernoulli.(p))

# Build the DataFrame
df = DataFrame(
    y     = y,
    x     = x,
    z     = z,
    group = group
)

first(df, 5) |> println


# ─────────────────────────────────────────────────────────────────────────────
# (2) Fit a mixed‐effects logistic regression
# ─────────────────────────────────────────────────────────────────────────────
# Model: logit(Pr(y=1)) = 1 + x + z + x*z + (1 | group)
fm = fit!(
    GeneralizedLinearMixedModel(
        @formula(y ~ 1 + x*z + (1 | group)),
        df,
        Bernoulli()
    )
)

println(fm)  # show a brief summary: fixed‐effects and random‐effects


# ─────────────────────────────────────────────────────────────────────────────
# (3) Compute a single AME at, say, z = 0.0
# ─────────────────────────────────────────────────────────────────────────────
# We assume the default invlink = logistic, so no need to pass dinvlink/d2invlink

ame_result = ame_numeric(
    df, fm;
    x      = :x,
    z      = :z,
    z_val  = 0.0,      # freeze z at 0
    δ      = 1e-6,
    typical = mean
)

@show ame_result.ame       # the average ∂P/∂x at z=0
@show ame_result.se        # its standard error
@show ame_result.n         # sample size
println("First five η_base: ", ame_result.η_base[1:5])
println("First five μ_base: ", ame_result.μ_base[1:5])


# ─────────────────────────────────────────────────────────────────────────────
# (4) Compute ΔAME between z = −0.5 and z = +0.5
# ─────────────────────────────────────────────────────────────────────────────
z_low  = -0.5
z_high =  0.5

out_contrast = ame_contrast_numeric(
    df, fm;
    x      = :x,
    z      = :z,
    z_vals = (z_low, z_high),
    δ      = 1e-6,
    typical = mean
)

@show out_contrast.ame_low     # AME of x at z = -0.5
@show out_contrast.se_low
@show out_contrast.ame_high    # AME of x at z = +0.5
@show out_contrast.se_high
@show out_contrast.ame_diff    # AME_high - AME_low
@show out_contrast.se_diff

# (Optional) 95% Wald CI for the difference:
ci_lo = out_contrast.ame_diff - 1.96*out_contrast.se_diff
ci_hi = out_contrast.ame_diff + 1.96*out_contrast.se_diff
println("ΔAME(x | z = $z_high vs $z_low) = ",
    round(out_contrast.ame_diff, digits=3), " ± ",
    round(1.96*out_contrast.se_diff, digits=3),
    "  (95% CI [", round(ci_lo, digits=3), ", ", round(ci_hi, digits=3), "])"
)
```

#### Explanation of the example

1. **Data simulation**

   * We simulate $N = 10 \times 50 = 500$ observations.
   * Predictors:

     * `x` ∼ Normal(0, 1).
     * `z` ∼ Uniform(−1, +1).
     * `group` ∼ 10 levels, each gets a random intercept ∼ Normal(0, 0.7²).
   * True coefficients: β₀ = −0.5, β₁ = 1.2 (for `x`), β₂ = 0.8 (for `z`), β₃ = −1.5 (for `x⋅z`).
   * We compute η = β₀ + β₁·x + β₂·z + β₃·(x·z) + random intercept, then p = logistic(η), and draw y ∼ Bernoulli(p).

2. **Model fitting**

   ```julia
   fm = fit!(
     GeneralizedLinearMixedModel(
       @formula(y ~ 1 + x*z + (1 | group)),
       df,
       Bernoulli()
     )
   )
   ```

   This fits a mixed‐effects logistic regression with a random intercept for each `group`.

3. **Compute a single AME at z = 0**

   ```julia
   ame_result = ame_numeric(
       df, fm;
       x      = :x,
       z      = :z,
       z_val  = 0.0,
       δ      = 1e-6,
       typical = mean
   )
   ```

   * We freeze `z=0.0` for every observation, build the “base” design matrix, then numerically approximate ∂η/∂x via central difference of `x ± δ`.
   * The returned `ame_result` is of type `AME`, so you can access:

     * `ame_result.ame`: the average marginal effect $\frac{1}{n}\sum_{i=1}^n \partial P_i/\partial x_i$ at `z=0.0`.
     * `ame_result.se`: the delta‐method standard error.
     * `ame_result.grad`: the p‐vector ∇₍β₎\[AME] (in case you want to form contrasts later).
     * `ame_result.η_base`, `ame_result.μ_base`: the underlying linear predictor and fitted probability for each row at `z=0`.

4. **Compute the difference in AMEs between two z‐values**

   ```julia
   out_contrast = ame_contrast_numeric(
       df, fm;
       x      = :x,
       z      = :z,
       z_vals = (-0.5, 0.5),
       δ      = 1e-6,
       typical = mean
   )
   ```

   * This internally calls `ame_numeric` once at `z_low = -0.5` and once at `z_high = +0.5`, then constructs

     $$
       \Delta \mathrm{AME} = \mathrm{AME}(x \mid z=0.5) - \mathrm{AME}(x \mid z=-0.5),
     $$

     and computes its standard error via $\nabla_{\rm high} - \nabla_{\rm low}$.
   * The returned `out_contrast` is a `NamedTuple` with fields:

     * `ame_low`, `se_low`: AME of `x` when `z = -0.5`, and its SE.
     * `ame_high`, `se_high`: AME of `x` when `z = +0.5`, and its SE.
     * `ame_diff`, `se_diff`: the difference (`ame_high - ame_low`) and its SE.
     * `grad_low`, `grad_high`: the gradient vectors for each AME (useful if you want a joint covariance or more complex contrasts).
     * `n`: the sample size (same for both).
     * `names`: the vector of coefficient names from `fm` (for reference).

5. **Printing and interpretation**

   * The `@show` lines display the numeric value of each AME, its SE, and the difference.
   * We also compute a 95% Wald confidence interval for `ame_diff` as

     $$
       (\hat{\Delta} \pm 1.96 \times SE(\hat{\Delta})).
     $$

With these two functions and this example, you can adapt to your own data:

* Replace the data simulation with your real `DataFrame`.
* Fit your GLMM (perhaps with more covariates, different random‐effect structure, etc.).
* Call `ame_numeric` to get an `AME` at any single `z`‐value.
* Call `ame_contrast_numeric` to compare two `z`‐values of interest (e.g. the 10th vs. 90th percentile of your `z`).

Because we used finite differences, **any** formula involving `x` (e.g. `x^2`, `x*z*w`, `log(x)`, etc.) is handled automatically.
