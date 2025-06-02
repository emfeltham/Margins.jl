# marginal_effect_curve_x()

1. **Grid of x values**
   We take  `K=100` points from  `minimum(df.x)`  to  `maximum(df.x)`.

2. **Holding z fixed**
   For each grid point  `xj`, we build a one‐row DataFrame (`row_base`) with

   * `x = xj`,
   * `z = z_val`,
   * every other column (e.g. random‐effect grouping, any other covariates) at its original value taken from the first row of `df`.

3. **Finite difference for ∂η/∂x**
   We build two more one‐row DataFrames:

   * `row_p`: `x = xj + δ`, `z = z_val`, others as in `row_base`.
   * `row_m`: `x = xj − δ`, `z = z_val`, others as in `row_base`.

   Then compute

   $$
     \eta_p = X_p ⋅ β,\quad \eta_m = X_m ⋅ β,\quad
     \frac{∂η}{∂x} ≈ \frac{\eta_p - \eta_m}{2δ}.
   $$

4. **Compute ∂μ/∂x**

   $$
     ∂μ/∂x = \bigl(\tfrac{dμ}{dη}\bigr)(η_0) ⋅ \bigl(∂η/∂x\bigr),
   $$

   where  $\mu_0 = \mathrm{invlink}(\eta_0)$.

5. **Delta‐method SE**
   We compute the gradient

   $$
     ∇_{β}[∂μ/∂x] 
       = B ⋅ X_0 ⋅ (∂η/∂x) 
       \;+\; A ⋅ \frac{X_p - X_m}{2δ},
   $$

   with

   * $A = \tfrac{dμ}{dη}(η_0)$,
   * $B = \tfrac{d^2μ}{dη^2}(η_0)$.

   Then $\mathrm{Var} = ∇' Σ_{β} ∇$ and $\mathrm{SE} = \sqrt{\mathrm{Var}}$.

6. **Output**
   We return a DataFrame with columns

   * `x_grid[j]`,
   * `me[j] = ∂μ/∂x` at that `x_grid[j]`,
   * `se_me[j] = SE` of that marginal effect.

You can then plot or tabulate “marginal effect of `x` as a function of `x` at fixed `z = z_val`,” including a 95% confidence ribbon via `ribbon = 1.96 * se_me`.
