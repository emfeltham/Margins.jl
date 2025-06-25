# ame_continuous.jl

"""
    ame_continuous(
      df::DataFrame,
      model,
      x::Symbol;
      δ::Real = 1e-6,
      vcov = StatsBase.vcov
    ) -> AME

Compute the **Average Marginal Effect** of a continuous variable `x` by:
1. Building the base design matrix at the observed data,
2. Finite‐difference approximation ∂η/∂x ≈ (η(x+δ) − η(x−δ))/(2δ),
3. Applying the inverse‐link derivative,
4. Averaging over all observations,
5. Using the delta method to get a standard error and gradient.

# Arguments
- `df::DataFrame` — data used to fit `model`.
- `model`        — a fitted GLM/GLMM (e.g. from `MixedModels.jl`).
- `x::Symbol`    — name of the continuous predictor.
- `δ::Real`      — finite‐difference step (default 1e−6).
- `vcov`         — function extracting fixed‐effects covariance from `model`.

# Returns
An `AME` struct with fields `(ame, se, grad, n, η_base, μ_base)`.
"""
function ame_continuous(
    df::DataFrame,
    model,
    x::Symbol;
    δ::Real = 1e-6,
    vcov = vcov
)
    # 1) analytic link‐derivs
    invlink, dinvlink, d2invlink = link_functions(model)

    # 2) base design, β, Σβ
    n      = nrow(df)
    X0     = fixed_X(model, df)        # n×p
    β      = coef(model)               # p-vector
    Σβ     = vcov(model)               # p×p

    # 3) base η, μ, dμ, d²μ
    η0     = X0 * β                    # n-vector
    μ0     = invlink.(η0)
    dμ     = dinvlink.(η0)             # μ′(η)
    d2μ    = d2invlink.(η0)            # μ″(η)

    # 4) shallow‐copy + safe perturb of single column
    df_plus  = copy(df)
    df_plus[!, x] = df[!, x] .+ δ
    df_minus = copy(df)
    df_minus[!, x] = df[!, x] .- δ

    # 5) only two extra design matrices
    X_plus  = fixed_X(model, df_plus)
    X_minus = fixed_X(model, df_minus)

    # 6) per-obs ∂η/∂x via centered‐difference on the *design*
    Xdiff   = (X_plus .- X_minus) ./ (2δ)  # n×p
    δη_δx   = Xdiff * β                   # n-vector

    # 7) point estimate: average of dμ·∂η/∂x
    ame_val = mean(dμ .* δη_δx)

    # 8) vectorized gradient:
    #    ∇AME = 1/n [ X0' * (d²μ .* δη_δx)  +  Xdiff' * dμ ]
    grad    = (X0' * (d2μ .* δη_δx) .+ Xdiff' * dμ) ./ n

    # 9) delta‐method SE
    varAME  = dot(grad, Σβ * grad)
    seAME   = sqrt(varAME)

    # 10) return the same AME struct
    return AME(
        x, ame_val, seAME, grad, n, η0, μ0,
        string(model.resp.d), string(model.resp.link)
    )
end
