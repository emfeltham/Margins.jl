# ame_numeric.jl

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
    vcov = StatsBase.vcov
)

    # Link functions
    invlink, dinvlink, d2invlink = link_functions(model)

    # 1) Base data & design
    n = nrow(df)
    X0   = fixed_X(model, df)               # n×p
    β  = coef(model)                        # p
    Σβ = vcov(model)                        # p×p

    η_base = X0 * β
    μ_base = invlink.(η_base)

    # 2) dμ/dη
    dpdη = dinvlink.(η_base)

    # 3) Build perturbed design matrices
    df_plus  = deepcopy(df); df_plus[!, x]  .+= δ
    df_minus = deepcopy(df); df_minus[!, x] .-= δ

    X_plus  = fixed_X(model, df_plus)
    X_minus = fixed_X(model, df_minus)

    η_plus  = X_plus  * β
    η_minus = X_minus * β

    # 4) Finite‐difference for ∂η/∂x
    δη_δx = (η_plus .- η_minus) ./ (2δ)

    # 5) Marginal effects per observation
    me_vec = dpdη .* δη_δx
    ame_val = mean(me_vec)

    # 6) Second derivative d²μ/dη²
    d2dη2 = d2invlink.(η_base)

    # 7) Gradient ∇AME (p-vector)
    p = length(β)
    grads = zeros(p, n)
    for i in 1:n
        Xi0      = vec(X0[i, :])
        Xi_plus  = vec(X_plus[i, :])
        Xi_minus = vec(X_minus[i, :])

        dηdx_dβ = (Xi_plus .- Xi_minus) ./ (2δ)
        Ai = dpdη[i]; Bi = d2dη2[i]; Di = δη_δx[i]

        grads[:, i] = Bi .* Xi0 .* Di .+ Ai .* dηdx_dβ
    end
    grad_AME = sum(grads, dims=2)[:,1] ./ n

    # 8) Delta‐method SE
    var_AME = dot(grad_AME, Σβ * grad_AME)
    se_AME  = sqrt(var_AME)

    # Report model information
    family = string(model.resp.d)  # e.g. "GLM.GlmResp{…}" implies Bernoulli, Poisson, etc.
    link = string(model.resp.link)       # e.g. "LogitLink()"

    return AME(x, ame_val, se_AME, grad_AME, n, η_base, μ_base, family, link)
end
