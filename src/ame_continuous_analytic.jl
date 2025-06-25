###############################################################################
# ame_continuous_analytic.jl
#
# Analytic AME for a continuous predictor `x`, mirroring Stata’s `margins`.
###############################################################################

using ForwardDiff, StatsModels, DataFrames, LinearAlgebra, StatsBase

# ─────────────────────────────────────────────────────────────────────────────
# utilities already in your code base
#   • link_functions(model) -> (invlink, dinvlink, d2invlink)
#   • fixed_effects_form(model), fixed_X(model, df)
#   • mueta2 definitions
#   • struct AME  (ame, se, grad, n, η_base, μ_base, dist, link)
# ─────────────────────────────────────────────────────────────────────────────

"""
    ame_continuous_analytic(df, model, x; vcov = StatsBase.vcov) -> AME

Stata-style **average marginal effect** of the continuous variable `x`,
computed with analytic derivatives (no finite differences).

Works with any `StatsModels` formula – polynomials, splines, logs, interactions,
factor contrasts, offsets, random-effects (only fixed part is differentiated),
etc.  Falls back automatically to numeric differentiation only if AD fails on
a particular observation.

# Arguments
- `df::DataFrame` : the original data
- `model`         : a fitted `GLM`/`GLMM`/`LinearModel` (`MixedModels.jl` OK)
- `x::Symbol`     : column name of the continuous predictor of interest
- `vcov`          : a function extracting the fixed-effects covariance matrix
                    (defaults to `StatsBase.vcov`)

# Returns
`AME` with fields  
`ame, se, grad, n, η_base, μ_base, dist, link`.
"""
function ame_continuous_analytic(
    df::DataFrame,
    model,
    x::Symbol;
    vcov = StatsBase.vcov
)

    # --- link functions (μ, μ′, μ″) -------------------------------------------------
    invlink, dinvlink, d2invlink = link_functions(model)

    # --- fixed-effects design and fitted values at the observed data ---------------
    fe_form = fixed_effects_form(model)     # same coding the model used
    X0      = modelmatrix(fe_form, df)      # n×p
    β       = coef(model)                   # p-vector
    η0      = X0 * β                        # linear predictor (n-vector)
    μ0      = invlink.(η0)                  # mean response
    dμ      = dinvlink.(η0)                 # μ′(η)
    d2μ     = d2invlink.(η0)                # μ″(η)

    # --- containers ----------------------------------------------------------------
    n, p   = size(X0)
    δη_δx  = similar(η0)                    # per-obs derivative of η wrt x
    XdxTdμ = zeros(eltype(β), p)            # Σ_i (∂X_i/∂x)' dμ_i      (p-vector)

    # --- loop over observations --------------------------------------------------
    for i in 1:n
        # grab a 1×p DataFrame containing only row i
        df_row = df[i:i, :]
        x_val  = df_row[1, x]   # the current value of x

        # — analytic ∂η/∂x via AD on a one‐row DataFrame ——
        fη(v) = begin
            tmp = copy(df_row)
            tmp[!, x] .= v
            # modelmatrix on a 1‐row DF returns a 1×p matrix
            (modelmatrix(fe_form, tmp) * β)[1]
        end
        δη_δx[i] = ForwardDiff.derivative(fη, x_val)

        # — Jacobian of the design row wrt x (for Δ‐method) ——
        fX(v) = begin
            tmp = copy(df_row)
            tmp[!, x] .= v
            # vec(...) turns the 1×p matrix into a p‐vector
            vec(modelmatrix(fe_form, tmp))
        end
        ∂X_∂x_row = ForwardDiff.derivative(fX, x_val)

        # accumulate the second term of the gradient
        @inbounds XdxTdμ .+= ∂X_∂x_row .* dμ[i]
    end

    # --- point estimate -------------------------------------------------------------
    ame_val = mean(dμ .* δη_δx)

    # --- Δ-method gradient wrt β ----------------------------------------------------
    # ∇AME = 1/n [  X0' * (μ″ .* ∂η/∂x)  +  (∂X/∂x)' * μ′ ]
    grad = (X0' * (d2μ .* δη_δx) .+ XdxTdμ) ./ n

    # --- standard error -------------------------------------------------------------
    Σβ   = vcov(model)                      # p×p
    se   = sqrt(dot(grad, Σβ * grad))       # √(g' Σ g)

    # --- bundle result --------------------------------------------------------------
    fam = family(model)
    
    return AME(x, ame_val, se, grad, n, η0, μ0, string(fam.dist), string(fam.link))
end
