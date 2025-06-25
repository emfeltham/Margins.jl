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
function ame_continuous_analytic(df::DataFrame,
                                 model,
                                 x::Symbol;
                                 vcov = StatsBase.vcov)

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

    # convenient closure: turn DataFrame row into NamedTuple (faster for AD)
    row_as_nt = Tables.row  # Tables.row(df, i) gives a NamedTuple view

    # --- loop over observations -----------------------------------------------------
    for i in 1:n
        row_nt = row_as_nt(df, i)
        x_val  = row_nt[x]

        # —— η_i(x) as a scalar function of x_i ————————————————
        fη(v) = begin
            # replace x in the NamedTuple with v (Dual or real)
            row2 = Base.merge(row_nt, (; x => v))
            (modelmatrix(fe_form, row2) * β)[1]
        end
        # analytic derivative ∂η/∂x  (ForwardDiff Dual, so fast & exact)
        δη_δx[i] = ForwardDiff.derivative(fη, x_val)

        # —— Jacobian of design row wrt x  (needed for Δ-method gradient) ——
        fX(v) = begin
            row2 = Base.merge(row_nt, (; x => v))
            vec(modelmatrix(fe_form, row2))  # p-vector
        end
        ∂X_∂x_row = ForwardDiff.derivative(fX, x_val)  # p-vector

        # accumulate Σ_i (∂X/∂x)' · dμ_i
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
    return AME(x, ame_val, se, grad, n, η0, μ0,
               string(model.resp.d), string(model.resp.link))
end
