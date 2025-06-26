###############################################################################
# ame_continuous.jl
#
# Analytic AME for a continuous predictor `x`, mirroring Stata’s `margins`.
###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
# utilities already in your code base
#   • link_functions(model) -> (invlink, dinvlink, d2invlink)
#   • fixed_effects_form(model), fixed_X(model, df)
#   • mueta2 definitions
#   • struct AME  (ame, se, grad, n, η_base, μ_base, dist, link)
# ─────────────────────────────────────────────────────────────────────────────

function _ame_continuous(df, model, x, fe_form, β, dinvlink, d2invlink, vcov)
    # --- prepare design and link derivatives ---
    X0  = modelmatrix(fe_form, df)      # n×p
    η0  = X0 * β                        # linear predictor
    dμ  = dinvlink.(η0)                # μ′(η)
    d2μ = d2invlink.(η0)               # μ″(η)
    n, p = size(X0)

    # --- containers for per-observation derivatives ---
    δη_δx  = similar(η0)
    XdxTdμ = zeros(eltype(β), p)

    # --- loop over observations ---
    for i in 1:n
        df_row = df[i:i, :]
        x_val  = df_row[1, x]

        # analytic ∂η/∂x via AD
        fη(v) = begin
            tmp = copy(df_row)
            tmp[!, x] .= v
            (modelmatrix(fe_form, tmp) * β)[1]
        end
        δη_δx[i] = ForwardDiff.derivative(fη, x_val)

        # Jacobian of design row wrt x for Δ-method
        fX(v) = begin
            tmp = copy(df_row)
            tmp[!, x] .= v
            vec(modelmatrix(fe_form, tmp))
        end
        ∂X_∂x = ForwardDiff.derivative(fX, x_val)
        @inbounds XdxTdμ .+= ∂X_∂x .* dμ[i]
    end

    # --- point estimate ---
    ame_val = mean(dμ .* δη_δx)

    # --- Δ-method gradient wrt β ---
    grad = (X0' * (d2μ .* δη_δx) .+ XdxTdμ) ./ n

    # --- standard error via Δ-method ---
    Σβ = vcov(model)
    se = sqrt(dot(grad, Σβ * grad))

    return ame_val, se, grad
end
