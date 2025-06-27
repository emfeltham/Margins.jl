###############################################################################
# ame_continuous.jl (Dual‑based, interaction‑aware, single‑pass)
###############################################################################

using ForwardDiff, StatsModels, GLM, LinearAlgebra, StatsBase

"""
    _ame_continuous(
        df::AbstractDataFrame,
        model,
        x::Symbol,
        fe_form,
        β::AbstractVector,
        vcov::Function = StatsBase.vcov,
    ) -> (Float64, Float64, Vector{Float64})

Compute the average marginal effect (AME) of a continuous predictor `x`, its
standard error, and Δ‑method gradient.  Inject a single Dual seed into `x`
so ForwardDiff propagates through every interaction **once**, avoiding the
original per‑row AD loop.
"""
function _ame_continuous(
    df::AbstractDataFrame,
    model,
    x::Symbol,
    fe_form,
    β::AbstractVector,
    vcov::Function = StatsBase.vcov,
)
    ###########################################################################
    # 1. Dual‑inject only the focal column (seed = 1)
    ###########################################################################
    df2 = copy(df)
    colvals = df2[!, x]
    df2[!, x] = ForwardDiff.Dual.(colvals, one(eltype(colvals)))

    ###########################################################################
    # 2. Build Dual‑typed design matrix and linear predictor
    ###########################################################################
    X₂ = modelmatrix(fe_form, df2)               # Matrix{Dual}
    η₂ = X₂ * β                                  # Vector{Dual}

    # Separate value and x‑derivative components in one broadcast
    X   = ForwardDiff.value.(X₂)                 # n×p Float64
    dXdx = map(u -> ForwardDiff.partials(u)[1], X₂)  # n×p Float64

    η   = ForwardDiff.value.(η₂)                 # n‑vector Float64
    dηdx = map(u -> ForwardDiff.partials(u)[1], η₂)  # n‑vector Float64

    ###########################################################################
    # 3. Link‑function derivatives
    ###########################################################################
    _, dinvlink, d2invlink = link_functions(model)
    dμdη  = dinvlink.(η)     # μ′(η)
    d2μdη = d2invlink.(η)    # μ″(η)

    ###########################################################################
    # 4. AME  (average of ∂μ/∂x_i)
    ###########################################################################
    dμdx  = dμdη .* dηdx
    ame_val = mean(dμdx)

    ###########################################################################
    # 5. Δ‑method gradient wrt β (vectorised)
    #     grad = ( Xᵀ * (μ″ · ∂η/∂x)   +   (∂X/∂x)ᵀ * μ′ ) / n
    ###########################################################################
    n = length(η)
    grad = (X' * (d2μdη .* dηdx) + dXdx' * dμdη) ./ n

    ###########################################################################
    # 6. Standard error via Δ‑method
    ###########################################################################
    Vβ = vcov(model)
    se_val = sqrt(dot(grad, Vβ * grad))

    return ame_val, se_val, grad
end

#=
N.B.,

Replaced the nested-AD gradient with an analytic, fully vectorised expression:

$$
\nabla_{β}\widehat{AME}
   = \frac1n\Bigl[\,X^{\!T}\bigl(\mu''\!\odot\!\partial\eta/\partial x\bigr)
                   + (\partial X/\partial x)^{\!T}\bigl(\mu'\bigr)\Bigr],
$$

where all pieces are extracted in a **single** pass from the Dual-typed design matrix.
No nested Dual tags ⇒ the tag-ordering error disappears. Give this version a spin with your Iris test case.
=#