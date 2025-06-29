# ame_continuous.jl

###############################################################################
# 2. Continuous AME helper (analytic Δ-method on cached X, Xdx)
###############################################################################

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
standard error, and Δ-method gradient.  Inject a single Dual seed into `x`
so ForwardDiff propagates through every interaction **once**, avoiding the
original per-row AD loop.

    _ame_continuous(β, Σβ, X, Xdx, dinvlink, d2invlink)
Return (AME, SE, gradient) for **one** continuous variable, given

* `X`      – n×p design matrix  (Float64)
* `Xdx`    – n×p derivative-design for this variable
* `β`      – coefficient vector
* `Σβ`     – covariance of β
* `dinvlink`, `d2invlink` – link derivatives

    _ame_continuous(β, Σβ, X, Xdx, dinvlink, d2invlink)
"""
function _ame_continuous(β::Vector{Float64}, Σβ::AbstractMatrix{Float64},
                         X::Matrix{Float64}, Xdx::Matrix{Float64},
                         dinvlink::Function, d2invlink::Function)
    n   = size(X,1)
    η   = X * β
    dη  = Xdx * β
    μp  = dinvlink.(η)
    μpp = d2invlink.(η)

    ame  = mean(μp .* dη)
    grad = (X'*(μpp .* dη) + Xdx'*(μp)) ./ n
    se   = sqrt(dot(grad, Σβ * grad))
    return ame, se, grad
end
