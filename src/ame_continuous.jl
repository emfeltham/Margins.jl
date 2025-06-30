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
    n, p = size(X)

    # allocate working vectors
    η         = Vector{Float64}(undef, n)
    dη        = Vector{Float64}(undef, n)
    arr_grad1 = Vector{Float64}(undef, n)  # μpp * dη
    arr_grad2 = Vector{Float64}(undef, n)  # μp

    # compute η and dη
    mul!(η,  X,  β)
    mul!(dη, Xdx, β)

    # one-pass: build AME sum and gradient components
    sum_ame = 0.0
    @inbounds @simd for i in 1:n
        mp           = dinvlink(η[i])
        mpp          = d2invlink(η[i])
        arr_grad1[i] = mpp * dη[i]
        arr_grad2[i] = mp
        sum_ame     += mp * dη[i]
    end

    ame = sum_ame / n

    # gradient = (X' * (μpp .* dη) + Xdx' * (μp)) / n
    buf1 = Vector{Float64}(undef, p)
    buf2 = Vector{Float64}(undef, p)
    mul!(buf1, X',    arr_grad1)
    mul!(buf2, Xdx',  arr_grad2)
    grad = (buf1 .+ buf2) ./ n

    se = sqrt(dot(grad, Σβ * grad))
    return ame, se, grad
end
