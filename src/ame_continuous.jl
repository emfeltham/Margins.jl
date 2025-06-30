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
function _ame_continuous!(
    β::Vector{Float64},
    cholΣβ::Cholesky{Float64,<:AbstractMatrix{Float64}},
    X::AbstractMatrix{Float64},
    Xdx::AbstractMatrix{Float64},
    dinvlink::Function,
    d2invlink::Function,
    ws::AMEWorkspace,
)
    η, dη, arr1, arr2, buf1, buf2 = ws.η, ws.dη, ws.arr1, ws.arr2, ws.buf1, ws.buf2
    n, _ = size(X)

    # 2 BLAS calls to get η and dη
    mul!(η,  X,   β)
    mul!(dη, Xdx, β)

    # one pass for sum_ame and to fill arr1, arr2
    sum_ame = 0.0
    @inbounds @simd for i in 1:n
        let ηi = η[i], dηi = dη[i]
            mp  = dinvlink(ηi)
            mpp = d2invlink(ηi)
            sum_ame   += mp * dηi
            arr1[i]    = mpp * dηi     # for X' * arr1
            arr2[i]    = mp            # for Xdx' * arr2
        end
    end
    ame = sum_ame / n

    # assemble gradient via 2 more BLAS calls
    mul!(buf1,  X',  arr1)
    mul!(buf2, Xdx', arr2)
    grad = (buf1 .+= buf2) ./ n

    # use Cholesky factor for the Δ‐method SE
    # cholΣβ = cholesky(Σβ)  # done once up‐front
    # U is the upper factor so Σβ = U'U
    se = norm(cholΣβ.U * grad)

    return ame, se, grad
end
