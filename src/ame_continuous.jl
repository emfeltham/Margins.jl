# ame_continuous.jl - DROP-IN REPLACEMENT

###############################################################################
# Ultra-optimized continuous AME computation 
###############################################################################

"""
    _ame_continuous!(
        β::Vector{Float64},
        cholΣβ::Cholesky{Float64,<:AbstractMatrix{Float64}},
        X::AbstractMatrix{Float64},
        Xdx::AbstractMatrix{Float64},
        dinvlink::Function,
        d2invlink::Function,
        ws::AMEWorkspace,
    ) -> (Float64, Float64, Vector{Float64})

Ultra-optimized computation with:
- SIMD vectorization throughout
- Optimized BLAS usage
- Minimal memory allocations
- Cache-efficient operations
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
    # Unpack workspace for performance
    η, dη, arr1, arr2, buf1, buf2 = ws.η, ws.dη, ws.arr1, ws.arr2, ws.buf1, ws.buf2
    n, p = size(X)

    # Optimized linear predictor computation (BLAS Level 2)
    mul!(η, X, β)      # η = X * β
    mul!(dη, Xdx, β)   # dη = Xdx * β

    # Ultra-fast vectorized AME computation with SIMD
    sum_ame = 0.0
    @inbounds @simd ivdep for i in 1:n
        ηi, dηi = η[i], dη[i]
        mp = dinvlink(ηi)
        mpp = d2invlink(ηi)
        
        sum_ame += mp * dηi
        arr1[i] = mpp * dηi    # For gradient computation
        arr2[i] = mp           # For gradient computation
    end
    ame = sum_ame / n

    # Optimized gradient assembly (BLAS Level 2)
    mul!(buf1, X', arr1)      # buf1 = X' * arr1
    mul!(buf2, Xdx', arr2)    # buf2 = Xdx' * arr2
    
    # Combine and scale with SIMD
    inv_n = 1.0 / n
    @inbounds @simd ivdep for i in 1:p
        buf1[i] = (buf1[i] + buf2[i]) * inv_n
    end
    grad = buf1  # Reuse buf1 as gradient

    # Efficient standard error computation
    mul!(buf2, cholΣβ.U, grad)  # buf2 = U * grad
    se = norm(buf2)             # ||U * grad||

    return ame, se, copy(grad)  # Copy since grad references workspace
end
