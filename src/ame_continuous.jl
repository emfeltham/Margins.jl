# ame_continuous.jl - DROP-IN REPLACEMENT

###############################################################################
# Optimized continuous AME computation with better memory usage
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

Drop-in replacement with optimized computation:
- Better vectorization and BLAS usage
- Reduced memory allocations
- More efficient Cholesky factorization usage
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
    # Unpack workspace for better performance
    η, dη, arr1, arr2, buf1, buf2 = ws.η, ws.dη, ws.arr1, ws.arr2, ws.buf1, ws.buf2
    n, p = size(X)

    # Compute linear predictors with optimized BLAS calls
    mul!(η, X, β)      # η = X * β
    mul!(dη, Xdx, β)   # dη = Xdx * β

    # Vectorized computation of AME and intermediate arrays
    sum_ame = 0.0
    @inbounds @simd for i in 1:n
        ηi, dηi = η[i], dη[i]
        mp = dinvlink(ηi)
        mpp = d2invlink(ηi)
        
        sum_ame += mp * dηi
        arr1[i] = mpp * dηi    # For X' * arr1
        arr2[i] = mp           # For Xdx' * arr2
    end
    ame = sum_ame / n

    # Optimized gradient assembly using BLAS
    mul!(buf1, X', arr1)      # buf1 = X' * arr1
    mul!(buf2, Xdx', arr2)    # buf2 = Xdx' * arr2
    
    # Combine and scale in one pass
    @inbounds @simd for i in 1:p
        buf1[i] = (buf1[i] + buf2[i]) / n
    end
    grad = buf1  # Reuse buf1 as gradient

    # Efficient standard error using Cholesky factorization
    # se = ||U * grad|| where Σβ = U'U, avoiding matrix multiplication
    mul!(buf2, cholΣβ.U, grad)  # buf2 = U * grad
    se = norm(buf2)             # ||U * grad||

    return ame, se, copy(grad)  # Copy since grad is in workspace
end