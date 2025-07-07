# ame_continuous.jl - OPTIMIZED WITH EfficientModelMatrices.jl

###############################################################################
# Ultra-optimized continuous AME computation using InplaceModeler
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

Ultra-optimized computation with SIMD vectorization and optimized BLAS usage.
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

"""
    compute_continuous_ames_batch!(
        ipm::InplaceModeler,
        df::DataFrame,
        cts_vars::Vector{Symbol},
        β::Vector{Float64},
        cholΣβ::Cholesky{Float64,<:AbstractMatrix{Float64}},
        dinvlink::Function,
        d2invlink::Function,
    ) -> (Vector{Float64}, Vector{Float64}, Vector{Vector{Float64}})

Batch computation of AMEs for multiple continuous variables using shared workspace.
"""
function compute_continuous_ames_batch!(
    ipm::InplaceModeler,
    df::DataFrame,
    cts_vars::Vector{Symbol},
    β::Vector{Float64},
    cholΣβ::Cholesky{Float64,<:AbstractMatrix{Float64}},
    dinvlink::Function,
    d2invlink::Function,
)
    n, p = nrow(df), length(β)
    k = length(cts_vars)
    
    # Pre-allocate results
    ames = Vector{Float64}(undef, k)
    ses = Vector{Float64}(undef, k)
    grads = Vector{Vector{Float64}}(undef, k)
    
    # Shared workspace and matrices
    X = Matrix{Float64}(undef, n, p)
    Xdx = Matrix{Float64}(undef, n, p)
    ws = AMEWorkspace(n, p)
    
    # Build base design matrix once
    base_tbl = Tables.columntable(df)
    modelmatrix!(ipm, base_tbl, X)
    
    # Process each continuous variable
    for (j, var) in enumerate(cts_vars)
        # Create perturbed data
        original_vals = Float64.(base_tbl[var])
        h = sqrt(eps(Float64)) * max(1.0, maximum(abs, original_vals) * 0.01)
        perturbed_vals = original_vals .+ h
        
        # Zero-allocation matrix construction for perturbed data
        perturbed_tbl = merge(base_tbl, (var => perturbed_vals,))
        modelmatrix!(ipm, perturbed_tbl, Xdx)
        
        # Compute derivative matrix efficiently
        inv_h = 1.0 / h
        for row = 1:n
            @inbounds @simd for col in 1:p
                Xdx[row, col] = (Xdx[row, col] - X[row, col]) * inv_h
            end
        end
        
        # Compute AME for this variable
        ame, se, grad = _ame_continuous!(β, cholΣβ, X, Xdx, dinvlink, d2invlink, ws)
        
        ames[j] = ame
        ses[j] = se
        grads[j] = grad
    end
    
    return ames, ses, grads
end
