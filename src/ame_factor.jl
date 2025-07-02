# ame_factor.jl - DROP-IN REPLACEMENT

###############################################################################
# Optimized categorical AMEs with pre-allocated buffers
###############################################################################

# Helper for covariance multiplication (maintains compatibility)
_cov_mul(Σ::AbstractMatrix, g) = Σ * g
_cov_mul(C::LinearAlgebra.Cholesky, g) = Matrix(C) * g

"""
Optimized pairwise AME computation with buffer reuse.
"""
function _ame_factor_pair_matrix_fast!(
    Xj::Matrix, Xk::Matrix, workdf::DataFrame,
    fe_rhs, β, Σβ_or_chol,
    f::Symbol, lvl_i, lvl_j,
    invlink, dinvlink,
)
    # Build design matrices in-place using pre-allocated buffers
    fill!(workdf[!, f], lvl_i)
    modelmatrix!(Xj, fe_rhs, workdf)

    fill!(workdf[!, f], lvl_j)
    modelmatrix!(Xk, fe_rhs, workdf)

    # Vectorized computations
    n = size(Xj, 1)
    
    # Pre-allocate temporary vectors (could be moved to workspace for even better performance)
    ηj = Vector{Float64}(undef, n)
    ηk = Vector{Float64}(undef, n)
    μj = Vector{Float64}(undef, n)
    μk = Vector{Float64}(undef, n)
    μpj = Vector{Float64}(undef, n)
    μpk = Vector{Float64}(undef, n)
    
    # Compute linear predictors
    mul!(ηj, Xj, β)
    mul!(ηk, Xk, β)
    
    # Vectorized link function applications
    @inbounds @simd for i in 1:n
        μj[i] = invlink(ηj[i])
        μk[i] = invlink(ηk[i])
        μpj[i] = dinvlink(ηj[i])
        μpk[i] = dinvlink(ηk[i])
    end

    # Compute AME
    ame = (sum(μk) - sum(μj)) / n

    # Compute gradient efficiently
    grad = Vector{Float64}(undef, length(β))
    
    # Temporary vectors for matrix multiplication
    temp_j = Vector{Float64}(undef, length(β))
    temp_k = Vector{Float64}(undef, length(β))
    
    mul!(temp_j, Xj', μpj)
    mul!(temp_k, Xk', μpk)
    
    @inbounds @simd for i in 1:length(β)
        grad[i] = (temp_k[i] - temp_j[i]) / n
    end

    # Standard error
    se = sqrt(dot(grad, _cov_mul(Σβ_or_chol, grad)))
    
    return ame, se, grad
end

"""
Optimized baseline AME computation.
"""
function _ame_factor_baseline!(
    ame_d, se_d, grad_d,
    tbl0, fe_rhs, β, Σβ_or_chol,
    f::Symbol, invlink, dinvlink,
)
    lvls = levels(categorical(tbl0[f]))
    base = lvls[1]

    # Set up reusable work objects
    workdf = DataFrame(Tables.columntable(tbl0), copycols=true)
    n, p = nrow(workdf), length(β)
    
    # Pre-allocate buffers for matrix operations
    Xj = Matrix{Float64}(undef, n, p)
    Xk = Matrix{Float64}(undef, n, p)
    original_col = copy(workdf[!, f])

    # Process each level against baseline
    for lvl in lvls[2:end]
        ame, se, grad = _ame_factor_pair_matrix_fast!(
            Xj, Xk, workdf, fe_rhs, β, Σβ_or_chol,
            f, base, lvl, invlink, dinvlink
        )
        
        key = (base, lvl)
        ame_d[key] = ame
        se_d[key] = se  
        grad_d[key] = grad
    end

    # Restore original column
    workdf[!, f] = original_col
end

"""
Optimized all-pairs AME computation.
"""
function _ame_factor_allpairs!(
    ame_d, se_d, grad_d,
    tbl0, fe_rhs, β, Σβ_or_chol,
    f::Symbol, invlink, dinvlink,
)
    lvls = levels(categorical(tbl0[f]))

    # Set up reusable work objects
    workdf = DataFrame(Tables.columntable(tbl0), copycols=true)
    n, p = nrow(workdf), length(β)
    
    # Pre-allocate buffers
    Xj = Matrix{Float64}(undef, n, p)
    Xk = Matrix{Float64}(undef, n, p)
    original_col = copy(workdf[!, f])

    # Process all pairs
    for i in 1:length(lvls)-1, j in i+1:length(lvls)
        lvl_i, lvl_j = lvls[i], lvls[j]

        ame, se, grad = _ame_factor_pair_matrix_fast!(
            Xj, Xk, workdf, fe_rhs, β, Σβ_or_chol,
            f, lvl_i, lvl_j, invlink, dinvlink
        )
        
        key = (lvl_i, lvl_j)
        ame_d[key] = ame
        se_d[key] = se
        grad_d[key] = grad
    end

    # Restore original column
    workdf[!, f] = original_col
end
