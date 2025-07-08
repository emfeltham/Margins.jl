# ame_continuous.jl - BLAS OPTIMIZED VERSION

###############################################################################
# Ultra-optimized continuous AME computation using pure BLAS operations
###############################################################################

"""
Ultra-optimized AME computation with zero allocations
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
    n, p = size(X)
    
    # Unpack workspace vectors
    η, dη = ws.η, ws.dη
    μp_vals, μpp_vals = ws.μp_vals, ws.μpp_vals
    grad_work = ws.grad_work
    temp1, temp2 = ws.temp1, ws.temp2
    
    # Compute linear predictors (BLAS Level 2)
    mul!(η, X, β)      # η = X * β
    mul!(dη, Xdx, β)   # dη = Xdx * β
    
    # Vectorized link function computations with numerical stability
    sum_ame = 0.0
    @inbounds for i in 1:n
        ηi = η[i]
        dηi = dη[i]
        
        # First-level check for numerical issues in linear predictors
        if !(isnan(ηi) || isinf(ηi) || isnan(dηi) || isinf(dηi))
            μp = dinvlink(ηi)
            μpp = d2invlink(ηi)
            
            # Second-level check for numerical issues in link function outputs
            if !(isnan(μp) || isinf(μp) || isnan(μpp) || isinf(μpp))
                sum_ame += μp * dηi
                μp_vals[i] = μp
                μpp_vals[i] = μpp * dηi  # Pre-multiply for gradient
            else
                μp_vals[i] = 0.0
                μpp_vals[i] = 0.0
            end
        else
            μp_vals[i] = 0.0
            μpp_vals[i] = 0.0
        end
    end
    ame = sum_ame / n
    
    # Ultra-optimized gradient computation using BLAS Level 2
    BLAS.gemv!('T', 1.0, X, μpp_vals, 0.0, temp1)      # temp1 = X' * μpp_vals
    BLAS.gemv!('T', 1.0, Xdx, μp_vals, 0.0, temp2)     # temp2 = Xdx' * μp_vals
    
    # Combine and scale in-place with BLAS Level 1
    inv_n = 1.0 / n
    BLAS.axpy!(1.0, temp1, temp2)                       # temp2 = temp1 + temp2
    BLAS.scal!(inv_n, temp2)                            # temp2 = temp2 * inv_n
    
    # Copy result with numerical stability check
    @inbounds @simd for i in 1:p
        grad_val = temp2[i]
        grad_work[i] = isnan(grad_val) || isinf(grad_val) ? 0.0 : grad_val
    end
    
    # Check if gradient is all zeros (numerical failure)
    grad_norm = BLAS.nrm2(grad_work)
    if grad_norm == 0.0 || isnan(grad_norm) || isinf(grad_norm)
        se = NaN
    else
        # Efficient standard error via Cholesky (use mul! for UpperTriangular)
        mul!(temp1, cholΣβ.U, grad_work)  # temp1 = U * grad
        se = BLAS.nrm2(temp1)
    end
    
    return ame, se, copy(grad_work)  # Only allocation: copy result
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

BLAS-optimized batch computation of AMEs for multiple continuous variables.
"""
function compute_continuous_ames_batch!(
    ipm::InplaceModeler,
    df::DataFrame,
    cts_vars::Vector{Symbol},
    β::AbstractVector,
    cholΣβ::LinearAlgebra.Cholesky,
    dinvlink::Function,
    d2invlink::Function,
    ws::AMEWorkspace,                 # ← workspace is *passed in*
)
    n, p   = size(ws.X_base)
    k      = length(cts_vars)

    ames   = Vector{Float64}(undef, k)
    ses    = Vector{Float64}(undef, k)
    grads  = Vector{Vector{Float64}}(undef, k)

    # ensure each variable has a perturbation vector
    for v in cts_vars
        ws.pert_data[v] = Vector{Float64}(undef, n)
    end

    # ------------------------------------------------------------------------
    for (j, var) in enumerate(cts_vars)
        orig = ws.base_tbl[var]
        pert = ws.pert_data[var]

        # compute step h
        maxabs = maximum(abs, orig)
        h   = clamp(sqrt(eps(Float64))*max(1, maxabs*0.01), 1e-8, maxabs*0.1)
        invh = 1/h

        @inbounds @simd for i in 1:n
            pert[i] = orig[i] + h
        end

        pert_tbl = merge(ws.base_tbl, (var => pert,))
        modelmatrix!(ipm, pert_tbl, ws.Xdx)

        BLAS.axpy!(-1.0, vec(ws.X_base), vec(ws.Xdx))   # Xdx ← Xdx – X_base
        BLAS.scal!(invh, vec(ws.Xdx))                   # Xdx ← Xdx/h

        ame, se, g = _ame_continuous!(
            β, cholΣβ, ws.X_base, ws.Xdx,
            dinvlink, d2invlink, ws
        )
        ames[j]  = ame
        ses[j]   = se
        grads[j] = g
    end
    return ames, ses, grads
end
