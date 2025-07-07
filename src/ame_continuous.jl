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
    temp1, temp2 = ws.temp_vec1, ws.temp_vec2
    
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
    
    # Create optimized workspace - ONLY major allocation
    ws = AMEWorkspace(n, p, df)
    
    # Build base design matrix ONCE
    modelmatrix!(ipm, ws.base_tbl, ws.X_base)
    
    # Pre-allocate perturbation data for ALL variables
    for var in cts_vars
        ws.pert_data[var] = Vector{Float64}(undef, n)
    end
    
    # Process each continuous variable with ZERO allocations
    for (j, var) in enumerate(cts_vars)
        # Get original values (direct reference - no allocation)
        original_vals = ws.base_tbl[var]
        
        # Compute perturbation step with numerical stability - OPTIMIZED
        max_abs = 0.0
        @inbounds @simd for i in 1:n
            val = abs(original_vals[i])
            max_abs = val > max_abs ? val : max_abs
        end
        
        h = sqrt(eps(Float64)) * max(1.0, max_abs * 0.01)
        h = max(h, 1e-8)  # Lower bound
        h = min(h, max_abs * 0.1)  # Upper bound
        inv_h = 1.0 / h
        
        # Create perturbed values in pre-allocated array
        pert_vals = ws.pert_data[var]
        @inbounds @simd for i in 1:n
            pert_vals[i] = original_vals[i] + h
        end
        
        # Create perturbed table (minimal allocation - NamedTuple merge)
        pert_tbl = merge(ws.base_tbl, (var => pert_vals,))
        
        # Build perturbed matrix directly into derivative workspace
        modelmatrix!(ipm, pert_tbl, ws.Xdx)
        
        # BLAS-optimized derivative computation - ZERO allocations!
        # ws.Xdx = (ws.Xdx - ws.X_base) * inv_h
        
        # Step 1: ws.Xdx = ws.Xdx - ws.X_base (BLAS axpy: y = a*x + y)
        BLAS.axpy!(-1.0, vec(ws.X_base), vec(ws.Xdx))
        
        # Step 2: ws.Xdx = ws.Xdx * inv_h (BLAS scal: x = a*x)  
        BLAS.scal!(inv_h, vec(ws.Xdx))
        
        # Compute AME for this variable
        ame, se, grad = _ame_continuous!(β, cholΣβ, ws.X_base, ws.Xdx, dinvlink, d2invlink, ws)
        
        ames[j] = ame
        ses[j] = se
        grads[j] = grad
    end
    
    return ames, ses, grads
end
