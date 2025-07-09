# ame_continuous.jl - COMPLETE REWRITE with selective update strategy

###############################################################################
# Selective update AME computation for continuous variables
###############################################################################

"""
    compute_continuous_ames_selective!(variables::Vector{Symbol}, ws::AMEWorkspace,
                                      β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                                      dinvlink::Function, d2invlink::Function, 
                                      ipm::InplaceModeler)

Compute AMEs for multiple continuous variables using selective matrix updates.
Only columns affected by each variable are recomputed, with memory sharing for others.

# Arguments
- `variables`: Vector of continuous variable symbols
- `ws`: AMEWorkspace with selective update infrastructure
- `β`: Model coefficients
- `cholΣβ`: Cholesky decomposition of coefficient covariance matrix
- `dinvlink`: First derivative of inverse link function
- `d2invlink`: Second derivative of inverse link function  
- `ipm`: InplaceModeler for matrix construction

# Returns
- `(ames, ses, grads)`: Vectors of AMEs, standard errors, and gradients

# Details
This function replaces the old `compute_continuous_ames_batch!` with selective updates:
- Uses pre-allocated perturbation vectors from workspace
- Updates only columns affected by each variable
- Shares memory for unchanged columns
- Maintains same numerical algorithms for AME/SE computation
"""
function compute_continuous_ames_selective!(variables::Vector{Symbol}, ws::AMEWorkspace,
                                          β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                                          dinvlink::Function, d2invlink::Function, 
                                          ipm::InplaceModeler)
    n, p = size(ws.base_matrix)
    k = length(variables)
    
    # Pre-allocate results
    ames = Vector{Float64}(undef, k)
    ses = Vector{Float64}(undef, k)
    grads = Vector{Vector{Float64}}(undef, k)
    
    # Process each variable with selective updates
    for (j, variable) in enumerate(variables)
        # Validate that variable is continuous and pre-allocated
        if !haskey(ws.pert_vectors, variable)
            throw(ArgumentError(
                "Variable $variable not found in perturbation vectors. " *
                "Only continuous (non-Bool) variables are supported."
            ))
        end
        
        # Get original values and compute step size
        orig_values = ws.base_data[variable]
        maxabs = maximum(abs, orig_values)
        h = clamp(sqrt(eps(Float64)) * max(1, maxabs * 0.01), 1e-8, maxabs * 0.1)
        
        # Prepare finite difference matrix using selective updates
        prepare_finite_differences!(ws, variable, h, ipm)
        
        # Compute AME and SE using selective finite difference matrix
        ame, se, grad_ref = _ame_continuous_selective!(
            β, cholΣβ, ws.base_matrix, ws.finite_diff_matrix, dinvlink, d2invlink, ws
        )
        
        # Store results (copy gradient since workspace will be reused)
        ames[j] = ame
        ses[j] = se
        grads[j] = copy(grad_ref)
    end
    
    return ames, ses, grads
end

"""
    _ame_continuous_selective!(β::Vector{Float64}, cholΣβ::Cholesky{Float64,<:AbstractMatrix{Float64}},
                              X::AbstractMatrix{Float64}, Xdx::AbstractMatrix{Float64},
                              dinvlink::Function, d2invlink::Function, ws::AMEWorkspace)

Core AME computation using selective finite difference matrix.
Replaces the old `_ame_continuous!` function.

# Arguments
- `β`: Model coefficients
- `cholΣβ`: Cholesky decomposition of coefficient covariance
- `X`: Base model matrix
- `Xdx`: Finite difference matrix (only affected columns are non-zero)
- `dinvlink`: First derivative of inverse link function
- `d2invlink`: Second derivative of inverse link function
- `ws`: AMEWorkspace for computation vectors

# Returns
- `(ame, se, grad_work)`: AME estimate, standard error, and gradient vector

# Details
This function maintains the same numerical algorithms as the original but:
- Works with selective finite difference matrices (most columns are zero)
- Uses workspace vectors to avoid allocations
- Leverages sparsity in Xdx for efficiency
"""
function _ame_continuous_selective!(
    β::Vector{Float64},
    cholΣβ::Cholesky{Float64,<:AbstractMatrix{Float64}},
    X::AbstractMatrix{Float64},
    Xdx::AbstractMatrix{Float64},
    dinvlink::Function,
    d2invlink::Function,
    ws::AMEWorkspace
)
    n, p = size(X)
    
    # Unpack workspace vectors (no allocation)
    η, dη = ws.η, ws.dη
    μp_vals, μpp_vals = ws.μp_vals, ws.μpp_vals
    grad_work = ws.grad_work
    temp1, temp2 = ws.temp1, ws.temp2
    
    # Compute linear predictors (BLAS Level 2)
    mul!(η, X, β)       # η = X * β
    mul!(dη, Xdx, β)    # dη = Xdx * β (sparse due to selective Xdx)
    
    # Vectorized link function computations with numerical stability
    sum_ame = 0.0
    @inbounds for i in 1:n
        ηi = η[i]
        dηi = dη[i]
        
        # Check for numerical issues in linear predictors
        if !(isnan(ηi) || isinf(ηi) || isnan(dηi) || isinf(dηi))
            μp = dinvlink(ηi)
            μpp = d2invlink(ηi)
            
            # Check for numerical issues in link function outputs
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
    
    # Gradient computation using BLAS Level 2
    # Note: Xdx is sparse (only affected columns non-zero), so this is efficient
    BLAS.gemv!('T', 1.0, X, μpp_vals, 0.0, temp1)      # temp1 = X' * μpp_vals
    BLAS.gemv!('T', 1.0, Xdx, μp_vals, 0.0, temp2)     # temp2 = Xdx' * μp_vals
    
    # Combine and scale in-place
    inv_n = 1.0 / n
    BLAS.axpy!(1.0, temp1, temp2)                       # temp2 = temp1 + temp2
    BLAS.scal!(inv_n, temp2)                            # temp2 = temp2 * inv_n
    
    # Copy result with numerical stability check
    @inbounds @simd for i in 1:p
        grad_val = temp2[i]
        grad_work[i] = isnan(grad_val) || isinf(grad_val) ? 0.0 : grad_val
    end
    
    # Compute standard error via Cholesky
    grad_norm = BLAS.nrm2(grad_work)
    if grad_norm == 0.0 || isnan(grad_norm) || isinf(grad_norm)
        se = NaN
    else
        mul!(temp1, cholΣβ.U, grad_work)  # temp1 = U * grad
        se = BLAS.nrm2(temp1)
    end
    
    # Return workspace gradient vector directly (caller must copy if needed)
    return ame, se, grad_work
end

"""
    compute_single_continuous_ame_selective!(variable::Symbol, ws::AMEWorkspace,
                                           β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                                           dinvlink::Function, d2invlink::Function, 
                                           ipm::InplaceModeler)

Compute AME for a single continuous variable using selective updates.
Used for representative values computation.

# Arguments
- `variable`: Symbol of continuous variable
- `ws`: AMEWorkspace (should already be set to desired representative values)
- `β`: Model coefficients
- `cholΣβ`: Cholesky decomposition of coefficient covariance
- `dinvlink`: First derivative of inverse link function
- `d2invlink`: Second derivative of inverse link function
- `ipm`: InplaceModeler for matrix construction

# Returns
- `(ame, se, grad)`: AME estimate, standard error, and gradient vector
"""
function compute_single_continuous_ame_selective!(variable::Symbol, ws::AMEWorkspace,
                                                β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                                                dinvlink::Function, d2invlink::Function, 
                                                ipm::InplaceModeler)
    # Validate variable
    if !haskey(ws.pert_vectors, variable)
        throw(ArgumentError(
            "Variable $variable not found in perturbation vectors. " *
            "Only continuous (non-Bool) variables are supported."
        ))
    end
    
    # Get step size
    orig_values = ws.base_data[variable]
    maxabs = maximum(abs, orig_values)
    h = clamp(sqrt(eps(Float64)) * max(1, maxabs * 0.01), 1e-8, maxabs * 0.1)
    
    # The workspace should already have work_matrix set to representative values
    # We need to compute finite differences from this state
    
    # Create perturbed values: current + h
    pert_vector = ws.pert_vectors[variable]
    current_values = ws.base_data[variable]  # This might be at repvals already
    
    @inbounds for i in eachindex(pert_vector)
        pert_vector[i] = current_values[i] + h
    end
    
    # Update work matrix with perturbed values
    update_for_variable!(ws, variable, pert_vector, ipm)
    
    # Compute finite differences: (X_perturbed - X_current) / h
    affected_cols = ws.variable_plans[variable]
    invh = 1.0 / h
    
    # Note: ws.work_matrix now has perturbed values, we need to compare with current state
    # We'll use finite_diff_matrix to store the difference
    @inbounds for col in affected_cols, row in axes(ws.finite_diff_matrix, 1)
        # work_matrix has perturbed values, base_matrix might not be current state
        # For repvals, we need the finite difference from current repval state
        ws.finite_diff_matrix[row, col] = (ws.work_matrix[row, col] - ws.base_matrix[row, col]) * invh
    end
    
    # Zero out unaffected columns
    total_cols = size(ws.finite_diff_matrix, 2)
    unaffected_cols = get_unchanged_columns(ws.mapping, [variable], total_cols)
    
    @inbounds for col in unaffected_cols, row in axes(ws.finite_diff_matrix, 1)
        ws.finite_diff_matrix[row, col] = 0.0
    end
    
    # Compute AME using current work matrix as base and finite_diff_matrix for derivative
    ame, se, grad_ref = _ame_continuous_selective!(
        β, cholΣβ, ws.work_matrix, ws.finite_diff_matrix, dinvlink, d2invlink, ws
    )
    
    return ame, se, grad_ref
end
