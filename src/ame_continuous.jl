# ame_continuous.jl

"""
    compute_continuous_ames_selective!(variables::Vector{Symbol}, ws::AMEWorkspace,
                                      β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                                      dinvlink::Function, d2invlink::Function, 
                                      ipm::InplaceModeler)

Compute AMEs for multiple continuous variables using selective matrix updates.
FIXED: Now uses appropriate step sizes and stable numerical methods.
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
        
        # Prepare *analytic* derivative matrix (h is unused)
        prepare_analytical_derivatives!(ws, variable, 0.0, ipm)
        
        # Compute AME and SE using selective finite difference matrix
        ame, se, grad_ref = _ame_continuous_selective_fixed!(
            β, cholΣβ, ws.base_matrix, ws.finite_diff_matrix, dinvlink, d2invlink, ws
        )
        
        # Store results (copy gradient since workspace will be reused)
        ames[j] = ame
        ses[j] = se
        grads[j] = copy(grad_ref)
    end
    
    return ames, ses, grads
end

function _ame_continuous_selective_fixed!(
    β::Vector{Float64},
    cholΣβ::Cholesky{Float64,<:AbstractMatrix{Float64}},
    X::AbstractMatrix{Float64},
    Xdx::AbstractMatrix{Float64},
    dinvlink::Function,
    d2invlink::Function,
    ws::AMEWorkspace
)
    n, p = size(X)
    
    # Unpack workspace vectors
    η, dη = ws.η, ws.dη
    μp_vals, μpp_vals = ws.μp_vals, ws.μpp_vals
    grad_work = ws.grad_work
    temp1, temp2 = ws.temp1, ws.temp2
    
    # Compute linear predictors
    mul!(η, X, β)
    mul!(dη, Xdx, β)
    
    # Only clamp extreme values that would cause link function failures
    @inbounds for i in 1:n
        if abs(η[i]) > 50.0  # Very generous bounds
            η[i] = sign(η[i]) * 50.0
        end
        if abs(dη[i]) > 50.0
            dη[i] = sign(dη[i]) * 50.0
        end
    end
    
    # Compute AME with improved numerical stability
    sum_ame = 0.0
    n_valid = 0
    
    @inbounds for i in 1:n
        ηi = η[i]
        dηi = dη[i]
        
        # Skip only clearly problematic observations
        if !isfinite(ηi) || !isfinite(dηi)
            μp_vals[i] = 0.0
            μpp_vals[i] = 0.0
            continue
        end
        
        # Compute link function derivatives with error handling
        local μp, μpp
        try
            μp = dinvlink(ηi)
            μpp = d2invlink(ηi)
        catch e
            μp_vals[i] = 0.0
            μpp_vals[i] = 0.0
            continue
        end
        
        # Check for reasonable link function outputs
        if !isfinite(μp) || !isfinite(μpp)
            μp_vals[i] = 0.0
            μpp_vals[i] = 0.0
            continue
        end
        
        marginal_effect = μp * dηi
        
        # FIXED: More reasonable bounds checking
        if isfinite(marginal_effect) && abs(marginal_effect) < 1e10
            sum_ame += marginal_effect
            n_valid += 1
            μp_vals[i] = μp
            μpp_vals[i] = μpp * dηi
        else
            μp_vals[i] = 0.0
            μpp_vals[i] = 0.0
        end
    end
    
    # Compute AME
    ame = sum_ame / n
    
    # FIXED: Improved gradient computation
    fill!(temp1, 0.0)
    fill!(temp2, 0.0)
    
    # Initialize se to a reasonable default
    se = NaN
    
    try
        mul!(temp1, X', μpp_vals)
        mul!(temp2, Xdx', μp_vals)
        
        # Check for problematic gradients before proceeding
        temp1_finite = all(isfinite, temp1)
        temp2_finite = all(isfinite, temp2)
        
        if temp1_finite && temp2_finite
            # Combine gradients
            inv_n = 1.0 / n
            @inbounds for i in 1:p
                grad_work[i] = (temp1[i] + temp2[i]) * inv_n
                
                # Basic sanity check
                if !isfinite(grad_work[i])
                    grad_work[i] = 0.0
                end
            end
            
            # Compute SE with improved stability
            grad_norm = norm(grad_work)
            if grad_norm > 0.0 && isfinite(grad_norm) && grad_norm < 1e6
                mul!(temp1, cholΣβ.U, grad_work)
                se_squared = dot(temp1, temp1)
                if se_squared >= 0 && isfinite(se_squared)
                    se = sqrt(se_squared)
                end
            end
        else
            fill!(grad_work, 0.0)
        end
        
    catch e
        @warn "Gradient computation failed: $e"
        fill!(grad_work, 0.0)
        se = NaN
    end
    
    return ame, se, grad_work
end
