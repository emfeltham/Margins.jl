# Unified prediction and gradient computation for Margins.jl.
#
# This file consolidates the core prediction computation patterns that were
# duplicated across multiple files. It handles both η (link) and μ (response)
# scales with proper chain rule application for GLMs.
#
# All functions maintain the zero-allocation characteristics where possible
# and preserve exact statistical behavior.

"""
    compute_predictions_batch!(results, gradients, compiled, data_nt, β, link, scale, row_buf)

Batch computation for multiple observations with optimized memory usage.

This function is optimized for population margins where we need predictions
and gradients for many observations. It writes results directly into 
pre-allocated arrays to minimize allocation overhead.

# Arguments
- `results::Vector{T}`: Pre-allocated vector to store predictions (modified in-place)
- `gradients::Matrix{Float64}`: Pre-allocated matrix to store gradients (modified in-place)
- `compiled`: CompiledFormula  
- `data_nt::NamedTuple`: Data in columntable format
- `β::Vector{Float64}`: Model coefficients
- `link`: GLM link function
- `scale::Symbol`: Either `:link` or `:response`
- `row_buf::Vector{Float64}`: Pre-allocated buffer for model row

# Performance
Designed for production use with large datasets:
- In-place writes to pre-allocated arrays
- Minimal allocation overhead per observation
- Vectorized operations where possible
"""
function compute_predictions_batch!(
    results::AbstractVector{T}, gradients::Matrix{Float64},
    compiled, data_nt, β::Vector{Float64}, link, scale::Symbol,
    row_buf::Vector{Float64}
) where T<:Real
    
    n_obs = length(results)
    
    if scale === :response
        # Response scale with chain rule
        for i in 1:n_obs
            modelrow!(row_buf, compiled, data_nt, i)
            η = dot(row_buf, β)
            
            results[i] = GLM.linkinv(link, η)
            dμ_dη = GLM.mueta(link, η)
            
            # Store gradient: ∂μ/∂β = dμ_dη * ∂η/∂β
            gradients[i, :] .= dμ_dη .* row_buf
        end
    else
        # Link scale (simpler case)
        for i in 1:n_obs
            modelrow!(row_buf, compiled, data_nt, i)
            η = dot(row_buf, β)
            
            results[i] = η
            gradients[i, :] .= row_buf
        end
    end
end
 
# End of predictions.jl
