# Unified prediction and gradient computation for Margins.jl.
#
# This file consolidates the core prediction computation patterns that were
# duplicated across multiple files. It handles both η (link) and μ (response)
# scales with proper chain rule application for GLMs.
#
# All functions maintain the zero-allocation characteristics where possible
# and preserve exact statistical behavior.

"""
    PredictionWithGradient{T<:Real}

Container for a prediction value and its gradient vector.
Used to return both the computed prediction and the gradient
needed for delta-method standard error computation.
"""
struct PredictionWithGradient{T<:Real}
    value::T
    gradient::Vector{Float64}
    scale::Symbol  # :link or :response
end

"""
    compute_prediction_with_gradient(compiled, data_nt, row_idx, β, link, scale, row_buf) -> PredictionWithGradient

Compute prediction and gradient for a single observation.

This is the core prediction computation used throughout Margins.jl.
It consolidates the repeated pattern of:
1. Building model row with FormulaCompiler
2. Computing linear predictor η = X'β  
3. Applying inverse link function if scale=:response
4. Computing gradient with chain rule if needed

# Arguments
- `compiled`: FormulaCompiler.CompiledFormula
- `data_nt::NamedTuple`: Data in columntable format
- `row_idx::Int`: Index of observation to compute
- `β::Vector{Float64}`: Model coefficients
- `link`: GLM link function
- `scale::Symbol`: Either `:link` (link scale) or `:response` (response scale)
- `row_buf::Vector{Float64}`: Pre-allocated buffer for model row

# Returns
`PredictionWithGradient` containing:
- `value`: Predicted value at specified scale
- `gradient`: Gradient vector for delta-method SEs
- `scale`: Target scale used

# Performance
- Zero allocation when `scale=:link` (link scale)
- Minimal allocation for `scale=:response` (response scale gradient copy)
"""
function compute_prediction_with_gradient(
    compiled, data_nt, row_idx::Int, β::Vector{Float64}, 
    link, scale::Symbol, row_buf::Vector{Float64}
)
    # Core computation: build model row and compute linear predictor
    FormulaCompiler.modelrow!(row_buf, compiled, data_nt, row_idx)
    η = dot(row_buf, β)
    
    if scale === :response
        # Response scale: apply inverse link function and chain rule
        μ = GLM.linkinv(link, η)
        dμ_dη = GLM.mueta(link, η)
        
        # Chain rule: ∂μ/∂β = (∂μ/∂η) * (∂η/∂β) = dμ_dη * row_buf
        gradient = dμ_dη .* row_buf
        
        return PredictionWithGradient(μ, gradient, :response)
    else
        # Link scale: gradient is just the model row
        # Use copy to ensure gradient can be safely modified
        return PredictionWithGradient(η, copy(row_buf), :link)
    end
end

"""
    compute_predictions_batch!(results, gradients, compiled, data_nt, β, link, scale, row_buf)

Batch computation for multiple observations with optimized memory usage.

This function is optimized for population margins where we need predictions
and gradients for many observations. It writes results directly into 
pre-allocated arrays to minimize allocation overhead.

# Arguments
- `results::Vector{T}`: Pre-allocated vector to store predictions (modified in-place)
- `gradients::Matrix{Float64}`: Pre-allocated matrix to store gradients (modified in-place)
- `compiled`: FormulaCompiler.CompiledFormula  
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
            FormulaCompiler.modelrow!(row_buf, compiled, data_nt, i)
            η = dot(row_buf, β)
            
            results[i] = GLM.linkinv(link, η)
            dμ_dη = GLM.mueta(link, η)
            
            # Store gradient: ∂μ/∂β = dμ_dη * ∂η/∂β
            gradients[i, :] .= dμ_dη .* row_buf
        end
    else
        # Link scale (simpler case)
        for i in 1:n_obs
            FormulaCompiler.modelrow!(row_buf, compiled, data_nt, i)
            η = dot(row_buf, β)
            
            results[i] = η
            gradients[i, :] .= row_buf
        end
    end
end

"""
    compute_single_prediction(compiled, data_nt, row_idx, β, link, scale, row_buf) -> Float64

Compute prediction for a single observation without gradient (lighter weight).

Use this when you only need the prediction value and not the gradient
(e.g., for intermediate computations where SEs are not needed).

# Arguments
Same as `compute_prediction_with_gradient` but returns only the prediction value.

# Returns
`Float64`: Predicted value at specified scale

# Performance
- Zero allocation regardless of scale
- Fastest option when gradient not needed
"""
function compute_single_prediction(
    compiled, data_nt, row_idx::Int, β::Vector{Float64},
    link, scale::Symbol, row_buf::Vector{Float64}
)::Float64
    
    FormulaCompiler.modelrow!(row_buf, compiled, data_nt, row_idx)
    η = dot(row_buf, β)
    
    if scale === :response
        return GLM.linkinv(link, η)
    else
        return η
    end
end

"""
    compute_predictions_only!(results, compiled, data_nt, β, link, scale, row_buf)

Batch prediction computation without gradients (lighter weight).

Like `compute_predictions_batch!` but only computes predictions,
not gradients. Use when standard errors are not needed.

# Arguments
- `results::Vector{T}`: Pre-allocated vector to store predictions
- Other arguments same as `compute_predictions_batch!`

# Performance
- Fastest batch computation option
- Zero gradient allocation overhead
- Ideal for counterfactual analysis without SEs
"""
function compute_predictions_only!(
    results::AbstractVector{T}, compiled, data_nt, β::Vector{Float64}, 
    link, scale::Symbol, row_buf::Vector{Float64}
) where T<:Real
    
    n_obs = length(results)
    
    if scale === :response
        for i in 1:n_obs
            FormulaCompiler.modelrow!(row_buf, compiled, data_nt, i)
            η = dot(row_buf, β)
            results[i] = GLM.linkinv(link, η)
        end
    else
        for i in 1:n_obs
            FormulaCompiler.modelrow!(row_buf, compiled, data_nt, i)
            results[i] = dot(row_buf, β)
        end
    end
end

# End of predictions.jl