# gradients.jl - Parameter Gradient Utilities

"""
    delta_method_se(gβ, Σ)

Compute standard error using delta method: SE = sqrt(gβ' * Σ * gβ)

Arguments:
- `gβ::AbstractVector{Float64}`: Parameter gradient vector
- `Σ::AbstractMatrix{Float64}`: Parameter covariance matrix from model

Returns:
- `Float64`: Standard error

Notes:
- Zero allocations per call
- Implements Var(m) = gβ' Σ gβ where m is marginal effect
- Works with gradients computed by any backend (AD, FD, analytical)
"""
function delta_method_se(gβ::AbstractVector{Float64}, Σ::AbstractMatrix{Float64})
    # Zero-allocation computation of sqrt(gβ' * Σ * gβ)
    # Use BLAS dot product to avoid temporary arrays
    n = length(gβ)
    result = 0.0
    @inbounds for i in 1:n
        temp = 0.0
        for j in 1:n
            temp += Σ[i, j] * gβ[j]
        end
        result += gβ[i] * temp
    end

    # Debug check for negative variance (should not happen with valid covariance matrix)
    if result < 0.0
        @warn "Negative variance detected in delta method: gβ'Σgβ = $result. " *
              "This suggests numerical issues or invalid covariance matrix. " *
              "Check gradient computation and covariance matrix conditioning."
        return NaN
    end

    return sqrt(result)
end

"""
    delta_method_se(evaluator, row, var, from, to, β, vcov, [link]) -> Float64

Compute standard error for discrete effects using delta method - zero allocations.

Uses the mathematical formula: SE = √(∇β' Σ ∇β) where:
- ∇β = parameter gradient from `contrast_gradient!`
- Σ = parameter covariance matrix

# Arguments
- `evaluator::ContrastEvaluator`: Pre-configured contrast evaluator
- `row::Int`: Row index to evaluate
- `var::Symbol`: Variable to contrast
- `from`, `to`: Reference and target levels
- `β::AbstractVector{<:Real}`: Model coefficients
- `vcov::AbstractMatrix{<:Real}`: Parameter covariance matrix
- `link`: GLM link function (optional, defaults to linear scale)

# Returns
- `Float64`: Standard error for the discrete effect

# Performance
- **Zero allocations** - reuses evaluator's gradient buffer
- **Type flexibility** - accepts any Real matrix/vector types

# Example
```julia
# Standard error for treatment effect
se = delta_method_se(evaluator, 1, :treatment, "Control", "Drug", β, vcov)

# Response scale standard error
link = GLM.LogitLink()
se_mu = delta_method_se(evaluator, 1, :treatment, "Control", "Drug", β, vcov, link)
```
"""
function delta_method_se(
    evaluator::ContrastEvaluator,
    row::Int,
    var::Symbol,
    from, to,
    β::AbstractVector{<:Real},
    vcov::AbstractMatrix{<:Real},
    link=nothing
)
    # Hot path: let method dispatch handle unsupported links naturally
    # Compute parameter gradient (reuses evaluator's buffer)
    contrast_gradient!(evaluator.gradient_buffer, evaluator, row, var, from, to, β, link)

    # Delta method variance: ∇β' Σ ∇β
    variance = dot(evaluator.gradient_buffer, vcov, evaluator.gradient_buffer)

    return sqrt(max(0.0, variance))  # Ensure non-negative due to numerical precision
end
