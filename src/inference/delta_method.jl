# delta_method.jl - Delta Method Standard Error Computation
#
# ═══════════════════════════════════════════════════════════════════════════
# WHY vcov(model) IS ALWAYS ON THE RIGHT SCALE
# ═══════════════════════════════════════════════════════════════════════════
#
# A common question: "When computing standard errors for marginal effects,
# how do we know that vcov(model) (which gives Var(β)) is on the right scale?"
#
# ANSWER: The gradient ∂θ/∂β automatically handles the scale transformation
# through the chain rule. This is guaranteed by the mathematics of the delta method.
#
# Mathematical Justification
# ──────────────────────────
# For any marginal effect θ = f(β) (a function of model parameters):
#
#   Var(θ) ≈ [∂θ/∂β]' Σ [∂θ/∂β]
#
# where Σ = vcov(model) = Var(β).
#
# Dimensional Analysis (proves correctness)
# ─────────────────────────────────────────
# Let's trace the units:
#
#   [Σ] = [β]²                        (covariance of parameters)
#   [∂θ/∂β] = [θ]/[β]                 (gradient converts scales)
#   [Var(θ)] = ([θ]/[β])' × [β]² × ([θ]/[β])
#            = ([θ]/[β])² × [β]²
#            = [θ]²                    ✓ Correct!
#
# The gradient acts as a conversion factor from parameter scale to effect scale.
#
# Concrete Example: Logistic Regression
# ──────────────────────────────────────
# Model: μ = expit(Xβ) where expit(z) = 1/(1+exp(-z))
# Marginal effect: ME = ∂μ/∂x = β_x × μ(1-μ)
#
# Gradient w.r.t. all parameters β_j (computed by Margins.jl):
#   ∂ME/∂β_j = μ(1-μ) × 1[j=x] + β_x × μ(1-μ)(1-2μ) × X_j
#              ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#              direct effect      chain rule through link function
#
# Then: SE(ME) = sqrt([∂ME/∂β]' vcov(model) [∂ME/∂β])
#
# Units:
#   vcov(model) has units [log-odds]² (parameter space)
#   ∂ME/∂β has units [probability]/[log-odds] (converts spaces)
#   Result has units [probability]² ✓
#
# Why This Works for ANY Model
# ─────────────────────────────
# 1. vcov(model) ALWAYS lives in parameter space (β scale)
# 2. The gradient ∂θ/∂β ALWAYS converts from β scale to θ scale
# 3. The quadratic form automatically produces variance in θ scale
# 4. Different link functions → different gradients → correct scale transformation
#
# Implementation in Margins.jl
# ────────────────────────────
# See src/inference/marginal_effects.jl for the full chain rule implementation:
#
#   For linear predictor (η = Xβ):
#     ∂(∂η/∂x)/∂β is computed directly (simple!)
#
#   For response scale with link g (μ = g⁻¹(η)):
#     ∂(∂μ/∂x)/∂β = g'(η) × J + (J'β) × g''(η) × X
#     where J = ∂(∂η/∂x)/∂β
#
# The second derivative g''(η) is crucial - it captures how the link function's
# curvature affects uncertainty propagation.
#
# Validation
# ──────────
# This is validated in test/statistical_validation/:
# - analytical_se_validation.jl: Hand-calculated formulas match
# - bootstrap_se_validation.jl: Bootstrap SEs confirm delta method
# - backend_consistency.jl: AD and FD backends agree
#
# Bottom Line
# ───────────
# You never need a "different vcov matrix" for different scales.
# The gradient does all the transformation work automatically through calculus!
#
# ═══════════════════════════════════════════════════════════════════════════

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
