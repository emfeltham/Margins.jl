# =============================================================================
# Contrast Parameter Gradients (Migrated from FormulaCompiler.jl v2.0)
# =============================================================================
#
# Compute parameter gradients ∂(Δη)/∂β for categorical contrasts.
# Used in delta method standard error calculations for discrete effects.
#
# Migration date: 2025-10-09
# Original location: FormulaCompiler/src/compilation/contrast_evaluator.jl (lines 483-632)
#
# Architecture rationale:
# - contrast_modelrow!: Computational primitive (stays in FormulaCompiler)
# - contrast_gradient!: Statistical inference (migrated to Margins)
# - Parallel to continuous variables: marginal_effects_eta! computes Gβ in Margins
#
# =============================================================================

# Import required FormulaCompiler utilities
using FormulaCompiler:
    ContrastEvaluator,
    reset_all_counterfactuals!,
    update_counterfactual_for_var!,
    update_counterfactual_row!,
    _dmu_deta

using LinearAlgebra: dot

"""
    contrast_gradient!(∇β, evaluator, row, var, from, to, β, [link])

Compute parameter gradient vector ∇β for a discrete contrast effect.

# Arguments
- `∇β::AbstractVector{Float64}`: Pre-allocated gradient vector (modified in-place)
- `evaluator::ContrastEvaluator`: Contrast evaluator from `contrastevaluator()`
- `row::Int`: Data row index
- `var::Symbol`: Variable name to change
- `from`: Baseline level (reference)
- `to`: Counterfactual level (treatment)
- `β::AbstractVector{<:Real}`: Model coefficients
- `link=nothing`: Optional link function for response scale effects

# Returns
- `∇β`: Modified gradient vector (∂(Δη)/∂β or ∂(Δμ)/∂β)

# Scales
- **Linear scale** (`link=nothing`): ∇β = ΔX = X₁ - X₀
- **Response scale** (`link` provided): ∇β = g'(η₁) × X₁ - g'(η₀) × X₀

# Usage
```julia
using Margins, FormulaCompiler

# Setup
evaluator = contrastevaluator(model, data, :treatment)
∇β = Vector{Float64}(undef, length(evaluator))

# Compute gradient
contrast_gradient!(∇β, evaluator, 1, :treatment, "Control", "Treatment", coef(model))

# Use with delta method
se = delta_method_se(∇β, vcov(model))
```

# Mathematical Background
For categorical contrasts, the parameter gradient captures how uncertainty in β
propagates to uncertainty in the contrast effect Δη = η₁ - η₀.

**Linear scale**: Direct difference in model matrix rows
- ∇β = X₁ - X₀ where X is the model matrix row

**Response scale**: Chain rule through link function
- ∇β = g'(η₁) × X₁ - g'(η₀) × X₀
- η₁ = X₁'β, η₀ = X₀'β (linear predictors)
- g'(η) = dμ/dη (link function derivative)

# See Also
- `contrast_gradient`: Allocating convenience version
- `delta_method_se`: Standard error calculation
- `contrast_modelrow!`: Contrast computation (FormulaCompiler)
"""
function contrast_gradient!(
    ∇β::AbstractVector{Float64},
    evaluator::ContrastEvaluator,
    row::Int,
    var::Symbol,
    from, to,
    β::AbstractVector{<:Real},
    link=nothing
)
    # Hot path: assume inputs are valid (validation at construction time)
    # Keep dimension checks as they're cheap and prevent memory corruption
    length(∇β) == length(evaluator.compiled) || throw(DimensionMismatch("Gradient buffer size mismatch"))
    length(β) == length(evaluator.compiled) || throw(DimensionMismatch("Coefficient vector size mismatch"))

    if link === nothing
        # Linear scale: ∇β = ΔX = X₁ - X₀ (contrast vector)
        _contrast_gradient_linear_scale!(∇β, evaluator, row, var, from, to)
    else
        # Response scale: ∇β = g'(η₁) × X₁ - g'(η₀) × X₀ (chain rule)
        # Let Julia's method dispatch handle unsupported links naturally
        _contrast_gradient_response_scale!(∇β, evaluator, row, var, from, to, β, link)
    end

    return ∇β
end

"""
    contrast_gradient(evaluator, row, var, from, to, β, [link]) -> Vector{Float64}

Convenience version that allocates and returns the gradient vector.

# Arguments
Same as `contrast_gradient!`, but without pre-allocated `∇β`.

# Returns
- `Vector{Float64}`: Newly allocated gradient vector

# Example
```julia
∇β = contrast_gradient(evaluator, 1, :treatment, "Control", "Treatment", coef(model))
se = delta_method_se(∇β, vcov(model))
```
"""
function contrast_gradient(
    evaluator::ContrastEvaluator,
    row::Int,
    var::Symbol,
    from, to,
    β::AbstractVector{<:Real},
    link=nothing
)
    ∇β = Vector{Float64}(undef, length(evaluator))
    contrast_gradient!(∇β, evaluator, row, var, from, to, β, link)
    return ∇β
end

# =============================================================================
# Internal Implementation Functions
# =============================================================================

"""
    _contrast_gradient_linear_scale!(∇β, evaluator, row, var, from, to)

Compute parameter gradient for linear scale discrete effects: ∇β = ΔX = X₁ - X₀.

# Mathematical Formula
∇β = X₁ - X₀

Where:
- X₁: Model matrix row at counterfactual level (var=to)
- X₀: Model matrix row at baseline level (var=from)

# Implementation Notes
- Uses CounterfactualVector system for efficient scenario evaluation
- Resets counterfactuals between evaluations to prevent state bleeding
- Zero-allocation after warmup (reuses pre-allocated buffers)
"""
function _contrast_gradient_linear_scale!(
    ∇β::AbstractVector{Float64},
    evaluator::ContrastEvaluator,
    row::Int,
    var::Symbol,
    from, to
)
    # General path: compute X₁ - X₀ using model matrix evaluation

    # Reset all counterfactuals to prevent state bleeding
    reset_all_counterfactuals!(evaluator.counterfactuals)

    # Compute X₀ (baseline model matrix row)
    update_counterfactual_for_var!(evaluator, var, row, from)
    evaluator.compiled(evaluator.xrow_from_buf, evaluator.data_counterfactual, row)

    # Reset again before computing X₁
    reset_all_counterfactuals!(evaluator.counterfactuals)

    # Compute X₁ (counterfactual model matrix row)
    update_counterfactual_for_var!(evaluator, var, row, to)
    evaluator.compiled(evaluator.xrow_to_buf, evaluator.data_counterfactual, row)

    # ΔX = X₁ - X₀
    @inbounds @fastmath for i in eachindex(∇β)
        ∇β[i] = evaluator.xrow_to_buf[i] - evaluator.xrow_from_buf[i]
    end

    # Clean up: Reset counterfactuals to inactive state
    reset_all_counterfactuals!(evaluator.counterfactuals)

    return ∇β
end

"""
    _contrast_gradient_response_scale!(∇β, evaluator, row, var, from, to, β, link)

Compute parameter gradient for response scale discrete effects using mathematically correct chain rule.

# Mathematical Formula
∇β = g'(η₁) × X₁ - g'(η₀) × X₀

Where:
- η₁ = X₁'β (linear predictor at counterfactual level)
- η₀ = X₀'β (linear predictor at baseline level)
- g'(η) = dμ/dη (link function derivative)
- X₁, X₀: Model matrix rows at counterfactual and baseline levels

# Chain Rule Derivation
For response scale effect Δμ = μ₁ - μ₀ where μᵢ = g(ηᵢ):

∂(Δμ)/∂β = ∂μ₁/∂β - ∂μ₀/∂β
         = g'(η₁) × ∂η₁/∂β - g'(η₀) × ∂η₀/∂β
         = g'(η₁) × X₁ - g'(η₀) × X₀

**No computational shortcuts are used** - mathematical correctness is paramount.

# Supported Link Functions
- Identity, Log, Logit, Probit, Cloglog, Cauchit, Inverse, Sqrt, InverseSquare
- See FormulaCompiler documentation for full list
"""
function _contrast_gradient_response_scale!(
    ∇β::AbstractVector{Float64},
    evaluator::ContrastEvaluator,
    row::Int,
    var::Symbol,
    from, to,
    β::AbstractVector{<:Real},
    link
)
    # Reset all counterfactuals to prevent state bleeding
    @inbounds for i in 1:length(evaluator.counterfactuals)
        update_counterfactual_row!(evaluator.counterfactuals[i], 0)
    end

    # Step 1: Compute X₀ and η₀ = X₀'β
    update_counterfactual_for_var!(evaluator, var, row, from)
    evaluator.compiled(evaluator.xrow_from_buf, evaluator.data_counterfactual, row)
    η₀ = dot(β, evaluator.xrow_from_buf)

    # Reset again before computing X₁
    @inbounds for i in 1:length(evaluator.counterfactuals)
        update_counterfactual_row!(evaluator.counterfactuals[i], 0)
    end

    # Step 2: Compute X₁ and η₁ = X₁'β
    update_counterfactual_for_var!(evaluator, var, row, to)
    evaluator.compiled(evaluator.xrow_to_buf, evaluator.data_counterfactual, row)
    η₁ = dot(β, evaluator.xrow_to_buf)

    # Step 3: Compute link function derivatives (exact evaluation)
    g_prime_η₀ = _dmu_deta(link, η₀)  # g'(η₀) - exact
    g_prime_η₁ = _dmu_deta(link, η₁)  # g'(η₁) - exact

    # Step 4: Apply mathematically correct chain rule formula
    # ∇β = g'(η₁) × X₁ - g'(η₀) × X₀
    @inbounds @fastmath for i in eachindex(∇β)
        ∇β[i] = g_prime_η₁ * evaluator.xrow_to_buf[i] - g_prime_η₀ * evaluator.xrow_from_buf[i]
    end

    # Clean up: Reset counterfactuals to inactive state
    @inbounds for i in 1:length(evaluator.counterfactuals)
        update_counterfactual_row!(evaluator.counterfactuals[i], 0)
    end

    return ∇β
end
