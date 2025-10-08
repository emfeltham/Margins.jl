# kernels/categorical.jl
# Zero-allocation categorical effects kernel using FormulaCompiler primitives
#
# This module implements the categorical kernel layer (MARGINS_PLAN.md Section 4.2)
# using FormulaCompiler's ContrastEvaluator for zero-allocation contrast computation.

using FormulaCompiler: contrast_modelrow!, contrast_gradient!
using LinearAlgebra: dot
using GLM: linkinv

"""
    categorical_contrast_ame!(
        contrast_buf::Vector{Float64},
        gradient_buf::Vector{Float64},
        gradient_accum::Vector{Float64},
        evaluator::ContrastEvaluator,
        var::Symbol,
        from_level,
        to_level,
        β::Vector{Float64},
        vcov::Matrix{Float64},
        link,
        rows::AbstractVector{Int},
        weights::Union{Nothing, Vector{Float64}} = nothing
    ) -> (ame::Float64, se::Float64)

Compute population average marginal effect for a categorical contrast using pre-allocated buffers.

**In-place operation**: Uses pre-allocated buffers from MarginsEngine for zero-allocation computation.
The gradient is stored in-place in `gradient_accum` and not returned.

Uses FormulaCompiler's zero-allocation primitives:
- `contrast_modelrow!` for per-row contrasts
- `contrast_gradient!` for parameter gradients
- `delta_method_se` for standard errors

# Arguments
- `contrast_buf::Vector{Float64}`: Pre-allocated buffer for contrast vectors (length n_params)
- `gradient_buf::Vector{Float64}`: Pre-allocated buffer for per-row gradients (length n_params)
- `gradient_accum::Vector{Float64}`: Pre-allocated buffer for accumulated gradients (length n_params)
- `evaluator::ContrastEvaluator`: Pre-built contrast evaluator from engine
- `var::Symbol`: Categorical variable name
- `from_level`: Baseline level (typically reference category)
- `to_level`: Treatment level (level to contrast against baseline)
- `β::Vector{Float64}`: Model coefficients
- `vcov::Matrix{Float64}`: Parameter covariance matrix
- `link`: GLM link function (for response-scale effects)
- `rows::AbstractVector{Int}`: Rows to average over
- `weights::Union{Nothing, Vector{Float64}}`: Optional observation weights

# Returns
- `ame::Float64`: Average marginal effect (population contrast)
- `se::Float64`: Standard error via delta method

Note: The parameter gradient is stored in-place in `gradient_accum` (not returned)

# Performance
- **0 bytes allocated** for unweighted computation (pre-allocated buffers)
- **O(n) time complexity** where n = length(rows)
- **Type-stable** through concrete ContrastEvaluator

# Mathematical Details
**Linear scale (link=nothing)**:
AME = (1/n) Σᵢ [η(xᵢ, var=to) - η(xᵢ, var=from)] = (1/n) Σᵢ [(X₁'β) - (X₀'β)]

**Response scale (with link function)**:
AME = (1/n) Σᵢ [μ(xᵢ, var=to) - μ(xᵢ, var=from)] = (1/n) Σᵢ [g⁻¹(η₁) - g⁻¹(η₀)]

Where:
- η = Xβ is the linear predictor
- μ = g⁻¹(η) is the response via link inverse function
- X₀ and X₁ are model matrix rows at baseline and treatment levels
- g is the link function (e.g., logit for binomial models)

Gradient: ∂AME/∂β = (1/n) Σᵢ [∂μ(xᵢ, var=to)/∂β - ∂μ(xᵢ, var=from)/∂β]

Standard error: SE = √(∂AME/∂β' × Σ × ∂AME/∂β)
"""
function categorical_contrast_ame!(
    contrast_buf::Vector{Float64},
    gradient_buf::Vector{Float64},
    gradient_accum::Vector{Float64},
    evaluator::ContrastEvaluator,
    var::Symbol,
    from_level,
    to_level,
    β::Vector{Float64},
    vcov::Matrix{Float64},
    link,
    rows::AbstractVector{Int},
    weights::Union{Nothing, Vector{Float64}} = nothing
)
    # Use pre-allocated buffers from engine (zero allocations)
    # Initialize gradient accumulator
    fill!(gradient_accum, 0.0)

    # Accumulators for AME
    ame_sum = 0.0
    total_weight = 0.0

    if isnothing(weights)
        # Unweighted computation (most common case)
        for row in rows
            # Compute contrast on model matrix scale (0 bytes)
            # This populates evaluator.y_from_buf (X₀) and evaluator.y_to_buf (X₁)
            contrast_modelrow!(contrast_buf, evaluator, row, var, from_level, to_level)

            # Compute effect on appropriate scale
            if link === nothing
                # Linear scale: Δη = ΔX'β = (X₁ - X₀)'β
                contrast_effect = dot(contrast_buf, β)
            else
                # Response scale: Δμ = g⁻¹(η₁) - g⁻¹(η₀)
                # After contrast_modelrow!, evaluator has X₀ in y_from_buf and X₁ in y_to_buf
                η₀ = dot(evaluator.y_from_buf, β)  # Linear predictor at baseline
                η₁ = dot(evaluator.y_to_buf, β)    # Linear predictor at treatment
                μ₀ = linkinv(link, η₀)              # Transform to response scale
                μ₁ = linkinv(link, η₁)              # Transform to response scale
                contrast_effect = μ₁ - μ₀           # Contrast on response scale
            end
            ame_sum += contrast_effect

            # Gradient (0 bytes)
            # For identity link: ∂(Δη)/∂β = Δ
            # For non-identity links: handled by contrast_gradient!
            contrast_gradient!(gradient_buf, evaluator, row, var, from_level, to_level, β, link)
            gradient_accum .+= gradient_buf
        end

        # Average over observations
        n = length(rows)
        ame = ame_sum / n
        gradient_accum .= gradient_accum ./ n

    else
        # Weighted computation
        for row in rows
            w = weights[row]
            if w > 0
                # Compute contrast on model matrix scale (0 bytes)
                # This populates evaluator.y_from_buf (X₀) and evaluator.y_to_buf (X₁)
                contrast_modelrow!(contrast_buf, evaluator, row, var, from_level, to_level)

                # Compute effect on appropriate scale
                if link === nothing
                    # Linear scale: Δη = ΔX'β = (X₁ - X₀)'β
                    contrast_effect = dot(contrast_buf, β)
                else
                    # Response scale: Δμ = g⁻¹(η₁) - g⁻¹(η₀)
                    η₀ = dot(evaluator.y_from_buf, β)  # Linear predictor at baseline
                    η₁ = dot(evaluator.y_to_buf, β)    # Linear predictor at treatment
                    μ₀ = linkinv(link, η₀)              # Transform to response scale
                    μ₁ = linkinv(link, η₁)              # Transform to response scale
                    contrast_effect = μ₁ - μ₀           # Contrast on response scale
                end
                ame_sum += w * contrast_effect

                # Weighted gradient (0 bytes)
                contrast_gradient!(gradient_buf, evaluator, row, var, from_level, to_level, β, link)
                gradient_accum .+= w .* gradient_buf

                total_weight += w
            end
        end

        if total_weight <= 0
            error("All weights are zero for contrast $(to_level) - $(from_level); cannot compute weighted effect.")
        end

        # Weighted average
        ame = ame_sum / total_weight
        gradient_accum .= gradient_accum ./ total_weight
    end

    # Compute standard error via delta method: SE = √(g'Σg)
    se = sqrt(max(0.0, dot(gradient_accum, vcov, gradient_accum)))

    # gradient_accum already contains the accumulated gradient (in-place)
    return (ame, se)
end

"""
    categorical_contrast_ame_batch!(
        results_ame::Vector{Float64},
        results_se::Vector{Float64},
        gradient_matrix::Matrix{Float64},
        contrast_buf::Vector{Float64},
        gradient_buf::Vector{Float64},
        gradient_accum::Vector{Float64},
        evaluator::ContrastEvaluator,
        var::Symbol,
        contrast_pairs::Vector{ContrastPair{T}},
        β::Vector{Float64},
        vcov::Matrix{Float64},
        link,
        rows::AbstractVector{Int},
        weights::Union{Nothing, Vector{Float64}} = nothing
    ) where T -> Nothing

Compute multiple categorical contrasts efficiently in a single batch with zero allocations.

**In-place operation**: Uses pre-allocated result arrays and buffers from MarginsEngine.
All results are stored in-place in the provided arrays; nothing is returned.

Processes all contrast pairs for a single variable with shared infrastructure:
- Single ContrastEvaluator reused across all pairs
- Pre-allocated buffers from engine (zero allocations)
- Efficient row traversal

# Arguments
- `results_ame::Vector{Float64}`: Pre-allocated array for AME results [length(contrast_pairs)]
- `results_se::Vector{Float64}`: Pre-allocated array for standard errors [length(contrast_pairs)]
- `gradient_matrix::Matrix{Float64}`: Pre-allocated matrix for gradients [length(contrast_pairs) × n_params]
- `contrast_buf::Vector{Float64}`: Pre-allocated contrast buffer from engine [n_params]
- `gradient_buf::Vector{Float64}`: Pre-allocated gradient buffer from engine [n_params]
- `gradient_accum::Vector{Float64}`: Pre-allocated accumulator from engine [n_params]
- `evaluator::ContrastEvaluator`: Pre-built contrast evaluator
- `var::Symbol`: Categorical variable name
- `contrast_pairs::Vector{ContrastPair{T}}`: Pairs of (from, to) levels to contrast
- `β::Vector{Float64}`: Model coefficients
- `vcov::Matrix{Float64}`: Parameter covariance matrix
- `link`: GLM link function
- `rows::AbstractVector{Int}`: Rows to average over
- `weights::Union{Nothing, Vector{Float64}}`: Optional weights

# Returns
- `Nothing`: All results stored in-place in pre-allocated arrays

# Performance
- **0 bytes allocated** (all buffers pre-allocated)
- **Single evaluator** reused for all pairs
- **Type-stable** dispatch through ContrastPair{T}
- **O(n × k)** time where n=length(rows), k=length(contrast_pairs)

# Use Cases
- **Baseline contrasts**: Each level vs reference (typical for AME)
- **Pairwise contrasts**: All unique pairs (comprehensive analysis)

# Example
```julia
# Pre-allocate at outer scope
n_contrasts = length(contrast_pairs)
results_ame = Vector{Float64}(undef, n_contrasts)
results_se = Vector{Float64}(undef, n_contrasts)
gradient_matrix = Matrix{Float64}(undef, n_contrasts, length(β))

# Zero-allocation batch computation
categorical_contrast_ame_batch!(
    results_ame, results_se, gradient_matrix,
    engine.contrast_buf, engine.contrast_grad_buf, engine.contrast_grad_accum,
    engine.contrast, var, contrast_pairs,
    engine.β, engine.Σ, engine.link, rows, weights
)
```
"""
function categorical_contrast_ame_batch!(
    results_ame::AbstractVector{Float64},
    results_se::AbstractVector{Float64},
    gradient_matrix::AbstractMatrix{Float64},
    contrast_buf::Vector{Float64},
    gradient_buf::Vector{Float64},
    gradient_accum::Vector{Float64},
    evaluator::ContrastEvaluator,
    var::Symbol,
    contrast_pairs::Vector{ContrastPair{T}},
    β::Vector{Float64},
    vcov::Matrix{Float64},
    link,
    rows::AbstractVector{Int},
    weights::Union{Nothing, Vector{Float64}} = nothing
) where T

    # Process each contrast pair, storing results in-place
    for (i, pair) in enumerate(contrast_pairs)
        # Compute single contrast using zero-allocation kernel
        ame, se = categorical_contrast_ame!(
            contrast_buf, gradient_buf, gradient_accum,
            evaluator, var, pair.level1, pair.level2,
            β, vcov, link, rows, weights
        )

        # Store results in pre-allocated arrays (0 bytes)
        results_ame[i] = ame
        results_se[i] = se

        # Copy gradient to matrix using view (0 bytes)
        copyto!(view(gradient_matrix, i, :), gradient_accum)
    end

    return nothing
end
