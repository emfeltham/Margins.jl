# kernels/categorical.jl
#
# The categorical-contrast kernel shared by the population, scenarios/groups and profile
# paths: the (weighted) average over rows of a discrete change ΔŷX = ŷ(X₁) − ŷ(X₀), where
# X₀ / X₁ are the model row with the variable set to the "from" / "to" level, on a
# `PredScale` (link: Δη = (X₁ − X₀)'β; response: g⁻¹(η₁) − g⁻¹(η₀)), with its delta-method
# gradient and standard error.

using FormulaCompiler: contrast_modelrow!, level_modelrow!, contrast_endpoints
using LinearAlgebra: dot

# ŷ(X₀) and dŷ/dη at X₀, which every contrast from the same level shares within a row
# (the link scale needs neither: its contrast is linear in X₁ − X₀, and the offset cancels).
# On the response scale η includes the row's offset (`offset[row]`).
@inline _contrast_base(::LinkScale, x₀, β, offset, row) = (0.0, 0.0)
@inline _contrast_base(s::ResponseScale, x₀, β, offset, row) = yhat(s, linear_predictor(x₀, β, offset, row))

"""
    _contrast_row!(acc, w, Δx, x₀, x₁, base, β, s::PredScale, offset, row) -> Δŷ

One row's contrast on scale `s`, adding `w × ∇β` of it to `acc`. `Δx = x₁ − x₀` and
`base = _contrast_base(s, x₀, β, offset, row)`.

- Link scale: Δŷ = Δx'β, ∇β = Δx
- Response scale: Δŷ = g⁻¹(η₁) − g⁻¹(η₀), ∇β = g′(η₁) x₁ − g′(η₀) x₀
"""
@inline function _contrast_row!(acc::AbstractVector{Float64}, w::Float64, Δx::Vector{Float64},
                                x₀::Vector{Float64}, x₁::Vector{Float64}, base,
                                β::Vector{Float64}, ::LinkScale, offset, row::Int)
    @inbounds @simd for i in eachindex(acc)
        acc[i] += w * Δx[i]
    end
    return dot(Δx, β)
end

@inline function _contrast_row!(acc::AbstractVector{Float64}, w::Float64, Δx::Vector{Float64},
                                x₀::Vector{Float64}, x₁::Vector{Float64}, base,
                                β::Vector{Float64}, s::ResponseScale, offset, row::Int)
    μ₀, g₀ = base
    μ₁, g₁ = yhat(s, linear_predictor(x₁, β, offset, row))
    @inbounds @simd for i in eachindex(acc)
        acc[i] += w * (g₁ * x₁[i] - g₀ * x₀[i])
    end
    return μ₁ - μ₀
end

"""
    categorical_contrast_ame!(contrast_buf, gradient_accum, evaluator, var, from_level,
                              to_level, β, vcov, s::PredScale, rows, weights=nothing,
                              offset=nothing) -> (ame, se)

(Weighted) average over `rows` of the contrast `to_level − from_level` of `var` on scale
`s`, with its delta-method SE. The parameter gradient is left in `gradient_accum`.
`contrast_buf` (length n_params) is scratch. Rows with weight ≤ 0 are skipped; when weights
are given and every row's weight is zero, a `MarginsError` is thrown. `offset` (indexed by
row, see [`Offset`](@ref)) enters η on the response scale.

Allocation-free. Used for a single contrast (profile margins); the population and context
paths use [`categorical_contrast_ame_batch!`](@ref).

    AME = Σᵢ wᵢ Δŷᵢ / Σᵢ wᵢ,  ∇β AME = Σᵢ wᵢ ∇β Δŷᵢ / Σᵢ wᵢ,  SE = √(∇'Σ∇)
"""
function categorical_contrast_ame!(
    contrast_buf::Vector{Float64},
    gradient_accum::Vector{Float64},
    evaluator::ContrastEvaluator,
    var::Symbol,
    from_level,
    to_level,
    β::Vector{Float64},
    vcov::Matrix{Float64},
    s::PredScale,
    rows::AbstractVector{Int},
    weights::Union{Nothing, Vector{Float64}} = nothing,
    offset::Offset = nothing
)
    fill!(gradient_accum, 0.0)
    x₀, x₁ = contrast_endpoints(evaluator)
    ame_sum = 0.0
    total_weight = 0.0
    for row in rows
        w = rowweight(weights, row)
        w > 0 || continue
        # Δx = X₁ − X₀, leaving X₀ / X₁ in the evaluator's endpoint buffers (x₀, x₁)
        contrast_modelrow!(contrast_buf, evaluator, row, var, from_level, to_level)
        ame_sum += w * _contrast_row!(gradient_accum, w, contrast_buf, x₀, x₁,
                                      _contrast_base(s, x₀, β, offset, row), β, s, offset, row)
        total_weight += w
    end
    if !isnothing(weights) && total_weight <= 0
        _throw_zero_weights("contrast $(to_level) - $(from_level)")
    end
    ame = ame_sum / total_weight
    gradient_accum .= gradient_accum ./ total_weight
    se = sqrt(max(0.0, dot(gradient_accum, vcov, gradient_accum)))
    return (ame, se)
end

"""
    categorical_contrast_ame_batch!(results_ame, results_se, gradient_matrix, contrast_buf,
                                    evaluator, var, contrast_pairs, β, vcov, s::PredScale,
                                    rows, weights=nothing, offset=nothing) -> nothing

All contrasts `contrast_pairs[k]` (`level2 − level1`) of `var` in one pass over `rows`:
`results_ame[k]`, `results_se[k]` and row `k` of `gradient_matrix` (n_pairs × n_params)
receive what [`categorical_contrast_ame!`](@ref) would compute for that pair, with the same
arithmetic. Within a row, X₀ (and ŷ, g′ there) is evaluated once for each run of consecutive
pairs sharing their `level1` — once per row for baseline contrasts — instead of once per
pair.

Allocation-free (`contrast_buf` is scratch of length n_params).
"""
function categorical_contrast_ame_batch!(
    results_ame::AbstractVector{Float64},
    results_se::AbstractVector{Float64},
    gradient_matrix::AbstractMatrix{Float64},
    contrast_buf::Vector{Float64},
    evaluator::ContrastEvaluator,
    var::Symbol,
    contrast_pairs::Vector{ContrastPair{T}},
    β::Vector{Float64},
    vcov::Matrix{Float64},
    s::PredScale,
    rows::AbstractVector{Int},
    weights::Union{Nothing, Vector{Float64}} = nothing,
    offset::Offset = nothing
) where T
    n_pairs = length(contrast_pairs)
    fill!(results_ame, 0.0)
    fill!(gradient_matrix, 0.0)
    x₀, x₁ = contrast_endpoints(evaluator)   # the evaluator's own buffers, used as scratch
    total_weight = 0.0
    for row in rows
        w = rowweight(weights, row)
        w > 0 || continue
        base = (0.0, 0.0)   # set with X₀ at k = 1
        for k in 1:n_pairs
            pair = contrast_pairs[k]
            if k == 1 || pair.level1 != contrast_pairs[k - 1].level1
                level_modelrow!(x₀, evaluator, row, var, pair.level1)
                base = _contrast_base(s, x₀, β, offset, row)
            end
            level_modelrow!(x₁, evaluator, row, var, pair.level2)
            contrast_buf .= x₁ .- x₀
            results_ame[k] += w * _contrast_row!(view(gradient_matrix, k, :), w, contrast_buf,
                                                 x₀, x₁, base, β, s, offset, row)
        end
        total_weight += w
    end
    if !isnothing(weights) && total_weight <= 0 && n_pairs > 0
        p = first(contrast_pairs)
        _throw_zero_weights("contrast $(p.level2) - $(p.level1)")
    end
    for k in 1:n_pairs
        results_ame[k] /= total_weight
        g = view(gradient_matrix, k, :)
        g .= g ./ total_weight
        results_se[k] = sqrt(max(0.0, dot(g, vcov, g)))
    end
    return nothing
end
