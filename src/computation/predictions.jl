# computation/predictions.jl
#
# The prediction kernel: every place Margins computes a prediction ŷ (on the link or the
# response scale), its derivatives with respect to the linear predictor η, or its gradient
# with respect to β goes through the functions below.
#
# The scale is a type (`PredScale`), resolved once from the user's `scale` symbol with
# `predscale` before a function barrier, so hot loops specialize on it instead of branching on
# a Symbol per row. On the link scale ŷ = η (dŷ/dη = 1, d²ŷ/dη² = 0); on the response scale
# ŷ = g⁻¹(η) with derivatives from FormulaCompiler's `link_derivatives`.

"""
    PredScale

Scale on which a prediction ŷ is reported: `LinkScale()` (ŷ = η = x'β) or
`ResponseScale(link)` (ŷ = g⁻¹(η)). Built from a `scale` symbol by [`predscale`](@ref).
"""
abstract type PredScale end

"""
    LinkScale <: PredScale

The linear-predictor scale: ŷ = η, dŷ/dη = 1, d²ŷ/dη² = 0.
"""
struct LinkScale <: PredScale end

"""
    ResponseScale{L<:GLM.Link} <: PredScale

The response scale of a model with link `L`: ŷ = g⁻¹(η).
"""
struct ResponseScale{L<:GLM.Link} <: PredScale
    link::L
end

"""
    predscale(scale::Symbol, link) -> PredScale

`ResponseScale(link)` for `:response`, `LinkScale()` for `:link`; any other symbol is an
`ArgumentError`. Call it once, outside the hot loop.
"""
@inline function predscale(scale::Symbol, link)
    scale === :response && return ResponseScale(link)
    scale === :link && return LinkScale()
    throw(ArgumentError("scale must be :response or :link, got $(repr(scale))"))
end

# The `scale` symbol a PredScale came from (for error messages)
scale_symbol(::LinkScale) = :link
scale_symbol(::ResponseScale) = :response

"""
    yhat(s::PredScale, η) -> (ŷ, dŷ/dη)

Prediction and its first derivative with respect to the linear predictor.
"""
@inline yhat(::LinkScale, η::Real) = (η, one(η))
@inline yhat(s::ResponseScale, η::Real) = (GLM.linkinv(s.link, η), _dmu_deta(s.link, η))

"""
    yhat2(s::PredScale, η) -> (ŷ, dŷ/dη, d²ŷ/dη²)

Prediction and its first two derivatives with respect to the linear predictor.
"""
@inline yhat2(::LinkScale, η::Real) = (η, one(η), zero(η))
@inline yhat2(s::ResponseScale, η::Real) = link_derivatives(s.link, η)

"""
    rowweight(weights, row) -> Float64

Weight of `row`: `1.0` when unweighted (`weights === nothing`), else `weights[row]`.
"""
@inline rowweight(::Nothing, row::Int) = 1.0
@inline rowweight(weights::AbstractVector, row::Int) = Float64(weights[row])

"""
    predict_row!(x, compiled, data, β, s::PredScale, row) -> (ŷ, dŷ/dη)

Evaluate the model row xᵢ into `x` and return the prediction on scale `s` and its
derivative with respect to η. The gradient of ŷ with respect to β is `(dŷ/dη) .* x`.
"""
@inline function predict_row!(x::AbstractVector{Float64}, compiled, data::NamedTuple,
                              β::AbstractVector{Float64}, s::PredScale, row::Int)
    compiled(x, data, row)
    return yhat(s, dot(x, β))
end

"""
    average_prediction!(g, x, compiled, data, β, s::PredScale, rows, weights, total_weight) -> Float64

(Weighted) average prediction over `rows`, Σ wᵢ ŷᵢ / `total_weight`, with its gradient with
respect to β written into `g` (Σ wᵢ (dŷ/dη)ᵢ xᵢ / `total_weight`). Rows whose weight is not
positive are skipped; `weights === nothing` gives every row weight 1. `x` is a model-row
buffer (length of the compiled formula). Allocation-free.

Used by the population AAP, with and without scenarios/groups contexts.
"""
function average_prediction!(g::AbstractVector{Float64}, x::Vector{Float64}, compiled,
                             data::NamedTuple, β::Vector{Float64}, s::PredScale, rows,
                             weights::Union{Nothing, AbstractVector{Float64}},
                             total_weight::Float64)
    fill!(g, 0.0)
    acc = 0.0
    @inbounds for row in rows
        w = rowweight(weights, row)
        w > 0 || continue
        ŷ, d = predict_row!(x, compiled, data, β, s, row)
        acc += w * ŷ
        wd = w * d
        @simd for j in eachindex(g)
            g[j] += wd * x[j]
        end
    end
    @inbounds @simd for j in eachindex(g)
        g[j] /= total_weight
    end
    return acc / total_weight
end

"""
    predict_rows!(ŷ, G, x, compiled, data, β, s::PredScale, rows)

Prediction at each of `rows` into `ŷ[k]`, with its gradient with respect to β into row `k` of
`G` (`k` indexing `rows`). Used for profile predictions (APM/APR), one per grid row.
"""
function predict_rows!(ŷ::AbstractVector{Float64}, G::AbstractMatrix{Float64},
                       x::Vector{Float64}, compiled, data::NamedTuple, β::Vector{Float64},
                       s::PredScale, rows)
    for (k, row) in enumerate(rows)
        yk, d = predict_row!(x, compiled, data, β, s, row)
        ŷ[k] = yk
        @inbounds for j in eachindex(x)
            G[k, j] = d * x[j]
        end
    end
    return nothing
end
