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
    AveragedResponseScale{L<:GLM.Link} <: PredScale

The response scale averaged over a normal random effect u ~ N(0, τ²) added to η
(`random_effects = :marginal`): ŷ = E[g⁻¹(η + u)], by Gauss–Hermite quadrature, with
(`z`, `w`) the rule for N(0, 1) and `τ` the SD of u. With random slopes τ varies by row and
comes with η (a `SpreadEta`, from a `RowSpreadOffset`); `τ` is then unused. Differentiating under the
integral, dŷ/dη = E[g⁻¹′(η + u)] and d²ŷ/dη² = E[g⁻¹″(η + u)], so every kernel that uses
`yhat`/`yhat2` gets population-averaged quantities and their β-gradients unchanged.
"""
struct AveragedResponseScale{L<:GLM.Link} <: PredScale
    link::L
    z::Vector{Float64}
    w::Vector{Float64}
    τ::Float64
end

# Either response scale (the kernels that need ŷ = g⁻¹(η) on the response scale dispatch on this)
const AnyResponseScale = Union{ResponseScale, AveragedResponseScale}

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

"""
    predscale(scale::Symbol, link, average) -> PredScale

As `predscale(scale, link)`; with `average` a `REAveraging` (`random_effects = :marginal`),
the response scale is the `AveragedResponseScale` of its quadrature rule. The link scale is
unaffected (E[η + u] = η).
"""
@inline predscale(scale::Symbol, link, ::Nothing) = predscale(scale, link)
@inline function predscale(scale::Symbol, link, a::REAveraging)
    scale === :response && return AveragedResponseScale(link, a.z, a.w, a.τ)
    return predscale(scale, link)
end

# The `scale` symbol a PredScale came from (for error messages)
scale_symbol(::LinkScale) = :link
scale_symbol(::AnyResponseScale) = :response

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

@inline yhat(s::AveragedResponseScale, η::Real) = _averaged_yhat(s, η, s.τ)
@inline yhat2(s::AveragedResponseScale, η::Real) = _averaged_yhat2(s, η, s.τ)
@inline yhat(s::AveragedResponseScale, e::SpreadEta) = _averaged_yhat(s, e.η, e.τ)
@inline yhat2(s::AveragedResponseScale, e::SpreadEta) = _averaged_yhat2(s, e.η, e.τ)
"""
    yhat2z(s, η) -> (ŷ, d₁, d₂, d₁z, d₂z)

`yhat2` plus d₁z = E[g⁻¹′(η + τZ)·Z] and d₂z = E[g⁻¹″(η + τZ)·Z], the derivatives of ŷ with
respect to τ (∂ŷ/∂τ = d₁z, and its η-derivative d₂z): nonzero only on the averaged response scale.
"""
@inline yhat2z(s::PredScale, η) = ((ŷ, d₁, d₂) = yhat2(s, η); (ŷ, d₁, d₂, 0.0, 0.0))
@inline yhat2z(s::AveragedResponseScale, η::Real) = _averaged_yhat2z(s, Float64(η), s.τ)
@inline yhat2z(s::AveragedResponseScale, e::SpreadEta) = _averaged_yhat2z(s, e.η, e.τ)
@inline function _averaged_yhat2z(s::AveragedResponseScale, η::Float64, τ::Float64)
    μ = 0.0; d₁ = 0.0; d₂ = 0.0; d₁z = 0.0; d₂z = 0.0
    @inbounds for k in eachindex(s.w)
        zk = s.z[k]
        m, a, b = link_derivatives(s.link, η + τ * zk)
        wk = s.w[k]
        μ += wk * m; d₁ += wk * a; d₂ += wk * b
        d₁z += wk * a * zk; d₂z += wk * b * zk
    end
    return (μ, d₁, d₂, d₁z, d₂z)
end

# On the link scale the spread does not matter (E[η + u] = η)
@inline yhat(s::LinkScale, e::SpreadEta) = yhat(s, e.η)
@inline yhat2(s::LinkScale, e::SpreadEta) = yhat2(s, e.η)

@inline function _averaged_yhat(s::AveragedResponseScale, η::Float64, τ::Float64)
    μ = 0.0; d = 0.0
    @inbounds for k in eachindex(s.w)
        ηk = η + τ * s.z[k]
        μ += s.w[k] * GLM.linkinv(s.link, ηk)
        d += s.w[k] * _dmu_deta(s.link, ηk)
    end
    return (μ, d)
end
@inline function _averaged_yhat2(s::AveragedResponseScale, η::Float64, τ::Float64)
    μ = 0.0; d₁ = 0.0; d₂ = 0.0
    @inbounds for k in eachindex(s.w)
        m, a, b = link_derivatives(s.link, η + τ * s.z[k])
        μ += s.w[k] * m; d₁ += s.w[k] * a; d₂ += s.w[k] * b
    end
    return (μ, d₁, d₂)
end

"""
    rowweight(weights, row) -> Float64

Weight of `row`: `1.0` when unweighted (`weights === nothing`), else `weights[row]`.
"""
@inline rowweight(::Nothing, row::Int) = 1.0
@inline rowweight(weights::AbstractVector, row::Int) = Float64(weights[row])

"""
    Offset

The η offset of a computation: `nothing` (none) or a `Vector{Float64}` indexed like the rows
evaluated (data rows for population margins, reference-grid rows for profile margins). Kernels
take it as an argument, so each is compiled for one of the two and the row loop has no branch.
"""
const Offset = AnyOffset

"""
    linear_predictor(x, β, offset, row) -> η

η = x'β, plus `offset[row]` when there is an offset. Without one this is exactly `dot(x, β)`
(no `+ 0.0`), so results without an offset are bitwise those of a kernel without offsets.
"""
@inline linear_predictor(x::AbstractVector{Float64}, β::AbstractVector{Float64}, ::Nothing, row::Int) = dot(x, β)
@inline linear_predictor(x::AbstractVector{Float64}, β::AbstractVector{Float64}, offset::Vector{Float64}, row::Int) =
    dot(x, β) + offset[row]
# `random_effects = :marginal` with random slopes: η with the row's SD of u = z'b
@inline linear_predictor(x::AbstractVector{Float64}, β::AbstractVector{Float64}, offset::RowSpreadOffset, row::Int) =
    SpreadEta(linear_predictor(x, β, offset.shift, row), _row_tau(offset, x))
# The effect of a random-slope variable: η is the base offset's (the slopes enter the accumulator)
@inline linear_predictor(x::AbstractVector{Float64}, β::AbstractVector{Float64}, offset::SlopedOffset, row::Int) =
    linear_predictor(x, β, offset.base, row)

"""
    predict_row!(x, compiled, data, β, s::PredScale, row, offset=nothing) -> (ŷ, dŷ/dη)

Evaluate the model row xᵢ into `x` and return the prediction on scale `s` and its
derivative with respect to η = xᵢ'β (+ `offset[row]`). The gradient of ŷ with respect to β is
`(dŷ/dη) .* x`.
"""
@inline function predict_row!(x::AbstractVector{Float64}, compiled, data::NamedTuple,
                              β::AbstractVector{Float64}, s::PredScale, row::Int,
                              offset::Offset=nothing)
    compiled(x, data, row)
    return yhat(s, linear_predictor(x, β, offset, row))
end

"""
    average_prediction!(g, x, compiled, data, β, s::PredScale, rows, weights, total_weight,
                        offset=nothing) -> Float64

(Weighted) average prediction over `rows`, Σ wᵢ ŷᵢ / `total_weight`, with its gradient with
respect to β written into `g` (Σ wᵢ (dŷ/dη)ᵢ xᵢ / `total_weight`). Rows whose weight is not
positive are skipped; `weights === nothing` gives every row weight 1. `x` is a model-row
buffer (length of the compiled formula). `offset` is the η offset indexed by row (see
[`Offset`](@ref)). Allocation-free.

Used by the population AAP, with and without scenarios/groups contexts.
"""
function average_prediction!(g::AbstractVector{Float64}, x::Vector{Float64}, compiled,
                             data::NamedTuple, β::Vector{Float64}, s::PredScale, rows,
                             weights::Union{Nothing, AbstractVector{Float64}},
                             total_weight::Float64, offset::Offset=nothing)
    fill!(g, 0.0)
    acc = 0.0
    @inbounds for row in rows
        w = rowweight(weights, row)
        w > 0 || continue
        ŷ, d = predict_row!(x, compiled, data, β, s, row, offset)
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
    predict_rows!(ŷ, G, x, compiled, data, β, s::PredScale, rows, offset=nothing)

Prediction at each of `rows` into `ŷ[k]`, with its gradient with respect to β into row `k` of
`G` (`k` indexing `rows`; `offset` indexed by row). Used for profile predictions (APM/APR), one
per grid row.
"""
function predict_rows!(ŷ::AbstractVector{Float64}, G::AbstractMatrix{Float64},
                       x::Vector{Float64}, compiled, data::NamedTuple, β::Vector{Float64},
                       s::PredScale, rows, offset::Offset=nothing)
    for (k, row) in enumerate(rows)
        yk, d = predict_row!(x, compiled, data, β, s, row, offset)
        ŷ[k] = yk
        @inbounds for j in eachindex(x)
            G[k, j] = d * x[j]
        end
    end
    return nothing
end
