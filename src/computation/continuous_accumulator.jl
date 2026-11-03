# computation/continuous_accumulator.jl
#
# The continuous-effects accumulator: one row loop for every combination of scale
# (`PredScale`), measure (`Measure`), weights and derivative backend, shared by the
# population, scenarios/groups and profile paths.
#
# Per row i, with Jᵢ = ∂xᵢ/∂vars (the model-row Jacobian), xᵢ the model row, η = xᵢ'β (plus the
# row's offset, if any) and
# (ŷ, d₁, d₂) = (ŷ, dŷ/dη, d²ŷ/dη²) on the requested scale, for each requested variable j:
#
#   aⱼ = Jᵢ[:,j]'β            (∂η/∂xⱼ)
#   gⱼ = d₁ aⱼ                (∂ŷ/∂xⱼ, the unit marginal effect)
#   mⱼ = 1, xᵢⱼ/ŷ, xᵢⱼ or 1/ŷ (measure :effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx)
#   qⱼ = mⱼ gⱼ                (the unit quantity averaged)
#   ∇β qⱼ = (mⱼ d₁) Jᵢ[:,j] + (mⱼ d₂ aⱼ − δ qⱼ d₁/ŷ) xᵢ
#
# with δ = 1 for the measures that divide by ŷ. Both terms are accumulated column-wise
# (A[:,k] and B[:,k]) and added once at the end, so the per-row cost is two AXPYs per
# variable. On the link scale (d₂ = 0) with a measure that does not divide by ŷ, the second
# term vanishes and is skipped.

"""
    Measure

The quantity a continuous effect averages: `Effect()` (∂ŷ/∂x), `Eyex()` (`:elasticity`,
(∂ŷ/∂x)·x/ŷ), `Dyex()` (`:semielasticity_dyex`, (∂ŷ/∂x)·x) or `Eydx()`
(`:semielasticity_eydx`, (∂ŷ/∂x)/ŷ).
"""
abstract type Measure end
struct Effect <: Measure end
struct Eyex <: Measure end
struct Dyex <: Measure end
struct Eydx <: Measure end

measure_symbol(::Effect) = :effect
measure_symbol(::Eyex) = :elasticity
measure_symbol(::Dyex) = :semielasticity_dyex
measure_symbol(::Eydx) = :semielasticity_eydx

# Does the measure divide by ŷ (δ = 1)? Does it use the variable's value x?
@inline _divides_by_y(::Measure) = false
@inline _divides_by_y(::Union{Eyex, Eydx}) = true
@inline _uses_x(::Measure) = false
@inline _uses_x(::Union{Eyex, Dyex}) = true

# The measure factor mⱼ
@inline _measure_factor(::Effect, x, ŷ) = 1.0
@inline _measure_factor(::Eyex, x, ŷ) = x / ŷ
@inline _measure_factor(::Dyex, x, ŷ) = x
@inline _measure_factor(::Eydx, x, ŷ) = 1.0 / ŷ

# Whether the xᵢ term (B) can be nonzero
@inline _needs_xterm(::LinkScale, m::Measure) = _divides_by_y(m)
@inline _needs_xterm(::ResponseScale, ::Measure) = true

"""
    MEWorkspace

Scratch for [`accumulate_continuous!`](@ref): the Jacobian `J` (length(de) × length(derivative_variables(de)))
and model row `x` of one row, the column-wise gradient accumulators `A` and `B`
(n_params × n_vars) and per-row coefficients `c` (n_vars).
"""
struct MEWorkspace
    J::Matrix{Float64}
    x::Vector{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    c::Vector{Float64}
end

"""
    MEWorkspace(de::AbstractDerivativeEvaluator, n_vars::Int)

Workspace for accumulating `n_vars` of `de`'s variables.
"""
function MEWorkspace(de::AbstractDerivativeEvaluator, n_vars::Int)
    n_params = length(de)
    return MEWorkspace(Matrix{Float64}(undef, n_params, length(derivative_variables(de))),
                       Vector{Float64}(undef, n_params),
                       Matrix{Float64}(undef, n_params, n_vars),
                       Matrix{Float64}(undef, n_params, n_vars),
                       Vector{Float64}(undef, n_vars))
end

# An empty workspace, for engines without a derivative evaluator
MEWorkspace() = MEWorkspace(Matrix{Float64}(undef, 0, 0), Float64[],
                            Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), Float64[])

"""
    XColumns

Data columns of the requested variables for measures that multiply by x (`_uses_x`): one
`Vector{Float64}` per variable ([`_float_columns`](@ref)), or `()` for measures that do not
read x.
"""
const XColumns = Union{Tuple{}, Vector{Vector{Float64}}}

"""
    _float_columns(data_nt, vars) -> Vector{Vector{Float64}}
    _float_columns!(buf, data_nt, vars) -> buf

The columns `vars` of `data_nt` as `Vector{Float64}`: the column itself when it already is one
(continuous columns are, after `_prepare_margins_data`), otherwise a converted copy (e.g. an
integer column of a user-supplied reference grid). A homogeneous vector, unlike a tuple of the
columns, has the same type whatever the column types, so reading it needs no dynamic dispatch.
"""
_float_columns(data_nt::NamedTuple, vars::AbstractVector{Symbol}) =
    _float_columns!(Vector{Vector{Float64}}(undef, length(vars)), data_nt, vars)

# In place, into `buf` (resized to `length(vars)`)
function _float_columns!(buf::Vector{Vector{Float64}}, data_nt::NamedTuple, vars::AbstractVector{Symbol})
    resize!(buf, length(vars))
    for (k, v) in enumerate(vars)
        buf[k] = _float_column(getproperty(data_nt, v))
    end
    return buf
end
_float_column(col::Vector{Float64}) = col
_float_column(col::AbstractVector) = convert(Vector{Float64}, col)

"""
    accumulate_continuous!(vals, grads, ws::MEWorkspace, de, β, s::PredScale, m::Measure,
                           var_idx, xcols, rows, weights, total_weight, offset=nothing)

Add to `vals[k]` the (weighted) average over `rows` of the unit quantity of measure `m` on
scale `s` for variable `derivative_variables(de)[var_idx[k]]`, and to row `k` of `grads` its gradient with
respect to β (see the formulas at the top of this file).

Each row has weight `1/length(rows)` when `weights === nothing`, else
`weights[row] / total_weight`; rows with weight ≤ 0 are skipped. `xcols[k]` is the data
column of variable `k` (used only by measures that multiply by x; pass `()` otherwise; see
[`XColumns`](@ref)).
Measures that divide by ŷ throw a `MarginsError` at a row with ŷ = 0. `offset` (indexed by
row, see [`Offset`](@ref)) enters η, so ŷ and the link derivatives are taken at η + offset.

`ws` must be sized for `de` and at least `length(var_idx)` variables. Allocation-free per
row; the only backend-dependent call is `derivative_modelrow!(J, x, de, row)`.
"""
function accumulate_continuous!(vals::AbstractVector{Float64}, grads::AbstractMatrix{Float64},
                                ws::MEWorkspace, de::AbstractDerivativeEvaluator,
                                β::Vector{Float64}, s::PredScale, m::Measure,
                                var_idx::AbstractVector{Int}, xcols::XColumns, rows,
                                weights::Union{Nothing, AbstractVector{Float64}},
                                total_weight::Float64, offset::Offset=nothing)
    (; J, x, A, B, c) = ws
    n_params = length(β)
    n_vars = length(var_idx)
    xterm = _needs_xterm(s, m)
    @inbounds for k in 1:n_vars, i in 1:n_params
        A[i, k] = 0.0
        B[i, k] = 0.0
    end
    _accumulate_continuous_rows!(vals, J, x, A, B, c, de, β, s, m, var_idx, xcols, rows,
                                 weights, total_weight, 1.0 / length(rows), offset)
    @inbounds for k in 1:n_vars
        if xterm
            for i in 1:n_params
                grads[k, i] += A[i, k] + B[i, k]
            end
        else
            for i in 1:n_params
                grads[k, i] += A[i, k]
            end
        end
    end
    return nothing
end

function _accumulate_continuous_rows!(vals::AbstractVector{Float64}, J::Matrix{Float64},
                                      x::Vector{Float64}, A::Matrix{Float64}, B::Matrix{Float64},
                                      c::Vector{Float64}, de::AbstractDerivativeEvaluator,
                                      β::Vector{Float64}, s::PredScale, m::Measure,
                                      var_idx::AbstractVector{Int}, xcols::XColumns, rows,
                                      weights::Union{Nothing, AbstractVector{Float64}},
                                      total_weight::Float64, unweighted_weight::Float64,
                                      offset::Offset)
    n_params = length(β)
    n_vars = length(var_idx)
    xterm = _needs_xterm(s, m)
    @inbounds for row in rows
        w = if isnothing(weights)
            unweighted_weight
        else
            wr = weights[row]
            wr <= 0 && continue
            wr / total_weight
        end

        derivative_modelrow!(J, x, de, row)   # Jᵢ and xᵢ
        ŷ, d₁, d₂ = yhat2(s, linear_predictor(x, β, offset, row))
        _divides_by_y(m) && iszero(ŷ) && _throw_zero_prediction(measure_symbol(m), row, scale_symbol(s))

        for k in 1:n_vars
            j = var_idx[k]
            a = 0.0
            @simd for i in 1:n_params
                a += J[i, j] * β[i]
            end
            mk = _measure_factor(m, _uses_x(m) ? xcols[k][row] : 1.0, ŷ)
            α = w * mk * d₁                     # coefficient of Jᵢ[:,j]
            wq = α * a                          # w × unit quantity
            vals[k] += wq
            @simd for i in 1:n_params
                A[i, k] += α * J[i, j]
            end
            if xterm
                ck = (w * mk * d₂) * a
                _divides_by_y(m) && (ck -= (wq / ŷ) * d₁)
                c[k] = ck
            end
        end
        if xterm
            for k in 1:n_vars
                ck = c[k]
                @simd for i in 1:n_params
                    B[i, k] += ck * x[i]
                end
            end
        end
    end
    return nothing
end

@noinline _throw_zero_prediction(measure, row, scale) = throw(MarginsError(
    "measure=:$measure divides by the prediction ŷ, which is exactly 0 at row $row " *
    "(a data row, or a reference-grid row for profile margins) " *
    "(scale=:$scale), so the unit-level quantity is undefined there. Use " *
    "measure=:semielasticity_dyex or :effect, or exclude such observations."))

"""
    _measure_type(measure::Symbol) -> Measure

The `Measure` for a `measure` keyword value. Entry points convert once and pass the result
through the same dynamic call that passes the (runtime-built) engine or evaluator on, so the
code behind it is compiled for the requested measure only, not for all four.
"""
function _measure_type(measure::Symbol)
    measure === :effect && return Effect()
    measure === :elasticity && return Eyex()
    measure === :semielasticity_dyex && return Dyex()
    measure === :semielasticity_eydx && return Eydx()
    throw(ArgumentError("Unknown measure: $measure"))
end
