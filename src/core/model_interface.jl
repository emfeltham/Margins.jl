# core/model_interface.jl
# The model-class interface (Margins 3.0, step 5): everything Margins needs to know about a
# fitted model beyond its formula, as internal generics. GLM.jl methods are here; MixedModels
# and FixedEffectModels add theirs in `ext/`. A new model class is supported by adding methods
# to these functions, never by an `isa` test elsewhere.
#
# | generic | meaning | default |
# |:--|:--|:--|
# | `_link(model)` | link function g (η = g(μ)) | `GLM.Link`, or an error |
# | `_coef(model)` | β as `Vector{Float64}` | `coef(model)` |
# | `_default_vcov(model)` | the model's own covariance (what `vcov=GLM.vcov` means) | `GLM.vcov(model)` |
# | `_dropped_coefficients(model)` | coefficients the fit dropped as collinear (`nothing`: unknown) | `nothing` |
# | `_response_name(model)` | the response variable (`nothing` if not a plain column) | the formula's lhs |
# | `_inference_df(model)` | df of the reference distribution (t or z) | `Inf` |
# | `_model_weights(model, data_nt)` | estimation weights: `nothing`, a vector, or `UnknownWeights` | `_model_weights(model)` |
# | `_has_offset(model)` | does η include a per-row offset? | `false` |
# | `_offset(model, data_nt)` | that offset, one per data row | `nothing` |
# | `_grid_offset(model, grid)` | the offset at reference-grid rows, if the class can derive it | `nothing` |
# | `_estimation_rows(model, n)` | the data rows the model was fit to (`nothing`: all) | `nothing` |
# | `_validate_request(model, req)` | model-class checks on a normalized request | no-op (`request.jl`) |
# | `_validate_focal_vars(model, vars)` | model-class checks on the `vars` named | no-op (`request.jl`) |

# ---- link ---------------------------------------------------------------------------------

"""
    _link(model) -> GLM.Link

The model's link function. GLM.jl models (`lm`: identity; `glm`: its link), bare or wrapped
in a `TableRegressionModel`; MixedModels and FixedEffectModels add methods in their extensions.
Any other model class must add a method: there is no default, because a wrong link silently
gives wrong response-scale effects.
"""
_link(model::StatsModels.TableRegressionModel) = _link(model.model)
_link(::GLM.LinearModel) = GLM.IdentityLink()
_link(model::GLM.GeneralizedLinearModel) = GLM.Link(model)
function _link(model)
    # Models outside GLM.jl that define `GLM.Link` (the GLM API for the link)
    hasmethod(GLM.Link, Tuple{typeof(model)}) && return GLM.Link(model)
    error("Cannot determine link function for model type $(typeof(model)). " *
          "Statistical correctness cannot be guaranteed without proper link function. " *
          "Supported: GLM.jl models (lm, glm), MixedModels.jl (LinearMixedModel, " *
          "GeneralizedLinearMixedModel) and FixedEffectModels.jl.")
end

# ---- coefficients and covariance ----------------------------------------------------------

"""
    _coef(model) -> Vector{Float64}

The fixed-effect coefficients β the formula's model matrix multiplies.
"""
_coef(model)::Vector{Float64} = Vector{Float64}(coef(model))

"""
    _default_vcov(model) -> AbstractMatrix

The model's own covariance matrix of β, which the default `vcov = GLM.vcov` means.
"""
_default_vcov(model) = GLM.vcov(model)

"""
    _dropped_coefficients(model) -> Union{Nothing, Vector{Int}}

The coefficients (indices into `_coef(model)`, ascending) the fit dropped as collinear with the
others, from the model's own rank information; empty for a full-rank fit, `nothing` when the
model class does not expose it (Σ's non-finite rows are used instead; see `_vcov_and_dropped`).

- GLM.jl (`dropcollinear=true`, a pivoted Cholesky factorization): the pivot positions past the
  numerical rank, `chol.p[rank+1:end]`, the coefficients GLM.jl reports as 0 with NaN `vcov`
  rows. An unpivoted Cholesky (`dropcollinear=false`) is full rank.
- MixedModels (extension): `feterm.piv[feterm.rank+1:end]`.
- FixedEffectModels (extension): the coefficients reported as 0 with a NaN variance in the
  model's stored `vcov` (it keeps no pivot).
"""
_dropped_coefficients(model) = nothing
_dropped_coefficients(model::StatsModels.TableRegressionModel) = _dropped_coefficients(model.model)
_dropped_coefficients(model::GLM.LinPredModel) = _dropped_from_pivot(model.pp)
_dropped_from_pivot(pp) = nothing
_dropped_from_pivot(::GLM.DensePredChol{<:Any, <:LinearAlgebra.Cholesky}) = Int[]
function _dropped_from_pivot(pp::GLM.DensePredChol{<:Any, <:LinearAlgebra.CholeskyPivoted})
    ch = pp.chol
    return sort!(ch.p[(ch.rank + 1):end])
end

# ---- response -----------------------------------------------------------------------------

"""
    _response_name(model) -> Union{Symbol, Nothing}

The response variable, or `nothing` when the response is not a plain column (e.g. `log(y)`)
or the model gives no formula. Used to keep the response out of reference-grid output.
"""
function _response_name(model)
    lhs = try
        StatsModels.formula(model).lhs
    catch
        return nothing
    end
    return lhs isa StatsModels.AbstractTerm && hasproperty(lhs, :sym) ? Symbol(lhs.sym) : nothing
end

# ---- inference df -------------------------------------------------------------------------

"""
    _inference_df(model) -> Float64

Degrees of freedom of the reference distribution for every test statistic, p-value and
confidence interval Margins reports for `model`: a property of the model class, not of the
covariance estimator passed as `vcov` (decision D1 of the 3.0 plan).

- **Linear models** (a GLM.jl `LinearModel`, or a `GeneralizedLinearModel` with a `Normal`
  family and an `IdentityLink`): `dof_residual(model)`, t inference. This holds for any `vcov`,
  including robust and clustered estimators (`HC1()`, `CR1(cl)`), as in Stata's `regress,
  vce(robust)` followed by `margins` and R's `coeftest(m, vcovHC(m))`.
- **FixedEffectModels** (extension): `dof_residual(model)`, which FixedEffectModels already
  adjusts for clustering (G − 1 with a cluster `vcov`), the value its own `coeftable` uses.
- **Everything else** (other GLMs, linear and generalized linear mixed models): `Inf`, z.

The public entry points take `df=` to override it (e.g. `df = G - 1` for a clustered `vcov`
on an `lm`, or `df = Inf` for asymptotic inference).
"""
function _inference_df(model)::Float64
    _is_linear_regression(model) || return Inf
    return _finite_df(GLM.dof_residual(model))
end

# A df usable for t inference, else Inf
_finite_df(d)::Float64 = (d = Float64(d); (isfinite(d) && d > 0) ? d : Inf)

# A GLM.jl linear regression: `LinearModel`, or a Normal-family identity-link GLM (bare or
# wrapped in a `TableRegressionModel`). Purely type-based, so it never errors on other models.
function _is_linear_regression(model)::Bool
    inner = model isa StatsModels.TableRegressionModel ? model.model : model
    inner isa GLM.LinearModel && return true
    if inner isa GLM.GeneralizedLinearModel
        return inner.rr.d isa Normal && GLM.Link(inner) isa GLM.IdentityLink
    end
    return false
end

"""
    _validate_df(df) -> Float64

The `df=` keyword: a positive real number or `Inf` (z inference).
"""
function _validate_df(df)::Float64
    df isa Real || throw(ArgumentError("df must be a positive real number or Inf, got $(repr(df))"))
    d = Float64(df)
    (d > 0 && !isnan(d)) || throw(ArgumentError("df must be a positive real number or Inf, got $df"))
    return d
end

# ---- estimation rows ----------------------------------------------------------------------

"""
    _estimation_rows(model, n) -> Union{Nothing, Vector{Int}}

The rows of the `n`-row data the model was estimated on, or `nothing` for all of them.
FixedEffectModels returns its `esample` (rows dropped for missing values, non-positive weights
and singleton fixed-effect groups). The request subsets the data to these rows once, before
anything else reads it, so averages, weights, groups and offsets all refer to the estimation
sample. GLM.jl and MixedModels keep no record of dropped rows, so their data must already be
the estimation sample (the weights and offset length checks catch a mismatch).
"""
_estimation_rows(model, n::Int) = nothing

# ---- offsets ------------------------------------------------------------------------------

"""
    _has_offset(model) -> Bool

Whether the model's linear predictor includes a per-row offset, η = x'β + offset.
"""
_has_offset(model) = false
_has_offset(model::StatsModels.TableRegressionModel) = _has_offset(model.model)
_has_offset(model::Union{GLM.LinearModel, GLM.GeneralizedLinearModel}) = _nonzero_offset(model.rr.offset)

# GLM.jl stores an empty vector (glm) or zeros (lm) when there is no offset
_nonzero_offset(off) = !isempty(off) && any(!iszero, off)

"""
    _offset(model, data_nt) -> Union{Nothing, Vector{Float64}}

The model's offset at each row of `data_nt` (the estimation rows, see `_estimation_rows`), or
`nothing` when it has none. GLM.jl: `rr.offset` (the `offset=` given to `glm`); MixedModels
(extension): the GLMM response's `offset`; FixedEffectModels (extension): the sum of the
estimated absorbed fixed effects, Σₖ feₖ. An offset stored per fitted row must have one entry
per data row; otherwise it cannot be matched to rows and this is an `ArgumentError`.
"""
_offset(model, data_nt::NamedTuple) = nothing
_offset(model::StatsModels.TableRegressionModel, data_nt::NamedTuple) = _offset(model.model, data_nt)
function _offset(model::Union{GLM.LinearModel, GLM.GeneralizedLinearModel}, data_nt::NamedTuple)
    _has_offset(model) || return nothing
    return _checked_offset(model.rr.offset, data_nt)
end

# A stored offset as Vector{Float64}, checked against the data's row count
function _checked_offset(off::AbstractVector, data_nt::NamedTuple)
    n = length(first(data_nt))
    length(off) == n || throw(ArgumentError(
        "The model has an offset with $(length(off)) entries but the data has $n rows, so the " *
        "offset cannot be matched to rows (the fit probably dropped rows with missing values). " *
        "Pass the data the model was fit to, without the dropped rows."))
    v = Vector{Float64}(off)
    all(isfinite, v) || throw(ArgumentError("The model's offset contains NaN or Inf values"))
    return v
end

"""
    _grid_offset(model, grid::DataFrame) -> Union{Nothing, Vector{Float64}}

The model's offset at each row of a reference grid when the model class can derive it from
the grid's columns (FixedEffectModels: the estimated fixed effects of the levels the grid
sets), else `nothing`. Consulted only when the caller gives neither `offset=` nor an `offset`
grid column.
"""
_grid_offset(model, grid::DataFrame) = nothing

"""
    _offset_matters(link, type, scale, measure) -> Bool

Whether an η offset can change the requested quantity. Predictions always depend on it. An
effect depends on it only through ŷ and the link derivatives at η: on the response scale of a
non-identity link, or for a measure that divides by ŷ (`:elasticity`, `:semielasticity_eydx`).
On the link scale, or with an identity link, the `:effect` and `:semielasticity_dyex`
quantities do not involve η at all, so the offset is not needed (and not required at profile
rows).
"""
function _offset_matters(link, type::Symbol, scale::Symbol, measure::Symbol)::Bool
    type === :predictions && return true
    measure in (:elasticity, :semielasticity_eydx) && return true
    return scale === :response && !(link isa GLM.IdentityLink)
end

# ---- estimation weights -------------------------------------------------------------------

"""
    UnknownWeights(reason)

Returned by `_model_weights` for a model class that may have been fit with weights it does
not store (FixedEffectModels). `weights=:model` then averages without weights and warns once
per session (`reason` is the warning's explanation).
"""
struct UnknownWeights
    reason::String
end
