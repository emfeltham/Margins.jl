"""
    MixedModelsExt

Package extension supplying the MixedModels.jl methods of Margins' model-class interface
(`src/core/model_interface.jl`): the link function, the estimation weights and the GLMM offset.
The fixed-effects part of the formula comes from FormulaCompiler's `fixed_effects_form`; random
effects are not part of η by default (`random_effects = :zero`: every random effect 0); with
`random_effects = :conditional` the predicted random effects z'b̂ enter η as an offset (below).
The inference df is the default `Inf` (z).

The link used to be found by looking MixedModels up as `Main.MixedModels`, so a linear mixed
model failed with "Cannot determine link function" unless the user had bound the name
`MixedModels` in `Main`. Dispatching on the model types here removes the dependence on `Main`.
"""
module MixedModelsExt

using Margins
using Margins: RandomEffectsSpec, MarginsError
import MixedModels
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel, MixedModel, RandomEffectsTerm
import GLM
import StatsModels
import Tables

# A linear mixed model is Gaussian with the identity link, like `lm`.
Margins._link(::LinearMixedModel) = GLM.IdentityLink()
Margins._link(m::GeneralizedLinearMixedModel) = GLM.Link(m.resp)

# A GLMM's `offset=` (fit(MixedModel, f, df, dist; offset=…)), stored in its GLM response,
# empty without one. A LinearMixedModel has no offset.
Margins._has_offset(m::GeneralizedLinearMixedModel) = Margins._nonzero_offset(m.resp.offset)
Margins._offset(m::GeneralizedLinearMixedModel, data_nt::NamedTuple) =
    Margins._has_offset(m) ? Margins._checked_offset(m.resp.offset, data_nt) : nothing

# Estimation weights (`fit(MixedModel, f, df; wts=w)`), the default averaging weights of
# `weights=:model`. `StatsAPI.weights` returns ones for an unweighted fit, so the stored field
# decides whether the model is weighted: `sqrtwts` (√w) for a LinearMixedModel, `wt` for a
# GeneralizedLinearMixedModel; both are empty without `wts`.
Margins._model_weights(m::LinearMixedModel) = isempty(m.sqrtwts) ? nothing : Vector{Float64}(abs2.(m.sqrtwts))
Margins._model_weights(m::GeneralizedLinearMixedModel) = Margins._weights_from_storage(m.wt)

# Fixed-effect coefficients dropped as collinear (rank-deficient X): MixedModels pivots the
# fixed-effects term and keeps its rank; `coef` reports the dropped ones as (-)0 with NaN `vcov`
# rows. A GLMM holds its fixed-effects term in the embedded LinearMixedModel.
_feterm_dropped(fe) = sort!(fe.piv[(fe.rank + 1):end])
Margins._dropped_coefficients(m::LinearMixedModel) = _feterm_dropped(m.feterm)
Margins._dropped_coefficients(m::GeneralizedLinearMixedModel) = _feterm_dropped(m.LMM.feterm)

# ---- random_effects = :conditional ----------------------------------------------------------
#
# The predicted random effects (conditional modes, `ranef`) enter η as z'b̂, one term per grouping
# factor. A factor is a `ReMat` of `m.reterms` (named by `fnames`, e.g. `:rater` or
# `Symbol("village & rater")` for a nested factor); its columns of Z come from the random-effects
# terms of the fitted formula with that grouping (`(1 + w | rater)` and `(0 + x | rater)` are
# amalgamated into one `ReMat`, so a factor can have several), evaluated on the rows given with
# their schema (`modelcols`), and matched to the `ReMat`'s coefficient names. This is what
# MixedModels' `predict(m, newdata)` computes, without building a new model.

Margins._supports_random_effects(::MixedModel) = true

_lmm(m::LinearMixedModel) = m
_lmm(m::GeneralizedLinearMixedModel) = m.LMM

Margins._re_factors(m::MixedModel) = Symbol[Symbol(string(r.trm)) for r in _lmm(m).reterms]

# The random-effects terms of the fitted formula (a `zerocorr` term wraps one)
_re_term(t::RandomEffectsTerm) = t
_re_term(t::MixedModels.ZeroCorr) = _re_term(t.term)
_re_term(t) = nothing
function _formula_re_terms(m::MixedModel)
    rhs = _lmm(m).formula.rhs
    out = RandomEffectsTerm[]
    for t in (rhs isa Tuple ? rhs : (rhs,))
        re = _re_term(t)
        isnothing(re) || push!(out, re)
    end
    return out
end

_factor_terms(m::MixedModel, f::Symbol) = [t for t in _formula_re_terms(m) if Symbol(string(t.rhs)) === f]

Margins._re_grouping_vars(m::MixedModel, factors) =
    unique!(Symbol[v for f in factors for t in _factor_terms(m, f) for v in StatsModels.termvars(t.rhs)])
Margins._re_slope_vars(m::MixedModel, factors) =
    unique!(Symbol[v for f in factors for t in _factor_terms(m, f) for v in StatsModels.termvars(t.lhs)])

_level_key(v) = Margins.CategoricalArrays.unwrap(v)

# The level of grouping term `g` at row `i`: the value of its column, or a tuple of values for an
# interaction (nested) factor, as MixedModels stores `ReMat` levels
_row_level(g::StatsModels.InteractionTerm, tbl, i) = Tuple(_row_level(t, tbl, i) for t in g.terms)
_row_level(g, tbl, i) = _level_key(Tables.getcolumn(tbl, g.sym)[i])

_as_matrix(z::AbstractMatrix) = z
_as_matrix(z::AbstractVector) = reshape(z, :, 1)

function Margins._re_offset(m::MixedModel, tbl::NamedTuple, spec::RandomEffectsSpec)
    lmm = _lmm(m)
    n = length(first(tbl))
    off = zeros(Float64, n)
    b_all = MixedModels.ranef(m)
    for (k, r) in enumerate(lmm.reterms)
        f = Symbol(string(r.trm))
        f in spec.used || continue
        b = b_all[k]                                   # coefficients × levels
        for t in _factor_terms(m, f)
            missing_vars = [v for v in StatsModels.termvars(t) if !haskey(tbl, v)]
            isempty(missing_vars) || throw(ArgumentError(
                "random_effects = :conditional needs $(join(repr.(missing_vars), ", ")) at every " *
                "row (the random effects of $(repr(f)) depend on $(length(missing_vars) == 1 ? "it" : "them"))"))
            z = _as_matrix(StatsModels.modelcols(t.lhs, tbl))
            cn = StatsModels.coefnames(t.lhs)
            cols = [findfirst(==(c), r.cnames) for c in (cn isa AbstractVector ? cn : [cn])]
            any(isnothing, cols) && error("Internal: random-effects columns $cn not found in $(r.cnames) of $f")
            j = _level_indices(t.rhs, tbl, r.levels, n)
            if spec.new_levels === :error && any(iszero, j)
                i = findfirst(iszero, j)
                throw(ArgumentError(
                    "random_effects = :conditional: level $(repr(_row_level(t.rhs, tbl, i))) of $(repr(f)) " *
                    "was not in the fitted data, so it has no predicted random effect. Pass re_new = :zero " *
                    "to set its random effects to 0."))
            end
            _add_zb!(off, z, b, Vector{Int}(cols), j)
        end
    end
    all(isfinite, off) || throw(ArgumentError("The random-effects contribution contains NaN or Inf values"))
    return off
end

# The level of each row in the fitted factor's levels (0: a level absent from the fit). The keys
# are the rows' unwrapped values (a tuple for an interaction), mapped once per distinct value, so
# the lookup over rows is on a concrete key type.
function _level_indices(g, tbl, levels, n::Int)
    index = Dict{Any,Int}(_level_key(l) => j for (j, l) in enumerate(levels))
    keys_ = _row_keys(g, tbl, n)
    memo = Dict{eltype(keys_),Int}()
    out = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        key = keys_[i]
        out[i] = get!(() -> get(index, key, 0), memo, key)
    end
    return out
end
_row_keys(g::StatsModels.InteractionTerm, tbl, n) = [_row_level(g, tbl, i) for i in 1:n]
_row_keys(g, tbl, n) = _unwrapped(Tables.getcolumn(tbl, g.sym))
_unwrapped(col::Margins.CategoricalArrays.CategoricalArray) = Margins.CategoricalArrays.unwrap.(col)
_unwrapped(col) = col

function _add_zb!(off::AbstractVector{Float64}, z::AbstractMatrix, b::AbstractMatrix, cols::Vector{Int}, j::Vector{Int})
    @inbounds for i in eachindex(off)
        l = j[i]
        l == 0 && continue
        for (c, col) in enumerate(cols)
            off[i] += z[i, c] * b[col, l]
        end
    end
    return off
end

# ---- random_effects = :marginal -------------------------------------------------------------
# u = z'b is normal: the sum over factors of zᵢ_f' b_f, with b_f ~ N(0, Σ_f) independent across
# factors and Σ_f = s² λ_f λ_f' (s the dispersion, `MixedModels.dispersion`: the residual SD of an
# LMM, 1 for a Bernoulli or Poisson GLMM). With random intercepts only τ² = Σ_f σ_f² (`m.σs`); with
# random slopes τᵢ² = Σ_f zᵢ_f' Σ_f zᵢ_f varies by row.
function _reterm(m::MixedModel, f::Symbol)
    lmm = _lmm(m)
    return lmm.reterms[findfirst(r -> Symbol(string(r.trm)) === f, lmm.reterms)]
end

Margins._re_has_slopes(m::MixedModel, factors) =
    any(f -> _reterm(m, f).cnames != ["(Intercept)"], factors)

function Margins._re_intercept_variance(m::MixedModel, factors)
    σs = m.σs
    return sum(Float64(first(values(getproperty(σs, f))))^2 for f in factors; init=0.0)
end

# Z of factor `f` at the rows of `tbl`, its columns in the `ReMat`'s coefficient order (several
# formula terms may share one factor)
function _factor_Z(m::MixedModel, f::Symbol, tbl::NamedTuple)
    r = _reterm(m, f)
    n = length(first(tbl))
    Z = zeros(Float64, n, length(r.cnames))
    for t in _factor_terms(m, f)
        missing_vars = [v for v in StatsModels.termvars(t.lhs) if !haskey(tbl, v)]
        isempty(missing_vars) || throw(ArgumentError(
            "the random effects of $(repr(f)) depend on $(join(repr.(missing_vars), ", ")), which " *
            "must be given at every row"))
        z = _as_matrix(StatsModels.modelcols(t.lhs, tbl))
        cn = StatsModels.coefnames(t.lhs)
        for (c, name) in enumerate(cn isa AbstractVector ? cn : [cn])
            Z[:, findfirst(==(name), r.cnames)] .= view(z, :, c)
        end
    end
    return Z
end

_factor_Σ(m::MixedModel, f::Symbol) =
    (Λ = Matrix(_reterm(m, f).λ); Float64(MixedModels.dispersion(m))^2 .* (Λ * Λ'))

function Margins._re_spread_sq(m::MixedModel, tbl::NamedTuple, factors)
    τ² = zeros(Float64, length(first(tbl)))
    for f in factors
        _add_quadform!(τ², _factor_Z(m, f, tbl), _factor_Σ(m, f))
    end
    return τ²
end

# ---- phase 2: effects of a variable with a random slope ----------------------------------------
# ∂Z/∂w by a central difference of Z's model columns (exact, up to rounding, for terms linear in w
# such as `(1 + w | rater)`). `:conditional`: ∂(zᵢ'b̂)/∂w = (∂zᵢ/∂w)'b̂ of row i's level.
# `:marginal`: ∂τᵢ/∂w = Σ_f zᵢ_f' Σ_f (∂zᵢ_f/∂w) / τᵢ (0 where τᵢ = 0).
function Margins._re_slope_derivatives(m::MixedModel, tbl::NamedTuple, spec::RandomEffectsSpec,
                                       vars::Vector{Symbol}, τ)
    n = length(first(tbl))
    nv = length(vars)
    doff = zeros(Float64, n, nv)
    dτ = zeros(Float64, n, nv)
    factors = spec.mode === :conditional ? spec.used : spec.factors
    b_all = MixedModels.ranef(m)
    lmm = _lmm(m)
    for (k, v) in enumerate(vars)
        col = Float64.(Tables.getcolumn(tbl, v))
        h = 1e-6 .* max.(1.0, abs.(col))
        tp = merge(tbl, NamedTuple{(v,)}((col .+ h,)))
        tm = merge(tbl, NamedTuple{(v,)}((col .- h,)))
        for f in factors
            v in Margins._re_slope_vars(m, [f]) || continue
            dZ = (_factor_Z(m, f, tp) .- _factor_Z(m, f, tm)) ./ (2 .* h)
            if spec.mode === :conditional
                kf = findfirst(t -> Symbol(string(t.trm)) === f, lmm.reterms)
                r = lmm.reterms[kf]
                b = b_all[kf]
                j = _level_indices(r.trm, tbl, r.levels, n)
                _add_zb!(view(doff, :, k), dZ, b, collect(1:length(r.cnames)), j)
            else
                Z = _factor_Z(m, f, tbl); Σ = _factor_Σ(m, f)
                q = size(Z, 2)
                @inbounds for i in 1:n
                    acc = 0.0
                    for a in 1:q, bb in 1:q
                        acc += Z[i, a] * Σ[a, bb] * dZ[i, bb]
                    end
                    dτ[i, k] += acc
                end
            end
        end
        if spec.mode === :marginal
            @inbounds for i in 1:n
                dτ[i, k] = τ[i] > 0 ? dτ[i, k] / τ[i] : 0.0
            end
        end
    end
    return (doff, dτ)
end

# τ²[i] += Z[i, :]' Σ Z[i, :]
function _add_quadform!(τ²::Vector{Float64}, Z::Matrix{Float64}, Σ::Matrix{Float64})
    q = size(Z, 2)
    @inbounds for i in eachindex(τ²)
        acc = 0.0
        for a in 1:q, b in 1:q
            acc += Z[i, a] * Σ[a, b] * Z[i, b]
        end
        τ²[i] += acc
    end
    return τ²
end

end # module
