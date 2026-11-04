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
        index = Dict{Any,Int}(_level_key(l) => j for (j, l) in enumerate(r.levels))
        for t in _factor_terms(m, f)
            missing_vars = [v for v in StatsModels.termvars(t) if !haskey(tbl, v)]
            isempty(missing_vars) || throw(ArgumentError(
                "random_effects = :conditional needs $(join(repr.(missing_vars), ", ")) at every " *
                "row (the random effects of $(repr(f)) depend on $(length(missing_vars) == 1 ? "it" : "them"))"))
            z = _as_matrix(StatsModels.modelcols(t.lhs, tbl))
            cn = StatsModels.coefnames(t.lhs)
            cols = [findfirst(==(c), r.cnames) for c in (cn isa AbstractVector ? cn : [cn])]
            any(isnothing, cols) && error("Internal: random-effects columns $cn not found in $(r.cnames) of $f")
            for i in 1:n
                key = _row_level(t.rhs, tbl, i)
                j = get(index, key, 0)
                if j == 0
                    spec.new_levels === :error && throw(ArgumentError(
                        "random_effects = :conditional: level $(repr(key)) of $(repr(f)) was not in the " *
                        "fitted data, so it has no predicted random effect. Pass re_new = :zero to set " *
                        "its random effects to 0."))
                    continue
                end
                @inbounds for (c, col) in enumerate(cols)
                    off[i] += z[i, c] * b[col, j]
                end
            end
        end
    end
    all(isfinite, off) || throw(ArgumentError("The random-effects contribution contains NaN or Inf values"))
    return off
end

end # module
