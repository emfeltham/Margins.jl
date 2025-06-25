###############################################################################
# fixed_effects_utils.jl
#
# • fixed_effects_form(model)  – return *only* the fixed part of the formula
# • fixed_X(model, df)        – rebuild the fixed-effects design matrix for df
#
# These functions are agnostic to whether `model` came from GLM.jl or
# MixedModels.jl; they always reuse the exact same contrasts / dummy coding /
# intercept handling that the fitted model stored in its schema.
###############################################################################

using StatsModels

"""
    fixed_effects_form(model) -> FormulaTerm

Return a formula that contains **only the fixed-effects terms** of `model`.

* If `model` is a plain GLM/OLS fit, the original formula is already fixed-
  effects only and is returned unchanged.
* If `model` is a `LinearMixedModel` or `GeneralizedLinearMixedModel`, every
  `RandomEffectsTerm` (and the syntactic sugar `(… | …)`) is stripped out.

The function works entirely at the level of `StatsModels.FormulaTerm`s, so it
is independent of how the model was fitted.
"""
function fixed_effects_form(model)
    full = formula(model)                  # StatsModels.FormulaTerm
    rhs  = full.rhs                        # vector of top-level RHS terms

    # keep everything that is *not* a random-effect or |() call
    fe_terms = filter(t ->
                !(t isa RandomEffectsTerm) &&
                !(t isa FunctionTerm{typeof(|)}), rhs)

    # if nothing was removed we can just return the original formula
    fe_terms === rhs && return full

    # rebuild RHS; if we removed *every* term, fall back to an intercept only
    fe_rhs = isempty(fe_terms) ?
             ConstantTerm() :
             reduce((a,b)->a + b, fe_terms)

    return full.lhs ~ fe_rhs
end

"""
    fixed_X(model, df::AbstractDataFrame) -> Matrix

Recreate the **fixed-effects design matrix** for the new data `df`, using the
*same schema* (contrasts, reference levels, coding of interactions, etc.) that
was stored in `model`.

Passing `model` via the `model=` keyword lets `StatsModels.modelmatrix` reuse
that schema automatically, so the returned matrix lines up 1-for-1 with the
original columns.
"""
function fixed_X(model, df::AbstractDataFrame)
    fe_form = fixed_effects_form(model)
    return modelmatrix(fe_form, df; model = model)   # <- keep coding identical
end
