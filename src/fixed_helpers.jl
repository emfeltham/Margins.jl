
"""
    fixed_effects_form(model)

Given a MixedModels `model`, strip out any `(…|…)` or `|` terms
and return a pure fixed-effects `FormulaTerm`.
"""
function fixed_effects_form(model)
    fullform  = formula(model)            # e.g. y ~ x + z + x&z + (1|g)
    rhs_terms = fullform.rhs        # all top-level terms

    # drop any random‐effects or |() calls
    fe_terms = filter(t ->
        !(t isa RandomEffectsTerm) &&
        !(t isa FunctionTerm{typeof(|)}),
      rhs_terms)

    # recombine with + (additive syntax)
    fe_rhs = length(fe_terms)==1 ? fe_terms[1] : reduce((a,b)->a+b, fe_terms)

    return fullform.lhs ~ fe_rhs
end

"""
    fixed_X(model, df::DataFrame)

Build the n×p fixed-effects design matrix for `df`, using exactly the
same coding (contrasts, interactions, offsets, etc.) that was used in `model`.
"""
function fixed_X(model, df::DataFrame)
    fe_form = fixed_effects_form(model)
    return modelmatrix(fe_form, df)
end

