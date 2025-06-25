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

###############################################################################
# fixed_effects_utils.jl
#
# Dispatch on model type so GLM/OLS do nothing, MixedModels get stripped.
###############################################################################

using StatsModels

# bring in the relevant model types
import GLM: LinearModel, GeneralizedLinearModel
import MixedModels: LinearMixedModel, GeneralizedLinearMixedModel

# also need these to manipulate the formula
const RET = MixedModels.RandomEffectsTerm
const FT  = StatsModels.FunctionTerm{typeof(|)}
const CT  = StatsModels.ConstantTerm

# ─────────────────────────────────────────────────────────────────────────────
# 1) GLM/OLS methods: identity
# ─────────────────────────────────────────────────────────────────────────────

fixed_effects_form(model::StatsModels.TableRegressionModel) = formula(model)

"""
    fixed_effects_form(model::LinearModel)
    fixed_effects_form(model::GeneralizedLinearModel)

For plain OLS (`lm`) or GLM (`glm`) fits, there are no random‐effects terms, 
so we just return the original formula unchanged.
"""
fixed_effects_form(model::Union{LinearModel, GeneralizedLinearModel}) = formula(model)

# ─────────────────────────────────────────────────────────────────────────────
# 2) MixedModels methods: strip out `(…|…)`
# ─────────────────────────────────────────────────────────────────────────────

"""
    fixed_effects_form(model::LinearMixedModel)
    fixed_effects_form(model::GeneralizedLinearMixedModel)

Remove any random‐effects terms `( … | … )` from the RHS and return the
pure fixed‐effects formula.
"""
function fixed_effects_form(model::Union{LinearMixedModel,
                                          GeneralizedLinearMixedModel})
    full = formula(model)      # e.g. y ~ x + z + x&z + (1|g)
    rhs  = full.rhs            # vector of top‐level terms

    # drop any RandomEffectsTerm or the FunctionTerm for `|`
    fe = filter(t -> !(t isa RET) && !(t isa FT), rhs)

    # if nothing was removed, just hand back the original
    fe === rhs && return full

    # otherwise rebuild: if you stripped *all* terms, leave only the intercept
    new_rhs = isempty(fe) ? CT() : reduce(+, fe)
    return full.lhs ~ new_rhs
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
