"""
    MixedModelsExt

Package extension supplying the MixedModels.jl methods of Margins' model-class interface
(`src/core/model_interface.jl`): the link function, the estimation weights and the GLMM offset.
The fixed-effects part of the formula comes from FormulaCompiler's `fixed_effects_form`; random
effects are not part of η here (margins are population-level, with the random effects at 0), and
the inference df is the default `Inf` (z).

The link used to be found by looking MixedModels up as `Main.MixedModels`, so a linear mixed
model failed with "Cannot determine link function" unless the user had bound the name
`MixedModels` in `Main`. Dispatching on the model types here removes the dependence on `Main`.
"""
module MixedModelsExt

using Margins
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel
import GLM

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

end # module
