"""
    MixedModelsExt

Package extension supplying the link function and the estimation weights of MixedModels.jl
models to Margins.

`_auto_link` used to look MixedModels up as `Main.MixedModels`, so a linear mixed model
failed with "Cannot determine link function" unless the user had bound the name
`MixedModels` in `Main` (`import MixedModels as MM`, or loading it from inside another
module, broke it). Dispatching on the model types here removes the dependence on `Main`.
"""
module MixedModelsExt

using Margins
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel
import GLM

# A linear mixed model is Gaussian with the identity link, like `lm`.
Margins._mixedmodel_link(::LinearMixedModel) = GLM.IdentityLink()

# MixedModels defines `GLM.Link(::GeneralizedLinearMixedModel)`, so `_auto_link` normally
# resolves GLMMs before reaching this method; it is kept as a fallback.
Margins._mixedmodel_link(m::GeneralizedLinearMixedModel) = GLM.Link(m.resp)

# Estimation weights (`fit(MixedModel, f, df; wts=w)`), the default averaging weights of
# `weights=:model`. `StatsAPI.weights` returns ones for an unweighted fit, so the stored field
# decides whether the model is weighted: `sqrtwts` (√w) for a LinearMixedModel, `wt` for a
# GeneralizedLinearMixedModel; both are empty without `wts`.
Margins._model_weights(m::LinearMixedModel) = isempty(m.sqrtwts) ? nothing : Vector{Float64}(abs2.(m.sqrtwts))
Margins._model_weights(m::GeneralizedLinearMixedModel) = Margins._weights_from_storage(m.wt)

end # module
