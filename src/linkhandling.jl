
# """
# Return (invlink, dinvlink, d2invlink) for model `m`.
# """
# function link_functions(model)
#     link = model.resp.link
#     return (
#       invlink   = η -> linkinv(link, η),
#       dinvlink  = η -> mueta(link,  η),
#       d2invlink = η -> mueta2(link, η),
#     )
# end

###############################################################################
# robust_link_functions.jl
#
# Drop this next to `ame_continuous_analytic.jl` and `mueta2.jl`
###############################################################################

using GLM: linkinv, mueta, IdentityLink, link     # GLM exports `link(::Model)`
import GLM                                         # (to test whether accessor exists)

"""
    _extract_link(model) -> Link

Retrieve the `Link` object regardless of whether `model` came from
`GLM.lm`, `GLM.glm`, or `MixedModels.fit!(GeneralizedLinearMixedModel, …)`.
Falls back to `IdentityLink()` for ordinary least–squares fits, which do
not carry an explicit link.
"""
function _extract_link(model)

  return if typeof(model.model) <: GeneralizedLinearModel
    # — 1 — GLM accessor (works for GeneralizedLinearModel and many others)
    model.model.rr.link
  elseif hasproperty(model, :resp) && hasproperty(model.resp, :link)
    # — 2 — MixedModels:  model.resp.link
    model.resp.link
    # — 3 — OLS (lm): implicit identity link
  else IdentityLink()
  end
end

"""
    link_functions(model)

Return three anonymous functions  
`(invlink, dinvlink, d2invlink)` that compute  
μ(η),  dμ/dη,  d²μ/dη² for the *correct* link family behind `model`.
"""
function link_functions(model)
    L = _extract_link(model)          # GLM.Link subtype (or IdentityLink)
    return (
        invlink   = η -> linkinv(L, η),   # μ(η)
        dinvlink  = η -> mueta(L,  η),    # dμ/dη
        d2invlink = η -> mueta2(L, η),    # d²μ/dη²  (your own helper)
    )
end
