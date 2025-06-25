
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

using GLM: linkinv, mueta, IdentityLink # GLM exports `link(::Model)`
import GLM # (to test whether accessor exists)

"""
    link_functions(model)

Return three anonymous functions  
`(invlink, dinvlink, d2invlink)` that compute  
μ(η),  dμ/dη,  d²μ/dη² for the *correct* link family behind `model`.
"""
function link_functions(model)
    fam = family(model)
    L = fam.link
    return (
        invlink   = η -> linkinv(L, η),   # μ(η)
        dinvlink  = η -> mueta(L,  η),    # dμ/dη
        d2invlink = η -> mueta2(L, η),    # d²μ/dη²  (your own helper)
    )
end
