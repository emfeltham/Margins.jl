"""
    _auto_link(model)

Extract the GLM link from a model when available; fall back to IdentityLink().
"""
function _auto_link(model)
    try
        return GLM.link(model)
    catch
        return GLM.IdentityLink()
    end
end

"""
    _dmu_deta_local(link, η)

Derivative of μ with respect to η for common links, delegating to FormulaCompiler when possible.
"""
function _dmu_deta_local(link, η)
    try
        return FormulaCompiler._dmu_deta(link, η)
    catch
        # Fallback minimal set
        if link isa GLM.IdentityLink
            return 1.0
        elseif link isa GLM.LogLink
            return exp(η)
        elseif link isa GLM.LogitLink
            σ = 1 / (1 + exp(-η))
            return σ * (1 - σ)
        else
            return 1.0
        end
    end
end
