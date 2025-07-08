# ame_factor.jl - ULTRA OPTIMIZED

###############################################################################
# Zero-allocation categorical AMEs using InplaceModeler
###############################################################################

function _ame_factor_pair!(
    ws::FactorAMEWorkspace,
    ipm::InplaceModeler,
    β::AbstractVector, Σβ::AbstractMatrix,
    f::Symbol, lvl_i, lvl_j,
    invlink::Function, dinvlink::Function,
)
    X, η, μ, μp  = ws.X, ws.η, ws.μ, ws.μp
    buf, tmp, g  = ws.buf, ws.tmp, ws.grad
    workdf       = ws.workdf
    n            = size(X, 1)

    # ---------- level i ------------------------------------------------------
    fill!(workdf[!, f], lvl_i)
    modelmatrix!(ipm, Tables.columntable(workdf), X)

    mul!(η, X, β)                                    # η = Xβ
    @inbounds @simd for k in 1:n
        μ[k]  = invlink(η[k])
        μp[k] = dinvlink(η[k])
    end
    sumμ_i = sum(μ)
    mul!(buf, X', μp)                               # buf = X'μp   (p-vector)

    # ---------- level j (overwrite the same buffers) -------------------------
    fill!(workdf[!, f], lvl_j)
    modelmatrix!(ipm, Tables.columntable(workdf), X)

    mul!(η, X, β)
    @inbounds @simd for k in 1:n
        μ[k]  = invlink(η[k])
        μp[k] = dinvlink(η[k])
    end
    sumμ_j = sum(μ)
    mul!(tmp, X', μp)                               # tmp = X'μp

    # ---------- AME, gradient, SE -------------------------------------------
    ame = (sumμ_j - sumμ_i) / n

    @inbounds @simd for k in 1:length(β)
        g[k] = (tmp[k] - buf[k]) / n
    end
    se = sqrt(dot(g, Σβ * g))

    return ame, se, copy(g)                         # copy grad for safety
end

"""
Optimized baseline AME computation (zero allocations).
"""
function _ame_factor_baseline!(
    ame_d, se_d, grad_d,
    ipm::InplaceModeler,
    tbl0::NamedTuple, df::DataFrame,
    β::AbstractVector, Σβ::AbstractMatrix,
    f::Symbol, invlink::Function, dinvlink::Function,
)
    lvls = levels(categorical(tbl0[f]))
    base = lvls[1]

    n, p = nrow(df), length(β)
    ws    = FactorAMEWorkspace(n, p, df)

    for lvl in lvls[2:end]
        ame, se, grad = _ame_factor_pair!(
            ws, ipm, β, Σβ, f, base, lvl, invlink, dinvlink
        )
        key            = (base, lvl)
        ame_d[key]     = ame
        se_d[key]      = se
        grad_d[key]    = grad
    end
end

"""
Optimized all-pairs AME computation (zero allocations).
"""
function _ame_factor_allpairs!(
    ame_d, se_d, grad_d,
    ipm::InplaceModeler,
    tbl0::NamedTuple, df::DataFrame,
    β::AbstractVector, Σβ::AbstractMatrix,
    f::Symbol, invlink::Function, dinvlink::Function,
)
    lvls = levels(categorical(tbl0[f]))
    n, p = nrow(df), length(β)
    ws    = FactorAMEWorkspace(n, p, df)

    for i in 1:length(lvls)-1, j in i+1:length(lvls)
        lvl_i, lvl_j = lvls[i], lvls[j]
        ame, se, grad = _ame_factor_pair!(
            ws, ipm, β, Σβ, f, lvl_i, lvl_j, invlink, dinvlink
        )
        key          = (lvl_i, lvl_j)
        ame_d[key]   = ame
        se_d[key]    = se
        grad_d[key]  = grad
    end
end

