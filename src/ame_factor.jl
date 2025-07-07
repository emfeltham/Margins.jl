# ame_factor.jl - OPTIMIZED WITH EfficientModelMatrices.jl

###############################################################################
# Zero-allocation categorical AMEs using InplaceModeler
###############################################################################

"""
Enhanced workspace for factor AME computations with EfficientModelMatrices
"""
struct FactorAMEWorkspace
    # Pre-allocated design matrices
    Xj::Matrix{Float64}
    Xk::Matrix{Float64}
    
    # Pre-allocated working vectors
    ηj::Vector{Float64}
    ηk::Vector{Float64}
    μj::Vector{Float64}
    μk::Vector{Float64}
    μpj::Vector{Float64}
    μpk::Vector{Float64}
    temp_j::Vector{Float64}
    temp_k::Vector{Float64}
    grad::Vector{Float64}
    
    # Working DataFrame (reused across calls)
    workdf::DataFrame
    
    function FactorAMEWorkspace(n::Int, p::Int, df::DataFrame)
        new(
            Matrix{Float64}(undef, n, p),
            Matrix{Float64}(undef, n, p),
            Vector{Float64}(undef, n),
            Vector{Float64}(undef, n),
            Vector{Float64}(undef, n),
            Vector{Float64}(undef, n),
            Vector{Float64}(undef, n),
            Vector{Float64}(undef, n),
            Vector{Float64}(undef, p),
            Vector{Float64}(undef, p),
            Vector{Float64}(undef, p),
            DataFrame(df, copycols=true)
        )
    end
end

"""
Ultra-optimized pairwise AME with InplaceModeler (zero allocations)
"""
function _ame_factor_pair!(
    ws::FactorAMEWorkspace,
    imp::InplaceModeler,
    β, Σβ,
    f::Symbol, lvl_i, lvl_j,
    invlink, dinvlink,
)
    # Unpack workspace
    Xj, Xk = ws.Xj, ws.Xk
    ηj, ηk = ws.ηj, ws.ηk
    μj, μk = ws.μj, ws.μk
    μpj, μpk = ws.μpj, ws.μpk
    temp_j, temp_k = ws.temp_j, ws.temp_k
    grad = ws.grad
    workdf = ws.workdf
    
    # Build design matrices using zero-allocation InplaceModeler
    fill!(workdf[!, f], lvl_i)
    tbl_i = Tables.columntable(workdf)
    modelmatrix!(imp, tbl_i, Xj)

    fill!(workdf[!, f], lvl_j)
    tbl_j = Tables.columntable(workdf)
    modelmatrix!(imp, tbl_j, Xk)

    # Vectorized computations
    n = size(Xj, 1)
    
    # Compute linear predictors
    mul!(ηj, Xj, β)
    mul!(ηk, Xk, β)
    
    # Vectorized link function applications
    @inbounds @simd for i in 1:n
        μj[i] = invlink(ηj[i])
        μk[i] = invlink(ηk[i])
        μpj[i] = dinvlink(ηj[i])
        μpk[i] = dinvlink(ηk[i])
    end

    # Compute AME
    ame = (sum(μk) - sum(μj)) / n

    # Compute gradient efficiently
    mul!(temp_j, Xj', μpj)
    mul!(temp_k, Xk', μpk)
    
    @inbounds @simd for i in 1:length(β)
        grad[i] = (temp_k[i] - temp_j[i]) / n
    end

    # Standard error
    se = sqrt(dot(grad, Σβ * grad))
    
    return ame, se, copy(grad)  # Copy grad since it's reused
end

"""
Optimized baseline AME computation with EfficientModelMatrices (zero allocations).
"""
function _ame_factor_baseline!(
    ame_d, se_d, grad_d,
    imp::InplaceModeler,
    tbl0::NamedTuple, df::DataFrame,
    β, Σβ,
    f::Symbol, invlink, dinvlink,
)
    lvls = levels(categorical(tbl0[f]))
    base = lvls[1]

    # Set up workspace (reused across all level pairs)
    n = length(tbl0[f])
    p = length(β)
    ws = FactorAMEWorkspace(n, p, df)
    
    # Store original column for restoration
    original_col = copy(ws.workdf[!, f])

    # Process each level against baseline
    for lvl in lvls[2:end]
        ame, se, grad = _ame_factor_pair!(
            ws, imp, β, Σβ, f, base, lvl, invlink, dinvlink
        )
        
        key = (base, lvl)
        ame_d[key] = ame
        se_d[key] = se  
        grad_d[key] = grad
    end

    # Restore original column
    ws.workdf[!, f] = original_col
end

"""
Optimized all-pairs AME computation with EfficientModelMatrices (zero allocations).
"""
function _ame_factor_allpairs!(
    ame_d, se_d, grad_d,
    imp::InplaceModeler,
    tbl0::NamedTuple, df::DataFrame,
    β, Σβ,
    f::Symbol, invlink, dinvlink,
)
    lvls = levels(categorical(tbl0[f]))

    # Set up workspace (reused across all level pairs)
    n = length(tbl0[f])
    p = length(β)
    ws = FactorAMEWorkspace(n, p, df)
    
    # Store original column for restoration
    original_col = copy(ws.workdf[!, f])

    # Process all pairs
    for i in 1:length(lvls)-1, j in i+1:length(lvls)
        lvl_i, lvl_j = lvls[i], lvls[j]

        ame, se, grad = _ame_factor_pair!(
            ws, imp, β, Σβ, f, lvl_i, lvl_j, invlink, dinvlink
        )
        
        key = (lvl_i, lvl_j)
        ame_d[key] = ame
        se_d[key] = se
        grad_d[key] = grad
    end

    # Restore original column
    ws.workdf[!, f] = original_col
end
