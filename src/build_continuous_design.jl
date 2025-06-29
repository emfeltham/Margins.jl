# build_continuous_design.jl

###############################################################################
# 1. Utility: build continuous design + derivatives in one shot
###############################################################################
"""
build_continuous_design(df, fe_form, cts_vars)
  → X::Matrix{Float64}, Xdx::Vector{Matrix{Float64}}
"""
function build_continuous_design(df, fe_form, cts_vars::Vector{Symbol})
    n, k = nrow(df), length(cts_vars)
    if k == 0
        return Matrix{Float64}(undef,0,0), Vector{Matrix{Float64}}()
    end
    tbl0 = Tables.columntable(df)
    seeds = [ntuple(i->Float64(i==j), k) for j in 1:k]
    tbl_cts = tbl0
    for (j,v) in enumerate(cts_vars)
        # ensure values are Float64 for Dual construction
        colvals_f = Float64.(tbl0[v])
        partials  = ForwardDiff.Partials{k,Float64}(seeds[j])
        dualT     = ForwardDiff.Dual{Nothing,Float64,k}
        tbl_cts   = merge(tbl_cts, (v => dualT.(colvals_f, Ref(partials)),))
    end
    Xdual = modelmatrix(fe_form, tbl_cts)
    p     = size(Xdual,2)
    X   = Matrix{Float64}(undef, n, p)
    Xdx = [Matrix{Float64}(undef, n, p) for _ in 1:k]
    @inbounds for idx in eachindex(Xdual)
        dcell = Xdual[idx]
        X[idx] = ForwardDiff.value(dcell)
        parts  = ForwardDiff.partials(dcell)
        @inbounds for j in 1:k
            Xdx[j][idx] = parts[j]
        end
    end
    return X, Xdx
end