# build_continuous_design.jl

###############################################################################
# 1. Utility: build continuous design + derivatives in one shot (optimized)
###############################################################################
"""
build_continuous_design(df, fe_form, cts_vars)
  → X::Matrix{Float64}, Xdx::Vector{Matrix{Float64}}

Optimized to avoid repeated `merge` and intermediate Dual arrays.
"""
function build_continuous_design(df, fe_form, cts_vars::Vector{Symbol})
    n, k = nrow(df), length(cts_vars)
    if k == 0
        return Matrix{Float64}(undef, 0, 0), Vector{Matrix{Float64}}()
    end

    # Extract base columns once
    tbl0 = Tables.columntable(df)
    coldict = Dict{Symbol, AbstractVector}()
    for (nm, col) in pairs(tbl0)
        coldict[nm] = col
    end

    # Precompute ForwardDiff seed partials
    seeds = [ntuple(i -> Float64(i == j), k) for j in 1:k]

    # Replace each continuous column with Dual values in-place
    for (j, v) in enumerate(cts_vars)
        basecol = Float64.(coldict[v])
        partials = ForwardDiff.Partials{k, Float64}(seeds[j])
        dualcol = Vector{ForwardDiff.Dual{Nothing, Float64, k}}(undef, n)
        @inbounds for i in 1:n
            dualcol[i] = ForwardDiff.Dual(basecol[i], partials)
        end
        coldict[v] = dualcol
    end

    # Build a temporary DataFrame from the mutated columns
    tempdf = DataFrame(coldict)
    Xdual  = modelmatrix(fe_form, tempdf)

    # Allocate output matrices
    p   = size(Xdual, 2)
    X   = Matrix{Float64}(undef, n, p)
    Xdx = [Matrix{Float64}(undef, n, p) for _ in 1:k]

    # One-pass extraction of values and partials
    @inbounds for idx in eachindex(Xdual)
        cell  = Xdual[idx]
        X[idx] = ForwardDiff.value(cell)
        part  = ForwardDiff.partials(cell)
        for j in 1:k
            Xdx[j][idx] = part[j]
        end
    end

    return X, Xdx
end
