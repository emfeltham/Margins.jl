# build_continuous_design.jl

###############################################################################
# 1. Utility: build continuous design + derivatives in one shot (optimized)
###############################################################################
"""
build_continuous_design(df, fe_rhs, cts_vars)
  → X::Matrix{Float64}, Xdx::Vector{Matrix{Float64}}

Optimized to avoid repeated `merge` and intermediate Dual arrays.
"""
function build_continuous_design(df, fe_rhs, cts_vars::Vector{Symbol})
    n, k = nrow(df), length(cts_vars)
    if k == 0
        # no continuous vars → zero‐column design
        return Matrix{Float64}(undef, n, 0), [Matrix{Float64}(undef, n, 0) for _ in 1:k]
    end

    # 1) Pull out the raw columns once into a Dict keyed by the *same* names
    # Skip the intermediate Dict entirely
    tbl0 = Tables.columntable(df)

    # 2) Build ForwardDiff seed vectors
    seeds = [ntuple(i -> Float64(i == j), k) for j in 1:k]

    # 3) Replace each continuous column with Duals in‐place
    # Create a new columntable with dualized columns
    dual_cols = Dict()
    for (nm, col) in pairs(tbl0)
        if nm in cts_vars
            j = findfirst(==(nm), cts_vars)  # Find index in cts_vars
            basecol = Float64.(col)
            partials = ForwardDiff.Partials{k, Float64}(seeds[j])
            dualcol = Vector{ForwardDiff.Dual{Nothing, Float64, k}}(undef, length(basecol))
            @inbounds for i in eachindex(basecol)
                dualcol[i] = ForwardDiff.Dual(basecol[i], partials)
            end
            dual_cols[nm] = dualcol
        else
            dual_cols[nm] = col  # Keep non-continuous columns as-is
        end
    end

    # 4) Build the "dualized" design matrix in one shot
    # Convert back to columntable format
    dual_tbl = (; dual_cols...)  # Named tuple from Dict
    Xdual = modelmatrix(fe_rhs, dual_tbl)

    # 5) Allocate Float64 outputs
    p   = size(Xdual, 2)
    X   = Matrix{Float64}(undef, n, p)
    Xdx = [Matrix{Float64}(undef, n, p) for _ in 1:k]

    # 6) Extract values + partials in a single pass
    @inbounds for idx in eachindex(Xdual)
        cell   = Xdual[idx]
        X[idx] = ForwardDiff.value(cell)
        part   = ForwardDiff.partials(cell)
        for j in 1:k
            Xdx[j][idx] = part[j]
        end
    end

    return X, Xdx
end


###############################################################################
# 2. Single‐column derivative helper (reuses the batch builder)
###############################################################################
"""
build_continuous_design_single!(
    df::DataFrame,
    fe_rhs,
    focal::Symbol,
    X::AbstractMatrix{Float64},
    Xdx::AbstractMatrix{Float64}
)

Mutate `X` and `Xdx` in place for the single continuous `focal` variable.
This simply calls `build_continuous_design(df, fe_rhs, [focal])`
and splats the results back into your pre-allocated buffers.
"""
function build_continuous_design_single!(
    df::DataFrame,
    fe_rhs,
    focal::Symbol,
    X::AbstractMatrix{Float64},
    Xdx::AbstractMatrix{Float64},
)
    # build the dualised design only once …
    Xfull, Xdx_list = build_continuous_design(df, fe_rhs, [focal],)

    # … and copy it directly into the caller’s buffers
    copyto!(X,    Xfull)
    copyto!(Xdx,  Xdx_list[1])
    return nothing
end
