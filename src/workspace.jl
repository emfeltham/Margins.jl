# workspace.jl - ALLOCATION-FREE VERSION

"""
Ultra-optimized workspace that reuses all matrices and vectors across AME computations.
Key changes: pre-allocate ALL perturbation vectors and eliminate runtime allocations.
"""
mutable struct AMEWorkspace
    X_base ::Matrix{Float64}   # design matrix for current data / rep combo
    Xdx    ::Matrix{Float64}   # finite-difference / perturbed matrix

    η      ::Vector{Float64}
    dη     ::Vector{Float64}
    μp_vals::Vector{Float64}
    μpp_vals::Vector{Float64}
    grad_work::Vector{Float64}
    temp1  ::Vector{Float64}
    temp2  ::Vector{Float64}

    base_tbl ::NamedTuple                      # cached column-table
    pert_data::Dict{Symbol,Vector{Float64}}    # PRE-ALLOCATED perturbation vectors
    
    # NEW: Pre-allocated storage for NamedTuple merging results
    pert_cache::Dict{Symbol,NamedTuple}        # cached perturbed NamedTuples

    function AMEWorkspace(n::Int, p::Int, df::DataFrame)
        # Extract all continuous variables from the DataFrame upfront
        pert_data = Dict{Symbol,Vector{Float64}}()
        pert_cache = Dict{Symbol,NamedTuple}()
        
        base_tbl = Tables.columntable(df)
        
        # Pre-allocate perturbation vectors for ALL continuous variables
        for (name, col) in pairs(base_tbl)
            if eltype(col) <: Real && eltype(col) != Bool
                pert_data[name] = Vector{Float64}(undef, n)
                # Pre-create the merged NamedTuple structure (reused later)
                pert_cache[name] = merge(base_tbl, (name => pert_data[name],))
            end
        end
        
        new(
            Matrix{Float64}(undef, n, p),      # X_base
            Matrix{Float64}(undef, n, p),      # Xdx
            Vector{Float64}(undef, n),         # η
            Vector{Float64}(undef, n),         # dη
            Vector{Float64}(undef, n),         # μ′
            Vector{Float64}(undef, n),         # μ″·dη
            Vector{Float64}(undef, p),         # grad
            Vector{Float64}(undef, p),         # temp1
            Vector{Float64}(undef, p),         # temp2
            base_tbl,
            pert_data,
            pert_cache,
        )
    end
end

"""
Workspace for categorical-AME computations - unchanged, already optimal
"""
mutable struct FactorAMEWorkspace
    X   ::Matrix{Float64}      # single design-matrix buffer (n × p)

    η   ::Vector{Float64}      # n-length work vectors
    μ   ::Vector{Float64}
    μp  ::Vector{Float64}

    buf ::Vector{Float64}      # p-length scratch vectors
    tmp ::Vector{Float64}
    grad::Vector{Float64}

    workdf::DataFrame          # reused DataFrame

    function FactorAMEWorkspace(n::Int, p::Int, df::DataFrame)
        new(
            Matrix{Float64}(undef, n, p),
            Vector{Float64}(undef, n),
            Vector{Float64}(undef, n),
            Vector{Float64}(undef, n),
            Vector{Float64}(undef, p),
            Vector{Float64}(undef, p),
            Vector{Float64}(undef, p),
            DataFrame(df, copycols = true),
        )
    end
end