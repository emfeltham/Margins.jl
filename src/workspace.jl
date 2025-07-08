# workspace.jl - ULTRA OPTIMIZED

"""
Ultra-optimized workspace that reuses all matrices across multiple AME computations
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
    pert_data::Dict{Symbol,Vector{Float64}}    # one n-vector per var

    function AMEWorkspace(n::Int, p::Int, df::DataFrame)
        new(
            Matrix{Float64}(undef, n, p),      # X_base   (1)
            Matrix{Float64}(undef, n, p),      # Xdx      (2)

            Vector{Float64}(undef, n),         # η
            Vector{Float64}(undef, n),         # dη
            Vector{Float64}(undef, n),         # μ′
            Vector{Float64}(undef, n),         # μ″·dη
            Vector{Float64}(undef, p),         # grad
            Vector{Float64}(undef, p),         # temp1
            Vector{Float64}(undef, p),         # temp2

            Tables.columntable(df),
            Dict{Symbol,Vector{Float64}}(),
        )
    end
end

"""
Workspace for categorical-AME computations that holds **one** n×p design matrix
and length-p scratch vectors.
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
