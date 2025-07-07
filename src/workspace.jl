# workspace.jl - ULTRA OPTIMIZED

"""
Ultra-optimized workspace that reuses all matrices across multiple AME computations
"""
mutable struct AMEWorkspace
    # Core matrices (reused across all variables)
    X_base::Matrix{Float64}
    X_pert::Matrix{Float64}
    Xdx::Matrix{Float64}
    
    # Working vectors (reused)
    η::Vector{Float64}
    dη::Vector{Float64}
    μp_vals::Vector{Float64}
    μpp_vals::Vector{Float64}
    grad_work::Vector{Float64}
    temp_vec1::Vector{Float64}
    temp_vec2::Vector{Float64}
    
    # Cached data
    base_tbl::NamedTuple
    pert_data::Dict{Symbol, Vector{Float64}}
    
    function AMEWorkspace(n::Int, p::Int, df::DataFrame)
        new(
            Matrix{Float64}(undef, n, p),
            Matrix{Float64}(undef, n, p),
            Matrix{Float64}(undef, n, p),
            Vector{Float64}(undef, n),
            Vector{Float64}(undef, n),
            Vector{Float64}(undef, n),
            Vector{Float64}(undef, n),
            Vector{Float64}(undef, p),  # grad_work size p
            Vector{Float64}(undef, p),  # temp_vec1 size p  
            Vector{Float64}(undef, p),  # temp_vec2 size p
            Tables.columntable(df),  # Compute ONCE
            Dict{Symbol, Vector{Float64}}()
        )
    end
end
