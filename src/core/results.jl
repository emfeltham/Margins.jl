# core/results.jl - Result types and display

# NEW: Row-aligned gradient matrix for Phase 1 implementation
struct GradientMatrix
    G::Matrix{Float64}              # rows == number of result rows; cols == length(β)
    βnames::Vector{Symbol}          # names/order of β columns (matches model coef order)
    computation_type::Symbol        # :population | :profile
    target::Symbol                  # :eta | :mu
    backend::Symbol                 # :fd | :ad
    
    function GradientMatrix(G::Matrix{Float64}, βnames::Vector{Symbol}, 
                           computation_type::Symbol, target::Symbol, backend::Symbol)
        # Validate dimensions
        size(G, 2) == length(βnames) || error("G columns ($(size(G, 2))) must match βnames length ($(length(βnames)))")
        
        # Validate metadata
        computation_type ∈ (:population, :profile) || error("Invalid computation_type: $computation_type")
        target ∈ (:eta, :mu) || error("Invalid target: $target")  
        backend ∈ (:fd, :ad) || error("Invalid backend: $backend")
        
        return new(G, βnames, computation_type, target, backend)
    end
end

# NEW: Abstract types for result components (Phase 1 foundation)
abstract type AbstractTerm end

struct ContinuousTerm <: AbstractTerm
    var::Symbol
end

struct ContrastTerm <: AbstractTerm
    var::Symbol
    from::String
    to::String
end

struct PredictionTerm <: AbstractTerm end

struct ProfileSpec
    keys::Vector{Symbol}            # canonical order
    values::Vector{Any}             # supports reals, categorical levels, mixtures
end

struct MarginsResult
    estimate::Vector{Float64}
    se::Union{Nothing,Vector{Float64}}
    terms::Vector{AbstractTerm}
    profiles::Vector{ProfileSpec}
    groups::Vector{NamedTuple}
    row_term::Vector{Int}
    row_profile::Vector{Int}
    row_group::Vector{Int}
    gradients::GradientMatrix
    metadata::NamedTuple
    _table_cache::Union{Nothing,DataFrame}
end

function MarginsResult(
    estimate::Vector{Float64}, se, terms::Vector{AbstractTerm},
    profiles::Vector{ProfileSpec}, groups::Vector{NamedTuple},
    row_term::Vector{Int}, row_profile::Vector{Int}, row_group::Vector{Int},
    gradients::GradientMatrix, metadata::NamedTuple;
    cache::Union{Nothing,DataFrame}=nothing,
)
    N = length(estimate)
    length(row_term) == N || error("row_term length must equal number of rows")
    length(row_profile) == N || error("row_profile length must equal number of rows") 
    length(row_group) == N || error("row_group length must equal number of rows")
    size(gradients.G, 1) == N || error("Gradients rows must equal number of result rows")
    if se !== nothing
        length(se) == N || error("se length must equal number of rows")
    end
    return MarginsResult(estimate, se, terms, profiles, groups, row_term, row_profile, row_group, gradients, metadata, cache)
end

# Helper functions
rows(res::MarginsResult) = length(res.estimate)
coefnames(res::MarginsResult) = res.gradients.βnames
gradient_row(res::MarginsResult, i::Integer) = view(res.gradients.G, i, :)

# Tables.jl interface for materialization on demand
import Tables
using Distributions: Normal, cdf, quantile

Tables.istable(::Type{MarginsResult}) = true

function Tables.schema(res::MarginsResult)
    # Build schema from axis-based structure
    cols = [:term, :estimate, :se]
    types = [String, Float64, Union{Float64, Missing}]
    
    # Add profile columns if present
    if !isempty(res.profiles)
        profile_keys = unique(vcat([p.keys for p in res.profiles]...))
        for key in profile_keys
            push!(cols, Symbol("at_$key"))
            push!(types, Any)
        end
    end
    
    # Add group columns if present  
    if !isempty(res.groups) && !all(isempty(g) for g in res.groups)
        group_keys = unique(vcat([collect(keys(g)) for g in res.groups]...))
        for key in group_keys
            push!(cols, key)
            push!(types, Any)
        end
    end
    
    return Tables.Schema(cols, types)
end

function Tables.columns(res::MarginsResult)
    N = rows(res)
    
    # Basic columns
    term_col = String[_term_name(res.terms[res.row_term[i]]) for i in 1:N]
    estimate_col = res.estimate
    se_col = res.se === nothing ? [missing for _ in 1:N] : res.se
    
    result_cols = (term = term_col, estimate = estimate_col, se = se_col)
    
    # Add profile columns if present
    if !isempty(res.profiles)
        profile_keys = unique(vcat([p.keys for p in res.profiles]...))
        profile_cols = NamedTuple()
        
        for key in profile_keys
            col_name = Symbol("at_$key") 
            col_values = Vector{Any}(undef, N)
            
            for i in 1:N
                profile_idx = res.row_profile[i]
                if profile_idx > 0
                    profile = res.profiles[profile_idx]
                    key_idx = findfirst(==(key), profile.keys)
                    col_values[i] = key_idx === nothing ? missing : profile.values[key_idx]
                else
                    col_values[i] = missing
                end
            end
            
            profile_cols = merge(profile_cols, NamedTuple{(col_name,)}((col_values,)))
        end
        
        result_cols = merge(result_cols, profile_cols)
    end
    
    # Add group columns if present
    if !isempty(res.groups) && !all(isempty(g) for g in res.groups)
        group_keys = unique(vcat([collect(keys(g)) for g in res.groups]...))
        group_cols = NamedTuple()
        
        for key in group_keys
            col_values = Vector{Any}(undef, N)
            
            for i in 1:N
                group_idx = res.row_group[i]
                if group_idx > 0
                    group = res.groups[group_idx]
                    col_values[i] = get(group, key, missing)
                else
                    col_values[i] = missing
                end
            end
            
            group_cols = merge(group_cols, NamedTuple{(key,)}((col_values,)))
        end
        
        result_cols = merge(result_cols, group_cols)
    end
    
    return result_cols
end

# Helper function to convert terms to strings
_term_name(term::ContinuousTerm) = string(term.var)
_term_name(term::ContrastTerm) = "$(term.var): $(term.from) → $(term.to)"
_term_name(term::PredictionTerm) = "prediction"

function Base.show(io::IO, res::MarginsResult)
    println(io, "MarginsResult:")
    md = res.metadata
    keys_to_show = (:type, :vars, :target, :scale, :at, :backend, :vcov, :n, :link, :dof)
    parts = String[]
    for k in keys_to_show
        if hasproperty(md, k)
            push!(parts, string(k, "=", getproperty(md, k)))
        end
    end
    if !isempty(parts)
        println(io, "  ", join(parts, ", "))
    end
    
    # Show the data via Tables interface
    N = rows(res)
    nshow = min(N, 10)
    if nshow > 0
        df_preview = DataFrame(Tables.columns(res))
        show(io, first(df_preview, nshow))
        if N > nshow
            println(io, "\n  … (", N - nshow, " more rows)")
        end
    else
        println(io, "  (empty result)")
    end
end
