# api/common.jl - Shared API utilities and helpers

function _try_dof_residual(model)
    try
        return dof_residual(model)
    catch
        return nothing
    end
end

"""
    _new_result(df, G, βnames, computation_type, target, backend; kwargs...)

Create a new MarginsResult with axis-based storage from DataFrame and gradient matrix.
"""
function _new_result(df::DataFrame, G::Matrix{Float64}, βnames::Vector{Symbol}, 
                    computation_type::Symbol, target::Symbol, backend::Symbol; kwargs...)
    N = nrow(df)
    
    # Build GradientMatrix
    gradients = GradientMatrix(G, βnames, computation_type, target, backend)
    
    # Extract estimates and SEs (updated column names)
    estimate = df.estimate
    se = hasproperty(df, :se) ? df.se : nothing
    
    # Build terms from unique term values
    unique_terms = unique(df.term)
    terms = AbstractTerm[]
    term_lookup = Dict{String, Int}()
    
    for (i, term_str) in enumerate(unique_terms)
        if term_str == "prediction"
            term = PredictionTerm()
        elseif contains(term_str, " → ")
            # Contrast term: "var: from → to"
            parts = split(term_str, ": ")
            var = Symbol(parts[1])
            contrast_parts = split(parts[2], " → ")
            term = ContrastTerm(var, contrast_parts[1], contrast_parts[2])
        else
            # Continuous term
            term = ContinuousTerm(Symbol(term_str))
        end
        push!(terms, term)
        term_lookup[term_str] = i
    end
    
    # Build profiles from at_* columns
    at_cols = filter(name -> startswith(string(name), "at_"), names(df))
    profiles = ProfileSpec[]
    profile_lookup = Dict{NamedTuple, Int}()
    
    if !isempty(at_cols)
        # Group by unique profile combinations
        profile_keys = Symbol[Symbol(string(col)[4:end]) for col in at_cols] # Remove "at_" prefix
        unique_profiles = unique(eachrow(select(df, at_cols)))
        
        for profile_row in unique_profiles
            values = [profile_row[col] for col in at_cols]
            profile = ProfileSpec(profile_keys, values)
            push!(profiles, profile)
            profile_lookup[NamedTuple(col => profile_row[col] for col in at_cols)] = length(profiles)
        end
    end
    
    # Build groups from remaining columns (excluding term, estimate, se, at_*)
    # Note: names(df) returns strings, but at_cols contains symbols
    excluded_cols = ["term", "estimate", "se", string.(at_cols)...]
    group_cols = setdiff(names(df), excluded_cols)
    groups = NamedTuple[]
    group_lookup = Dict{NamedTuple, Int}()
    
    # For Phase 1, skip complex grouping - just handle the simple case
    if !isempty(group_cols)
        @warn "Grouping columns detected but not yet fully implemented in Phase 1: $group_cols"
        # TODO: Implement proper grouping support
    end
    
    # Build row index vectors
    row_term = [term_lookup[df.term[i]] for i in 1:N]
    
    row_profile = if isempty(at_cols)
        zeros(Int, N)
    else
        [profile_lookup[NamedTuple(col => df[i, col] for col in at_cols)] for i in 1:N]
    end
    
    row_group = if isempty(group_cols)
        zeros(Int, N)
    else
        [group_lookup[NamedTuple(col => df[i, col] for col in group_cols)] for i in 1:N]
    end
    
    # Build metadata
    md = (; kwargs...)
    
    return MarginsResult(estimate, se, terms, profiles, groups, row_term, row_profile, row_group, gradients, md)
end

"""
    _add_ci!(df; level, dof, mcompare, groupcols)

Add confidence intervals, p-values, and z-statistics to a results DataFrame.
"""
function _add_ci!(df::DataFrame; level::Real=0.95, dof::Union{Nothing,Real}=nothing, mcompare::Symbol=:noadjust, groupcols::Vector{Symbol}=Symbol[])
    if !(:se in names(df)) || !(:dydx in names(df))
        return df
    end
    # Compute test statistic and p-values
    df[!, :z] = df.dydx ./ df.se
    if dof === nothing
        # Normal
        df[!, :p] = 2 .* (1 .- cdf.(Distributions.Normal(), abs.(df.z)))
        α = 1 - level
        q = Distributions.quantile(Distributions.Normal(), 1 - α/2)
    else
        # t distribution
        tdist = Distributions.TDist(Float64(dof))
        df[!, :p] = 2 .* (1 .- cdf.(tdist, abs.(df.z)))
        α = 1 - level
        q = Distributions.quantile(tdist, 1 - α/2)
    end
    # Multiple-comparison adjustment (Bonferroni/Sidak) for groups
    if mcompare != :noadjust && !isempty(groupcols)
        _adjust_multiple!(df; method=mcompare, groupcols)
        # Group-specific critical values for CIs
        keys = eachrow(select(df, groupcols...))
        idxs_by_group = Dict{Any, Vector{Int}}()
        for (i, key) in enumerate(keys)
            push!(get!(idxs_by_group, key, Int[]), i)
        end
        df[!, :ci_lo] = similar(df.dydx)
        df[!, :ci_hi] = similar(df.dydx)
        for idxs in values(idxs_by_group)
            k = length(idxs)
            α = 1 - level
            α_adj = mcompare === :bonferroni ? (α / k) : mcompare === :sidak ? (1 - (1 - α)^(1 / k)) : α
            qg = dof === nothing ? Distributions.quantile(Distributions.Normal(), 1 - α_adj/2) : Distributions.quantile(Distributions.TDist(Float64(dof)), 1 - α_adj/2)
            @inbounds for i in idxs
                df.ci_lo[i] = df.dydx[i] - qg * df.se[i]
                df.ci_hi[i] = df.dydx[i] + qg * df.se[i]
            end
        end
    else
        df[!, :ci_lo] = df.dydx .- q .* df.se
        df[!, :ci_hi] = df.dydx .+ q .* df.se
    end
    return df
end

function _adjust_multiple!(df::DataFrame; method::Symbol=:bonferroni, groupcols::Vector{Symbol}=Symbol[])
    # Define groups over which to adjust: default all rows together
    keys = isempty(groupcols) ? [namedtuple()] : eachrow(select(df, groupcols...))
    # Build mapping from group key to indices
    idxs_by_group = Dict{Any, Vector{Int}}()
    for (i, key) in enumerate(keys)
        push!(get!(idxs_by_group, key, Int[]), i)
    end
    # Adjust p-values within each group
    for idxs in values(idxs_by_group)
        k = length(idxs)
        if k <= 1; continue; end
        @inbounds for i in idxs
            p = df.p[i]
            p_adj = if method === :bonferroni
                min(p * k, 1.0)
            elseif method === :sidak
                1 - (1 - p)^k
            else
                p  # scheffe/noadjust fallback
            end
            df.p[i] = p_adj
        end
    end
    return df
end