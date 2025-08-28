# api.jl

# Convenience wrappers for legacy API compatibility
ame(model, data; dydx=:continuous, kwargs...) = population_margins(model, data; type=:effects, vars=dydx, kwargs...)
mem(model, data; dydx=:continuous, kwargs...) = profile_margins(model, data; at=:means, type=:effects, vars=dydx, kwargs...)
mer(model, data; at, dydx=:continuous, kwargs...) = profile_margins(model, data; at=at, type=:effects, vars=dydx, kwargs...)
ape(model, data; kwargs...) = population_margins(model, data; type=:predictions, kwargs...)  
apm(model, data; kwargs...) = profile_margins(model, data; at=:means, type=:predictions, kwargs...)
apr(model, data; at, kwargs...) = profile_margins(model, data; at=at, type=:predictions, kwargs...)

"""
    _build_groups(data_nt, over, within=nothing, idxs=nothing)

Build group index (labels→row indices) for over/within variables, optionally limited to idxs.
"""
function _build_groups(data_nt::NamedTuple, over, within=nothing, idxs=nothing)
    if over === nothing && within === nothing
        return nothing
    end
    vars = Symbol[]
    if over !== nothing
        append!(vars, over isa Symbol ? [over] : collect(over))
    end
    if within !== nothing
        append!(vars, within isa Symbol ? [within] : collect(within))
    end
    # Build mapping from group label -> row indices
    groups = Dict{NamedTuple, Vector{Int}}()
    rows = idxs === nothing ? collect(1 : _nrows(data_nt)) : idxs
    for i in rows
        key = NamedTuple{Tuple(vars)}(Tuple(getproperty(data_nt, v)[i] for v in vars))
        push!(get!(groups, key, Int[]), i)
    end
    return collect(groups)
end

"""
    _split_by(data_nt, by)

Return a list of (bylabels, idxs) or nothing if `by` is nothing.
"""
function _split_by(data_nt::NamedTuple, by)
    if by === nothing
        return nothing
    end
    vars = by isa Symbol ? [by] : collect(by)
    groups = Dict{NamedTuple, Vector{Int}}()
    n = _nrows(data_nt)
    for i in 1:n
        key = NamedTuple{Tuple(vars)}(Tuple(getproperty(data_nt, v)[i] for v in vars))
        push!(get!(groups, key, Int[]), i)
    end
    return collect(groups)
end

function _try_dof_residual(model)
    try
        return dof_residual(model)
    catch
        return nothing
    end
end

# New Profile/Population API

"""
    population_margins(model, data; kwargs...) -> MarginsResult

Compute marginal effects or predictions averaged over the population (observed data).
This function implements the "population" approach from the statistical framework,
where effects/predictions are calculated for each observation and then averaged.

# Arguments
- `model`: Fitted model (GLM, LM, etc.)  
- `data`: Tables.jl-compatible data (DataFrame, etc.)

# Keywords  
- `type::Symbol = :effects`: `:effects` (derivatives/slopes) or `:predictions` (levels/values)
- `vars = :continuous`: Variables for marginal effects (ignored for predictions)
- `target::Symbol = :mu`: `:mu` (response scale) or `:eta` (link scale) for effects
- `scale::Symbol = :response`: `:response` or `:link` for predictions 
- `weights = nothing`: Observation weights (vector or column name)
- `balance = :none`: Balance factor distributions (`:none`, `:all`, or `Vector{Symbol}`)
- `over = nothing`: Grouping variables for within-group analysis
- `within = nothing`: Nested grouping structure  
- `by = nothing`: Stratification variables
- `vcov = :model`: Covariance specification (`:model`, matrix, function, or estimator)
- `backend::Symbol = :fd`: Computational backend (`:fd` or `:ad`)
- `rows = :all`: Row subset for computation
- `contrasts::Symbol = :pairwise`: Contrast type for categorical variables
- `ci_level::Real = 0.95`: Confidence interval level
- `mcompare::Symbol = :noadjust`: Multiple comparison adjustment

# Examples
```julia
# Population average marginal effects (AME)
population_margins(model, df; type=:effects, vars=[:x, :z], target=:mu)

# Population average predictions  
population_margins(model, df; type=:predictions, scale=:response)

# With grouping and weights
population_margins(model, df; type=:effects, over=:region, weights=:survey_weight)
```

This corresponds to traditional AME/APE approaches in the econometrics literature.
"""
function population_margins(
    model, data;
    type::Symbol = :effects,
    vars = :continuous, 
    target::Symbol = :mu,
    scale::Symbol = :response,
    weights = nothing,
    balance = :none,
    over = nothing,
    within = nothing, 
    by = nothing,
    vcov = :model,
    backend::Symbol = :fd,
    rows = :all,
    contrasts::Symbol = :pairwise,
    ci_level::Real = 0.95,
    mcompare::Symbol = :noadjust
)
    # Parameter validation
    type in (:effects, :predictions) || throw(ArgumentError("type must be :effects or :predictions"))
    target in (:mu, :eta) || throw(ArgumentError("target must be :mu or :eta"))
    scale in (:response, :link) || throw(ArgumentError("scale must be :response or :link"))
    
    # Build computational engine
    engine = _build_engine(model, data, vars, target)
    Σ_override = _resolve_vcov(vcov, model, length(engine.β))
    engine = (; engine..., Σ=Σ_override)
    data_nt = engine.data_nt
    
    # Handle grouping and stratification
    groups = _build_groups(data_nt, over, within)  
    strata = _split_by(data_nt, by)
    
    if type === :effects
        # Population effects - call computational engines directly
        cont_vars = [v for v in engine.vars if !_is_categorical(data_nt, v)]
        cat_vars = [v for v in engine.vars if _is_categorical(data_nt, v)]
        df_parts = DataFrame[]
        
        if groups === nothing && strata === nothing
            # No grouping - compute once
            if !isempty(cont_vars)
                eng_cont = (; engine..., vars=cont_vars)
                row_idxs = rows === :all ? collect(1:_nrows(data_nt)) : rows
                ab_subset = balance === :none ? nothing : (balance === :all ? nothing : balance)
                bw = balance === :none ? nothing : _balanced_weights(data_nt, row_idxs, ab_subset)
                w_final = _merge_weights(weights, bw, data_nt, row_idxs)
                push!(df_parts, _ame_continuous(model, data_nt, eng_cont; target=target, backend=backend, rows=row_idxs, weights=w_final))
            end
            if !isempty(cat_vars)
                eng_cat = (; engine..., vars=cat_vars)
                push!(df_parts, _categorical_effects(model, data_nt, eng_cat; target=target, contrasts=contrasts, rows=row_idxs))
            end
            df = isempty(df_parts) ? DataFrame(term=Symbol[], dydx=Float64[], se=Float64[]) : reduce(vcat, df_parts)
        else
            # Grouped computation
            all_dfs = DataFrame[]
            row_idxs = rows === :all ? collect(1:_nrows(data_nt)) : rows
            strata_list = strata === nothing ? [(NamedTuple(), row_idxs)] : strata
            for (bylabels, sidxs) in strata_list
                local_groups = groups === nothing ? [(NamedTuple(), sidxs)] : _build_groups(data_nt, over, within, sidxs)
                for (labels, idxs) in local_groups
                    if !isempty(cont_vars)
                        eng_cont = (; engine..., vars=cont_vars)
                        ab_subset = balance === :none ? nothing : (balance === :all ? nothing : balance)
                        bw = balance === :none ? nothing : _balanced_weights(data_nt, idxs, ab_subset)
                        w_final = _merge_weights(weights, bw, data_nt, idxs)
                        df_group = _ame_continuous(model, data_nt, eng_cont; target=target, backend=backend, rows=idxs, weights=w_final)
                        # Add group columns
                        for (k, v) in pairs(bylabels)
                            df_group[!, k] = fill(v, nrow(df_group))
                        end
                        for (k, v) in pairs(labels)
                            df_group[!, k] = fill(v, nrow(df_group))
                        end
                        push!(all_dfs, df_group)
                    end
                    if !isempty(cat_vars)
                        eng_cat = (; engine..., vars=cat_vars)
                        df_group = _categorical_effects(model, data_nt, eng_cat; target=target, contrasts=contrasts, rows=idxs)
                        # Add group columns
                        for (k, v) in pairs(bylabels)
                            df_group[!, k] = fill(v, nrow(df_group))
                        end
                        for (k, v) in pairs(labels)
                            df_group[!, k] = fill(v, nrow(df_group))
                        end
                        push!(all_dfs, df_group)
                    end
                end
            end
            df = isempty(all_dfs) ? DataFrame(term=Symbol[], dydx=Float64[], se=Float64[]) : reduce(vcat, all_dfs)
        end
    else  # :predictions
        # Population predictions
        target_pred = scale === :response ? :mu : :eta
        if groups === nothing && strata === nothing
            row_idxs = rows === :all ? collect(1:_nrows(data_nt)) : rows
            ab_subset = balance === :none ? nothing : (balance === :all ? nothing : balance) 
            bw = balance === :none ? nothing : _balanced_weights(data_nt, row_idxs, ab_subset)
            w_final = _merge_weights(weights, bw, data_nt, row_idxs)
            val, se = _ape(model, data_nt, engine.compiled, engine.β, engine.Σ; target=target_pred, link=engine.link, rows=row_idxs, weights=w_final)
            df = DataFrame(term=[:prediction], dydx=[val], se=[se])
        else
            # Grouped predictions
            all_dfs = DataFrame[]
            row_idxs = rows === :all ? collect(1:_nrows(data_nt)) : rows  
            strata_list = strata === nothing ? [(NamedTuple(), row_idxs)] : strata
            for (bylabels, sidxs) in strata_list
                local_groups = groups === nothing ? [(NamedTuple(), sidxs)] : _build_groups(data_nt, over, within, sidxs)
                for (labels, idxs) in local_groups
                    ab_subset = balance === :none ? nothing : (balance === :all ? nothing : balance)
                    bw = balance === :none ? nothing : _balanced_weights(data_nt, idxs, ab_subset)
                    w_final = _merge_weights(weights, bw, data_nt, idxs)
                    val, se = _ape(model, data_nt, engine.compiled, engine.β, engine.Σ; target=target_pred, link=engine.link, rows=idxs, weights=w_final)
                    df_g = DataFrame(term=[:prediction], dydx=[val], se=[se])
                    # Add group columns
                    for (k, v) in pairs(bylabels)
                        df_g[!, k] = fill(v, nrow(df_g))
                    end
                    for (k, v) in pairs(labels)
                        df_g[!, k] = fill(v, nrow(df_g))
                    end
                    push!(all_dfs, df_g)
                end
            end
            df = reduce(vcat, all_dfs)
        end
    end
    
    # Add confidence intervals
    dof = _try_dof_residual(model)
    groupcols = Symbol[]
    if over !== nothing
        append!(groupcols, over isa Symbol ? [over] : collect(over))
    end
    _add_ci!(df; level=ci_level, dof=dof, mcompare=mcompare, groupcols=groupcols)
    
    # Build result metadata
    md = (; 
        mode = type === :effects ? :effects : :predictions,
        dydx = vars, 
        target = target, 
        scale = scale,
        at = :none,  # Population approach
        backend = backend, 
        rows = rows, 
        contrasts = contrasts, 
        by = by, 
        n = _nrows(data_nt), 
        link = string(typeof(engine.link)), 
        dof = dof, 
        vcov = vcov
    )
    
    return _new_result(df; md...)
end

"""
    profile_margins(model, data; at, kwargs...) -> MarginsResult

Compute marginal effects or predictions at specific covariate profiles.
This function implements the "profile" approach from the statistical framework,
where effects/predictions are evaluated at explicit covariate combinations.

# Arguments
- `model`: Fitted model (GLM, LM, etc.)
- `data`: Tables.jl-compatible data (DataFrame, etc.)

# Keywords
- `at`: Profile specification - `:means`, `Dict`, or `Vector{Dict}` of covariate values
- `type::Symbol = :effects`: `:effects` (derivatives/slopes) or `:predictions` (levels/values)
- `vars = :continuous`: Variables for marginal effects (ignored for predictions)
- `target::Symbol = :mu`: `:mu` (response scale) or `:eta` (link scale) for effects
- `scale::Symbol = :response`: `:response` or `:link` for predictions
- `average::Bool = false`: Collapse profile grid to single summary row
- `over = nothing`: Grouping variables for within-group analysis
- `by = nothing`: Stratification variables
- `vcov = :model`: Covariance specification (`:model`, matrix, function, or estimator)
- `backend::Symbol = :ad`: Computational backend (`:ad` recommended for profiles)
- `contrasts::Symbol = :pairwise`: Contrast type for categorical variables
- `ci_level::Real = 0.95`: Confidence interval level
- `mcompare::Symbol = :noadjust`: Multiple comparison adjustment

# Examples
```julia
# Effects at sample means (MEM)
profile_margins(model, df; at=:means, type=:effects, vars=[:x, :z])

# Predictions at specific profiles (APR-style)
profile_margins(model, df; at=Dict(:x=>[-1,0,1], :group=>["A","B"]), 
                type=:predictions, scale=:response)

# MER-style: focal variable grid with others at means
profile_margins(model, df; at=Dict(:education=>[8,12,16,20], :experience=>[mean]),
                type=:effects, vars=[:education])
```

This corresponds to traditional MEM/MER/APM/APR approaches in the econometrics literature.
"""
function profile_margins(
    model, data;
    at,
    type::Symbol = :effects,
    vars = :continuous,
    target::Symbol = :mu,
    scale::Symbol = :response,
    average::Bool = false,
    over = nothing,
    by = nothing,
    vcov = :model,
    backend::Symbol = :ad,
    contrasts::Symbol = :pairwise,
    ci_level::Real = 0.95,
    mcompare::Symbol = :noadjust
)
    # Parameter validation
    type in (:effects, :predictions) || throw(ArgumentError("type must be :effects or :predictions"))
    target in (:mu, :eta) || throw(ArgumentError("target must be :mu or :eta"))
    scale in (:response, :link) || throw(ArgumentError("scale must be :response or :link"))
    at !== :none || throw(ArgumentError("profile_margins requires at ≠ :none. Use population_margins for population-averaged analysis."))
    
    # Build computational engine
    engine = _build_engine(model, data, vars, target)
    Σ_override = _resolve_vcov(vcov, model, length(engine.β))
    engine = (; engine..., Σ=Σ_override)
    data_nt = engine.data_nt
    
    # Build profiles
    profs = _build_profiles(at, data_nt)
    
    if type === :effects
        # Profile effects - call MEM/MER computational path
        cont_vars = [v for v in engine.vars if !_is_categorical(data_nt, v)]
        cat_vars = [v for v in engine.vars if _is_categorical(data_nt, v)]
        df_parts = DataFrame[]
        
        if !isempty(cont_vars)
            eng_cont = (; engine..., vars=cont_vars)
            push!(df_parts, _mem_mer_continuous(model, data_nt, eng_cont, at; target=target, backend=backend))
        end
        if !isempty(cat_vars)
            eng_cat = (; engine..., vars=cat_vars)
            push!(df_parts, _categorical_effects(model, data_nt, eng_cat; target=target, contrasts=contrasts, at=at))
        end
        
        df = isempty(df_parts) ? DataFrame(term=Symbol[], dydx=Float64[], se=Float64[]) : reduce(vcat, df_parts)
    else  # :predictions
        # Profile predictions - call APM/APR computational path
        target_pred = scale === :response ? :mu : :eta
        df = _ap_profiles(model, data_nt, engine.compiled, engine.β, engine.Σ, profs; target=target_pred, link=engine.link)
        df[!, :term] = fill(:prediction, nrow(df))
    end
    
    # Handle grouping if specified
    if over !== nothing || by !== nothing
        # For profile analysis with grouping, we need to handle this differently
        # This is a complex case - for now, throw an informative error
        if over !== nothing || by !== nothing
            @warn "Grouping with profile_margins not fully implemented yet. Use population_margins for grouped analysis."
        end
    end
    
    # Handle averaging if requested
    if average && nrow(df) > 1
        # Group by non-profile columns and average across profiles
        profile_cols = [c for c in names(df) if startswith(String(c), "at_")]
        group_cols = setdiff(names(df), ["dydx", "se", "z", "p", "ci_lo", "ci_hi"] ∪ profile_cols)
        
        if isempty(group_cols) || all(col -> length(unique(df[!, col])) == 1, group_cols)
            # Simple case: just average all rows
            avg_result = DataFrame(
                term = [df.term[1]],
                dydx = [mean(df.dydx)],
                se = [sqrt(mean(df.se.^2))],  # RMS of standard errors
            )
            # Add other metadata columns that exist
            for col in names(df)
                if !(col in ["term", "dydx", "se", "z", "p", "ci_lo", "ci_hi"]) && !startswith(String(col), "at_")
                    if length(unique(df[!, col])) == 1
                        avg_result[!, col] = [df[1, col]]
                    end
                end
            end
        else
            # Group by non-profile columns and average within groups  
            grouped = groupby(df, group_cols)
            avg_parts = []
            for group in grouped
                avg_group = DataFrame(
                    term = [group.term[1]],
                    dydx = [mean(group.dydx)],
                    se = [sqrt(mean(group.se.^2))]
                )
                # Carry over group columns
                for col in group_cols
                    avg_group[!, col] = [group[1, col]]
                end
                push!(avg_parts, avg_group)
            end
            avg_result = vcat(avg_parts...)
        end
        df = avg_result
    end
    
    # Add confidence intervals
    dof = _try_dof_residual(model)
    groupcols = Symbol[]
    if over !== nothing
        append!(groupcols, over isa Symbol ? [over] : collect(over))
    end
    # Include profile columns in grouping for CI computation
    append!(groupcols, [c for c in names(df) if startswith(String(c), "at_")])
    _add_ci!(df; level=ci_level, dof=dof, mcompare=mcompare, groupcols=groupcols)
    
    # Build result metadata
    md = (; 
        mode = type === :effects ? :effects : :predictions,
        dydx = vars, 
        target = target, 
        scale = scale,
        at = at, 
        backend = backend, 
        contrasts = contrasts, 
        by = by, 
        average = average,
        n = _nrows(data_nt), 
        link = string(typeof(engine.link)), 
        dof = dof, 
        vcov = vcov
    )
    
    return _new_result(df; md...)
end
