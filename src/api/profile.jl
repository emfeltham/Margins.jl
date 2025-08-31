# api/profile.jl - Phase 3 Clean Profile Margins API

# ========================================================================================
# Phase 2: Unified Profile Core (from original file)
# ========================================================================================

"""
    _profile_margins_core(model, data_nt, profiles_source; kwargs...)

Unified internal entry point for all profile margins computations.
Consumes streaming profile sources and returns (df, G, metadata).
"""
function _profile_margins_core(
    model, data_nt, profiles_source;
    type::Symbol,
    vars,
    target::Symbol,
    scale::Symbol,
    measure::Symbol,
    average::Bool,
    over,
    by,
    vcov,
    backend::Symbol,
    contrasts::Symbol,
    ci_level::Real,
    mcompare::Symbol
)
    # Build computational engine
    engine = _build_engine(model, data_nt, vars, target)
    Σ_override = _resolve_vcov(vcov, model, length(engine.β))
    engine = (; engine..., Σ=Σ_override)
    
    if type === :effects
        # Profile effects - handle continuous and categorical separately
        cont_vars = [v for v in engine.vars if !_is_categorical(data_nt, v)]
        cat_vars = [v for v in engine.vars if _is_categorical(data_nt, v)]
        df_parts = DataFrame[]
        G_parts = Matrix{Float64}[]
        
        # Collect profiles once to pass to both functions
        profiles = collect(profiles_source)
        
        if !isempty(cont_vars)
            eng_cont = (; engine..., vars=cont_vars)
            df_cont, G_cont = _mem_mer_continuous_from_profiles_streaming(model, data_nt, eng_cont, profiles; target=target, backend=backend, measure=measure)
            push!(df_parts, df_cont)
            push!(G_parts, G_cont)
        end
        if !isempty(cat_vars)
            eng_cat = (; engine..., vars=cat_vars)
            df_cat, G_cat = _categorical_effects_from_profiles_streaming(model, data_nt, eng_cat, profiles; target=target, contrasts=contrasts)
            push!(df_parts, df_cat)
            push!(G_parts, G_cat)
        end
        
        # Stack df and G in identical row order
        df = isempty(df_parts) ? DataFrame(term=String[], estimate=Float64[], se=Float64[], level_from=String[], level_to=String[]) : reduce(vcat, df_parts)
        G = isempty(G_parts) ? Matrix{Float64}(undef, 0, length(engine.β)) : vcat(G_parts...)
        target_used = target
    else
        # Profile predictions
        target_pred = scale === :response ? :mu : :eta
        df, G = _ap_profiles(model, data_nt, engine.compiled, engine.β, engine.Σ, profiles_source; target=target_pred, link=engine.link)
        target_used = target_pred
    end
    
    # Handle averaging if requested
    if average && nrow(df) > 1
        # Profile averaging via averaged gradient matrix
        groupcols_str = String[]
        if over !== nothing
            append!(groupcols_str, String.(over isa Symbol ? [over] : collect(over)))
        end
        df, G = _average_rows_with_proper_se(df, G, engine.Σ; group_cols=groupcols_str)
    end
    
    # Build metadata
    metadata = (; 
        type = type,
        vars = vars, 
        target = target_used,
        scale = scale,
        backend = backend, 
        contrasts = contrasts, 
        by = by, 
        average = average,
        n = _nrows(data_nt), 
        link = string(typeof(engine.link)),
        vcov = vcov
    )
    
    return (df, G, metadata)
end

# ========================================================================================
# Phase 3: Clean Unified API
# ========================================================================================

"""
    profile_margins(model, data; at, kwargs...) -> MarginsResult

Compute marginal effects or predictions at specific covariate profiles using `at` specification.
Phase 3: Internally uses refgrid builders for clean, composable profile generation.
"""
function profile_margins(
    model, data;
    at,
    type::Symbol = :effects,
    vars = :continuous,
    target::Symbol = :mu,
    scale::Symbol = :response,
    measure::Symbol = :effect,
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
    measure in (:effect, :elasticity, :semielasticity_x, :semielasticity_y) || throw(ArgumentError("measure must be :effect, :elasticity, :semielasticity_x, or :semielasticity_y"))
    at !== :none || throw(ArgumentError("profile_margins requires at ≠ :none. Use population_margins for population-averaged analysis."))
    
    if type === :predictions && measure !== :effect
        throw(ArgumentError("measure parameter only applies when type = :effects"))
    end
    
    # Convert data to NamedTuple
    data_nt = Tables.columntable(data)
    
    # Phase 3: Use refgrid builders internally based on `at` type
    profiles_source = if at === :means
        refgrid_means(data_nt; over=over)
    elseif at isa NamedTuple
        refgrid_cartesian(at, data_nt; over=over)
    elseif at isa Vector{Pair}
        spec = NamedTuple{Tuple([p.first for p in at])}([p.second for p in at])
        refgrid_cartesian(spec, data_nt; over=over)
    else
        # Fallback for unsupported types
        to_profile_iterator(at, data_nt)
    end
    
    # Call unified core
    df, G, metadata = _profile_margins_core(
        model, data_nt, profiles_source;
        type = type, vars = vars, target = target, scale = scale, measure = measure,
        average = average, over = over, by = by, vcov = vcov,
        backend = backend, contrasts = contrasts, ci_level = ci_level, mcompare = mcompare
    )
    
    # Add CIs and build result  
    βnames = Symbol.(StatsModels.coefnames(model))
    dof = _try_dof_residual(model)
    groupcols = Symbol[]
    if over !== nothing
        append!(groupcols, over isa Symbol ? [over] : collect(over))
    end
    append!(groupcols, [Symbol(c) for c in names(df) if startswith(String(c), "at_")])
    _add_ci!(df; level=ci_level, dof=dof, mcompare=mcompare, groupcols=groupcols)
    
    md = merge(metadata, (; at = at, dof = dof))
    return _new_result(df, G, βnames, :profile, metadata.target, backend; md...)
end

"""
    profile_margins(model, data, profiles_source; kwargs...) -> MarginsResult

Phase 3: Unified method accepting both DataFrames and direct iterator sources from refgrid builders.

# Examples
```julia
# Direct builder usage (Phase 3 canonical)
profiles = refgrid_cartesian((x=[-1,0,1], z=[0]), data_nt)
result = profile_margins(model, data, profiles; type=:effects, vars=[:x])

# DataFrame reference grid (backward compatibility)
grid = DataFrame(x=[-1,0,1], z=[0,0,0])
result = profile_margins(model, data, grid; type=:effects, vars=[:x])

# With grouping built into iterator
profiles = refgrid_means(data_nt; over=:group)
result = profile_margins(model, data, profiles; type=:predictions)
```
"""
function profile_margins(
    model, data, profiles_source;
    type::Symbol = :effects,
    vars = :continuous,
    target::Symbol = :mu,
    scale::Symbol = :response,
    measure::Symbol = :effect,
    average::Bool = false,
    vcov = :model,
    backend::Symbol = :ad,
    contrasts::Symbol = :pairwise,
    ci_level::Real = 0.95,
    mcompare::Symbol = :noadjust
)
    # Convert data to NamedTuple
    data_nt = Tables.columntable(data)
    
    # Handle both DataFrames and iterators
    processed_profiles_source = if profiles_source isa AbstractDataFrame
        nrow(profiles_source) > 0 || throw(ArgumentError("reference_grid must have at least one row"))
        to_profile_iterator(profiles_source)
    else
        profiles_source  # Assume it's already an iterator from builders
    end
    
    # Call unified core
    df, G, metadata = _profile_margins_core(
        model, data_nt, processed_profiles_source;
        type = type, vars = vars, target = target, scale = scale, measure = measure,
        average = average, over = nothing, by = nothing, vcov = vcov,
        backend = backend, contrasts = contrasts, ci_level = ci_level, mcompare = mcompare
    )
    
    # Add CIs and build result  
    βnames = Symbol.(StatsModels.coefnames(model))
    dof = _try_dof_residual(model)
    groupcols = [Symbol(c) for c in names(df) if startswith(String(c), "at_") || c in [:group, :by]]
    _add_ci!(df; level=ci_level, dof=dof, mcompare=mcompare, groupcols=groupcols)
    
    at_value = profiles_source isa AbstractDataFrame ? profiles_source : "builder_source"
    md = merge(metadata, (; at = at_value, dof = dof))
    return _new_result(df, G, βnames, :profile, metadata.target, backend; md...)
end