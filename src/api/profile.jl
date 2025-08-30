# api/profile.jl - Profile margins API (MEM/MER/APM/APR)

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
- `measure::Symbol = :effect`: Effect measure - `:effect` (marginal effect), `:elasticity` (elasticity), `:semielasticity_x` (x semi-elasticity), `:semielasticity_y` (y semi-elasticity)
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
profile_margins(model, df; at = :means, type = :effects, vars = [:x, :z])

# Elasticities at sample means
profile_margins(model, df; at = :means, type = :effects, vars = [:x, :z], measure = :elasticity)

# Predictions at specific profiles (APR-style)
profile_margins(
    model, df; at = Dict(:x => [-1,0,1], :group => ["A","B"]), 
    type = :predictions, scale = :response
)

# MER-style: focal variable grid with others at means
profile_margins(
    model, df; at = Dict(:education => [8,12,16,20], :experience => [mean]),
    type = :effects, vars = [:education]
)
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
    
    # Measure parameter only applies to effects, not predictions
    if type === :predictions && measure !== :effect
        throw(ArgumentError("measure parameter only applies when type = :effects"))
    end
    
    # Build computational engine
    engine = _build_engine(model, data, vars, target)
    Σ_override = _resolve_vcov(vcov, model, length(engine.β))
    engine = (; engine..., Σ=Σ_override)
    data_nt = engine.data_nt
    
    # Build profiles
    profs = _build_profiles(at, data_nt)
    
    if type === :effects
        # Profile effects - now supported with (df, G) format
        cont_vars = [v for v in engine.vars if !_is_categorical(data_nt, v)]
        cat_vars = [v for v in engine.vars if _is_categorical(data_nt, v)]
        df_parts = DataFrame[]
        G_parts = Matrix{Float64}[]
        
        if !isempty(cont_vars)
            eng_cont = (; engine..., vars=cont_vars)
            df_cont, G_cont = _mem_mer_continuous(model, data_nt, eng_cont, at; target=target, backend=backend, measure=measure)
            push!(df_parts, df_cont)
            push!(G_parts, G_cont)
        end
        if !isempty(cat_vars)
            eng_cat = (; engine..., vars=cat_vars)
            df_cat, G_cat = _categorical_effects(model, data_nt, eng_cat; target=target, contrasts=contrasts, at=at)
            push!(df_parts, df_cat)
            push!(G_parts, G_cat)
        end
        
        # Stack df and G in identical row order
        df = isempty(df_parts) ? DataFrame(term=String[], estimate=Float64[], se=Float64[], level_from=String[], level_to=String[]) : reduce(vcat, df_parts)
        G = isempty(G_parts) ? Matrix{Float64}(undef, 0, length(engine.β)) : vcat(G_parts...)
        target_used = target
    else
        # Profile predictions - call APM/APR computational path
        target_pred = scale === :response ? :mu : :eta
        df, G = _ap_profiles(model, data_nt, engine.compiled, engine.β, engine.Σ, profs; target=target_pred, link=engine.link)
        target_used = target_pred
    end
    
    # Get coefficient names for gradient matrix
    βnames = Symbol.(StatsModels.coefnames(model))
    
    # Handle grouping if specified
    if over !== nothing || by !== nothing
        # Grouped profiles require additional implementation for gradient aggregation
        error("Grouped profile margins not yet implemented; use `population_margins` with grouping or ungrouped `profile_margins`")
    end
    
    # Handle averaging if requested
    if average && nrow(df) > 1
        # Profile averaging - now supported with row-aligned gradient matrix
        groupcols_str = String[]
        if over !== nothing
            append!(groupcols_str, String.(over isa Symbol ? [over] : collect(over)))
        end
        df, G = _average_rows_with_proper_se(df, G, engine.Σ; group_cols=groupcols_str)
    end
    
    # Add confidence intervals
    dof = _try_dof_residual(model)
    groupcols = Symbol[]
    if over !== nothing
        append!(groupcols, over isa Symbol ? [over] : collect(over))
    end
    # Include profile columns in grouping for CI computation
    append!(groupcols, [Symbol(c) for c in names(df) if startswith(String(c), "at_")])
    _add_ci!(df; level=ci_level, dof=dof, mcompare=mcompare, groupcols=groupcols)
    
    # Build result metadata
    md = (; 
        type = type,
        vars = vars, 
        target = target_used,  # Use actual target used for computation
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
    
    # Use new builder with gradient matrix
    computation_type = :profile
    return _new_result(df, G, βnames, computation_type, target_used, backend; md...)
end

"""
    profile_margins(model, data, reference_grid::AbstractDataFrame; kwargs...) -> MarginsResult

Compute marginal effects or predictions using a direct reference grid specification.
This method provides maximum control by accepting the reference grid as a DataFrame,
bypassing the `at` parameter parsing entirely.

# Arguments
- `model`: Fitted model (GLM, LM, etc.)
- `data`: Original data used to fit the model
- `reference_grid::AbstractDataFrame`: Table specifying exact covariate combinations for evaluation

# Keywords
- `type::Symbol = :effects`: `:effects` (derivatives/slopes) or `:predictions` (levels/values)
- `vars = :continuous`: Variables for marginal effects (ignored for predictions)
- `target::Symbol = :mu`: `:mu` (response scale) or `:eta` (link scale) for effects
- `scale::Symbol = :response`: `:response` or `:link` for predictions
- `measure::Symbol = :effect`: Effect measure - `:effect` (marginal effect), `:elasticity` (elasticity), `:semielasticity_x` (x semi-elasticity), `:semielasticity_y` (y semi-elasticity)
- `average::Bool = false`: Collapse reference grid to single summary row
- `vcov = :model`: Covariance specification (`:model`, matrix, function, or estimator)
- `backend::Symbol = :ad`: Computational backend (`:ad` recommended for profiles)
- `contrasts::Symbol = :pairwise`: Contrast type for categorical variables
- `ci_level::Real = 0.95`: Confidence interval level
- `mcompare::Symbol = :none`: Multiple comparison adjustment

# Examples
```julia
# Custom reference grid with exact control
reference_grid = DataFrame(
    x1 = [1.0, 2.0, 1.0, 2.0],
    x2 = [0.5, 0.5, 1.5, 1.5], 
    treated = [true, false, true, false]
)

# Effects at these exact combinations
result = profile_margins(model, data, reference_grid; 
                        type = :effects, vars = [:x1])

# Predictions at these exact combinations
result = profile_margins(model, data, reference_grid;
                        type = :predictions, scale = :response)
```

This approach provides maximum flexibility for complex reference grid specifications
that may be difficult to express through the `at` parameter Dict syntax.
"""
function profile_margins(
    model, data, reference_grid::AbstractDataFrame;
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
    mcompare::Symbol = :none
)
    # Parameter validation
    type in (:effects, :predictions) || throw(ArgumentError("type must be :effects or :predictions"))
    target in (:mu, :eta) || throw(ArgumentError("target must be :mu or :eta"))
    scale in (:response, :link) || throw(ArgumentError("scale must be :response or :link"))
    measure in (:effect, :elasticity, :semielasticity_x, :semielasticity_y) || throw(ArgumentError("measure must be :effect, :elasticity, :semielasticity_x, or :semielasticity_y"))
    
    # Measure parameter only applies to effects, not predictions
    if type === :predictions && measure !== :effect
        throw(ArgumentError("measure parameter only applies when type = :effects"))
    end
    
    # Validate reference_grid
    nrow(reference_grid) > 0 || throw(ArgumentError("reference_grid must have at least one row"))
    
    # Build computational engine
    engine = _build_engine(model, data, vars, target)
    Σ_override = _resolve_vcov(vcov, model, length(engine.β))
    engine = (; engine..., Σ=Σ_override)
    data_nt = engine.data_nt
    
    # Convert reference_grid DataFrame to Vector{Dict} format expected by internal functions
    profs = Dict{Symbol,Any}[]
    for row in eachrow(reference_grid)
        prof_dict = Dict{Symbol,Any}()
        for (col, val) in pairs(row)
            prof_dict[col] = val
        end
        push!(profs, prof_dict)
    end
    
    if type === :effects
        # Profile effects - now supported with (df, G) format (table-based)
        cont_vars = [v for v in engine.vars if !_is_categorical(data_nt, v)]
        cat_vars = [v for v in engine.vars if _is_categorical(data_nt, v)]
        df_parts = DataFrame[]
        G_parts = Matrix{Float64}[]
        
        if !isempty(cont_vars)
            eng_cont = (; engine..., vars=cont_vars)
            df_cont, G_cont = _mem_mer_continuous_from_profiles(model, data_nt, eng_cont, profs; target=target, backend=backend, measure=measure)
            push!(df_parts, df_cont)
            push!(G_parts, G_cont)
        end
        if !isempty(cat_vars)
            eng_cat = (; engine..., vars=cat_vars)
            df_cat, G_cat = _categorical_effects_from_profiles(model, data_nt, eng_cat, profs; target=target, contrasts=contrasts)
            push!(df_parts, df_cat)
            push!(G_parts, G_cat)
        end
        
        # Stack df and G in identical row order
        df = isempty(df_parts) ? DataFrame(term=String[], estimate=Float64[], se=Float64[], level_from=String[], level_to=String[]) : reduce(vcat, df_parts)
        G = isempty(G_parts) ? Matrix{Float64}(undef, 0, length(engine.β)) : vcat(G_parts...)
        target_used = target
    else
        # Profile predictions - reuse existing computational path
        target_pred = scale === :response ? :mu : :eta
        df, G = _ap_profiles(model, data_nt, engine.compiled, engine.β, engine.Σ, profs; target=target_pred, link=engine.link)
        target_used = target_pred
        # Note: term column already added by _ap_profiles in Phase 1
    end
    
    # Get coefficient names for gradient matrix
    βnames = Symbol.(StatsModels.coefnames(model))
    
    # Handle averaging if requested
    if average && nrow(df) > 1
        # Profile averaging (table-based) - now supported with row-aligned gradient matrix
        df, G = _average_rows_with_proper_se(df, G, engine.Σ; group_cols=String[])
    end
    
    # Add confidence intervals
    dof = _try_dof_residual(model)
    groupcols = Symbol[]
    # Include profile columns in grouping for CI computation
    append!(groupcols, [Symbol(c) for c in names(df) if startswith(String(c), "at_")])
    _add_ci!(df; level=ci_level, dof=dof, mcompare=mcompare, groupcols=groupcols)
    
    # Build result metadata
    md = (; 
        type = type,
        vars = vars, 
        target = target_used,  # Use actual target used for computation
        scale = scale,
        at = reference_grid,  # Store the original reference grid
        backend = backend, 
        contrasts = contrasts, 
        by = nothing,  # Grouping not supported in this dispatch
        average = average,
        n = _nrows(data_nt), 
        link = string(typeof(engine.link)), 
        dof = dof, 
        vcov = vcov
    )
    
    # Use new builder with gradient matrix
    computation_type = :profile
    return _new_result(df, G, βnames, computation_type, target_used, backend; md...)
end
