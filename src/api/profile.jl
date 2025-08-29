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
    gradients_for_averaging = Dict()  # Store gradients for proper averaging
    
    if type === :effects
        # Profile effects - call MEM/MER computational path
        cont_vars = [v for v in engine.vars if !_is_categorical(data_nt, v)]
        cat_vars = [v for v in engine.vars if _is_categorical(data_nt, v)]
        df_parts = DataFrame[]
        
        if !isempty(cont_vars)
            eng_cont = (; engine..., vars=cont_vars)
            df_cont, gradients_cont = _mem_mer_continuous(model, data_nt, eng_cont, at; target=target, backend=backend)
            push!(df_parts, df_cont)
            merge!(gradients_for_averaging, gradients_cont)
        end
        if !isempty(cat_vars)
            eng_cat = (; engine..., vars=cat_vars)
            push!(df_parts, _categorical_effects(model, data_nt, eng_cat; target=target, contrasts=contrasts, at=at))
        end
        
        df = isempty(df_parts) ? DataFrame(term=Symbol[], dydx=Float64[], se=Float64[]) : reduce(vcat, df_parts)
    else  # :predictions
        # Profile predictions - call APM/APR computational path
        target_pred = scale === :response ? :mu : :eta
        df, gradients_pred = _ap_profiles(model, data_nt, engine.compiled, engine.β, engine.Σ, profs; target=target_pred, link=engine.link)
        df[!, :term] = fill(:prediction, nrow(df))
        gradients_for_averaging = gradients_pred
    end
    
    # Handle grouping if specified
    if over !== nothing || by !== nothing
        # Apply grouping to profile results
        groups = _build_groups(data_nt, over, nothing)  # within not used in profiles
        strata = _split_by(data_nt, by)
        
        # Rebuild computation with grouping
        grouped_dfs = DataFrame[]
        strata_list = strata === nothing ? [(NamedTuple(), 1:_nrows(data_nt))] : strata
        
        for (bylabels, sidxs) in strata_list
            local_groups = groups === nothing ? [(NamedTuple(), sidxs)] : _build_groups(data_nt, over, nothing, sidxs)
            for (labels, idxs) in local_groups
                # Create group-specific data subset for profile computation
                group_data_nt = _subset_data(data_nt, idxs)
                group_engine = (; engine..., data_nt=group_data_nt)
                
                if type === :effects
                    cont_vars = [v for v in engine.vars if !_is_categorical(data_nt, v)]
                    cat_vars = [v for v in engine.vars if _is_categorical(data_nt, v)]
                    group_df_parts = DataFrame[]
                    
                    if !isempty(cont_vars)
                        eng_cont = (; group_engine..., vars=cont_vars)
                        df_group_cont, gradients_group_cont = _mem_mer_continuous(model, group_data_nt, eng_cont, at; target=target, backend=backend)
                        push!(group_df_parts, df_group_cont)
                    end
                    if !isempty(cat_vars)
                        eng_cat = (; group_engine..., vars=cat_vars)
                        push!(group_df_parts, _categorical_effects(model, group_data_nt, eng_cat; target=target, contrasts=contrasts, at=at))
                    end
                    
                    group_df = isempty(group_df_parts) ? DataFrame(term=Symbol[], dydx=Float64[], se=Float64[]) : reduce(vcat, group_df_parts)
                else  # :predictions
                    target_pred = scale === :response ? :mu : :eta
                    group_profs = _build_profiles(at, group_data_nt)
                    group_df, gradients_group_pred = _ap_profiles(model, group_data_nt, group_engine.compiled, group_engine.β, group_engine.Σ, group_profs; target=target_pred, link=group_engine.link)
                    group_df[!, :term] = fill(:prediction, nrow(group_df))
                end
                
                # Add group columns to results
                for (k, v) in pairs(bylabels)
                    group_df[!, k] = fill(v, nrow(group_df))
                end
                for (k, v) in pairs(labels)
                    group_df[!, k] = fill(v, nrow(group_df))
                end
                
                push!(grouped_dfs, group_df)
            end
        end
        
        df = isempty(grouped_dfs) ? DataFrame() : reduce(vcat, grouped_dfs)
    end
    
    # Handle averaging if requested
    if average && nrow(df) > 1
        profile_cols = [c for c in names(df) if startswith(String(c), "at_")]
        group_cols = setdiff(names(df), ["dydx", "se", "z", "p", "ci_lo", "ci_hi"] ∪ profile_cols)
        df = _average_profiles_with_proper_se(df, gradients_for_averaging, engine.Σ; group_cols=String.(group_cols))
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
    
    gradients_table_for_averaging = Dict()  # Store gradients for proper averaging
    
    if type === :effects
        # Profile effects - use existing computational paths but with direct profiles
        cont_vars = [v for v in engine.vars if !_is_categorical(data_nt, v)]
        cat_vars = [v for v in engine.vars if _is_categorical(data_nt, v)]
        df_parts = DataFrame[]
        
        if !isempty(cont_vars)
            eng_cont = (; engine..., vars=cont_vars)
            # Use existing function but pass profiles directly as the 'at' parameter
            df_table_cont, gradients_table_cont = _mem_mer_continuous_from_profiles(model, data_nt, eng_cont, profs; target=target, backend=backend)
            push!(df_parts, df_table_cont)
            merge!(gradients_table_for_averaging, gradients_table_cont)
        end
        if !isempty(cat_vars)
            # For now, categorical effects with table-based profiles not yet implemented
            @warn "Categorical effects with table-based reference grids not yet implemented. Only continuous effects supported."
        end
        
        df = isempty(df_parts) ? DataFrame(term=Symbol[], dydx=Float64[], se=Float64[]) : reduce(vcat, df_parts)
    else  # :predictions
        # Profile predictions - reuse existing computational path
        target_pred = scale === :response ? :mu : :eta
        df, gradients_table_pred = _ap_profiles(model, data_nt, engine.compiled, engine.β, engine.Σ, profs; target=target_pred, link=engine.link)
        df[!, :term] = fill(:prediction, nrow(df))
        gradients_table_for_averaging = gradients_table_pred
    end
    
    # Handle averaging if requested
    if average && nrow(df) > 1
        profile_cols = [c for c in names(df) if startswith(String(c), "at_")]
        group_cols = setdiff(names(df), ["dydx", "se", "z", "p", "ci_lo", "ci_hi"] ∪ profile_cols)
        df = _average_profiles_with_proper_se(df, gradients_table_for_averaging, engine.Σ; group_cols=String.(group_cols))
    end
    
    # Add confidence intervals
    dof = _try_dof_residual(model)
    groupcols = Symbol[]
    # Include profile columns in grouping for CI computation
    append!(groupcols, [Symbol(c) for c in names(df) if startswith(String(c), "at_")])
    _add_ci!(df; level=ci_level, dof=dof, mcompare=mcompare, groupcols=groupcols)
    
    # Build result metadata
    md = (; 
        mode = type === :effects ? :effects : :predictions,
        dydx = vars, 
        target = target, 
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
    
    return _new_result(df; md...)
end