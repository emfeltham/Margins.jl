# api/population.jl - Population margins API (AME/APE)

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
- `measure::Symbol = :effect`: Effect measure - `:effect` (marginal effect), `:elasticity` (elasticity), `:semielasticity_x` (x semi-elasticity), `:semielasticity_y` (y semi-elasticity) 
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
population_margins(model, df; type = :effects, vars = [:x, :z], target = :mu)

# Population average elasticities  
population_margins(model, df; type = :effects, vars = [:x, :z], measure = :elasticity)

# Population average predictions  
population_margins(model, df; type = :predictions, scale = :response)

# With grouping and weights
population_margins(model, df; type = :effects, over = :region, weights = :survey_weight)
```

This corresponds to traditional AME/APE approaches in the econometrics literature.
"""
function population_margins(
    model, data;
    type::Symbol = :effects,
    vars = :continuous, 
    target::Symbol = :mu,
    scale::Symbol = :response,
    measure::Symbol = :effect,
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
    measure in (:effect, :elasticity, :semielasticity_x, :semielasticity_y) || throw(ArgumentError("measure must be :effect, :elasticity, :semielasticity_x, or :semielasticity_y"))
    
    # Measure parameter only applies to effects, not predictions
    if type === :predictions && measure !== :effect
        throw(ArgumentError("measure parameter only applies when type = :effects"))
    end
    
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
                push!(df_parts, _ame_continuous(model, data_nt, eng_cont; target=target, backend=backend, rows=row_idxs, measure=measure, weights=w_final))
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
                        df_group = _ame_continuous(model, data_nt, eng_cont; target=target, backend=backend, rows=idxs, measure=measure, weights=w_final)
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