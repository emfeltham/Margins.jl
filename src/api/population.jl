# api/population.jl - Population margins API (AME/APE)
using Distributions

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
            row_idxs = rows === :all ? collect(1:_nrows(data_nt)) : rows
            df_parts = DataFrame[]
            G_parts = Matrix{Float64}[]
            
            if !isempty(cont_vars)
                eng_cont = (; engine..., vars=cont_vars)
                ab_subset = balance === :none ? nothing : (balance === :all ? nothing : balance)
                bw = balance === :none ? nothing : _balanced_weights(data_nt, row_idxs, ab_subset)
                w_final = _merge_weights(weights, bw, data_nt, row_idxs)
                df_cont, G_cont = _ame_continuous(model, data_nt, eng_cont; target=target, backend=backend, rows=row_idxs, measure=measure, weights=w_final)
                push!(df_parts, df_cont)
                push!(G_parts, G_cont)
            end
            if !isempty(cat_vars)
                eng_cat = (; engine..., vars=cat_vars)
                df_cat, G_cat = _categorical_effects(model, data_nt, eng_cat; target=target, contrasts=contrasts, rows=row_idxs, at=:none)
                push!(df_parts, df_cat)
                push!(G_parts, G_cat)
            end
            
            # Stack df and G in identical row order
            df = isempty(df_parts) ? DataFrame(term=String[], estimate=Float64[], se=Float64[], level_from=String[], level_to=String[]) : reduce(vcat, df_parts)
            G_all = isempty(G_parts) ? Matrix{Float64}(undef, 0, length(engine.β)) : vcat(G_parts...)
        else
            # Grouped computation - now supported with (df, G) format
            df_parts = DataFrame[]
            G_parts = Matrix{Float64}[]
            
            # Handle stratification (by)
            if strata !== nothing
                for (stratum_key, stratum_idxs) in strata
                    # For each stratum, compute over groups within that stratum
                    stratum_groups = groups === nothing ? [(NamedTuple(), stratum_idxs)] : 
                                   [(group_key, intersect(group_idxs, stratum_idxs)) for (group_key, group_idxs) in groups]
                    
                    for (group_key, group_idxs) in stratum_groups
                        if isempty(group_idxs)
                            continue  # Skip empty groups
                        end
                        
                        # Combine stratum and group keys
                        combined_key = merge(stratum_key, group_key)
                        
                        # Compute effects for this group
                        if !isempty(cont_vars)
                            eng_cont = (; engine..., vars=cont_vars)
                            ab_subset = balance === :none ? nothing : (balance === :all ? nothing : balance)
                            bw = balance === :none ? nothing : _balanced_weights(data_nt, group_idxs, ab_subset)
                            w_final = _merge_weights(weights, bw, data_nt, group_idxs)
                            df_cont, G_cont = _ame_continuous(model, data_nt, eng_cont; target=target, backend=backend, rows=group_idxs, measure=measure, weights=w_final)
                            
                            # Add group columns to df_cont
                            for (k, v) in pairs(combined_key)
                                df_cont[!, k] = fill(v, nrow(df_cont))
                            end
                            
                            push!(df_parts, df_cont)
                            push!(G_parts, G_cont)
                        end
                        if !isempty(cat_vars)
                            eng_cat = (; engine..., vars=cat_vars)
                            df_cat, G_cat = _categorical_effects(model, data_nt, eng_cat; target=target, contrasts=contrasts, rows=group_idxs, at=:none)
                            
                            # Add group columns to df_cat
                            for (k, v) in pairs(combined_key)
                                df_cat[!, k] = fill(v, nrow(df_cat))
                            end
                            
                            push!(df_parts, df_cat)
                            push!(G_parts, G_cat)
                        end
                    end
                end
            else
                # Just grouping, no stratification
                for (group_key, group_idxs) in groups
                    if isempty(group_idxs)
                        continue  # Skip empty groups
                    end
                    
                    # Compute effects for this group
                    if !isempty(cont_vars)
                        eng_cont = (; engine..., vars=cont_vars)
                        ab_subset = balance === :none ? nothing : (balance === :all ? nothing : balance)
                        bw = balance === :none ? nothing : _balanced_weights(data_nt, group_idxs, ab_subset)
                        w_final = _merge_weights(weights, bw, data_nt, group_idxs)
                        df_cont, G_cont = _ame_continuous(model, data_nt, eng_cont; target=target, backend=backend, rows=group_idxs, measure=measure, weights=w_final)
                        
                        # Add group columns to df_cont
                        for (k, v) in pairs(group_key)
                            df_cont[!, k] = fill(v, nrow(df_cont))
                        end
                        
                        push!(df_parts, df_cont)
                        push!(G_parts, G_cont)
                    end
                    if !isempty(cat_vars)
                        eng_cat = (; engine..., vars=cat_vars)
                        df_cat, G_cat = _categorical_effects(model, data_nt, eng_cat; target=target, contrasts=contrasts, rows=group_idxs, at=:none)
                        
                        # Add group columns to df_cat
                        for (k, v) in pairs(group_key)
                            df_cat[!, k] = fill(v, nrow(df_cat))
                        end
                        
                        push!(df_parts, df_cat)
                        push!(G_parts, G_cat)
                    end
                end
            end
            
            # Stack df and G in identical row order
            df = isempty(df_parts) ? DataFrame(term=String[], estimate=Float64[], se=Float64[], level_from=String[], level_to=String[]) : reduce(vcat, df_parts)
            G_all = isempty(G_parts) ? Matrix{Float64}(undef, 0, length(engine.β)) : vcat(G_parts...)
        end
    else  # :predictions
        # Population predictions (APE) - now supported with (df, G) format
        if groups === nothing && strata === nothing
            # No grouping - compute once
            row_idxs = rows === :all ? collect(1:_nrows(data_nt)) : rows
            target_pred = scale === :response ? :mu : :eta
            ab_subset = balance === :none ? nothing : (balance === :all ? nothing : balance)
            bw = balance === :none ? nothing : _balanced_weights(data_nt, row_idxs, ab_subset)
            w_final = _merge_weights(weights, bw, data_nt, row_idxs)
            df, G_all = _ape(model, data_nt, engine.compiled, engine.β, engine.Σ; target=target_pred, link=engine.link, rows=row_idxs, weights=w_final)
        else
            # Grouped computation - now supported with (df, G) format
            df_parts = DataFrame[]
            G_parts = Matrix{Float64}[]
            target_pred = scale === :response ? :mu : :eta
            
            # Handle stratification (by)
            if strata !== nothing
                for (stratum_key, stratum_idxs) in strata
                    # For each stratum, compute over groups within that stratum
                    stratum_groups = groups === nothing ? [(NamedTuple(), stratum_idxs)] : 
                                   [(group_key, intersect(group_idxs, stratum_idxs)) for (group_key, group_idxs) in groups]
                    
                    for (group_key, group_idxs) in stratum_groups
                        if isempty(group_idxs)
                            continue  # Skip empty groups
                        end
                        
                        # Combine stratum and group keys
                        combined_key = merge(stratum_key, group_key)
                        
                        # Compute predictions for this group
                        ab_subset = balance === :none ? nothing : (balance === :all ? nothing : balance)
                        bw = balance === :none ? nothing : _balanced_weights(data_nt, group_idxs, ab_subset)
                        w_final = _merge_weights(weights, bw, data_nt, group_idxs)
                        df_pred, G_pred = _ape(model, data_nt, engine.compiled, engine.β, engine.Σ; target=target_pred, link=engine.link, rows=group_idxs, weights=w_final)
                        
                        # Add group columns to df_pred
                        for (k, v) in pairs(combined_key)
                            df_pred[!, k] = fill(v, nrow(df_pred))
                        end
                        
                        push!(df_parts, df_pred)
                        push!(G_parts, G_pred)
                    end
                end
            else
                # Just grouping, no stratification
                for (group_key, group_idxs) in groups
                    if isempty(group_idxs)
                        continue  # Skip empty groups
                    end
                    
                    # Compute predictions for this group
                    ab_subset = balance === :none ? nothing : (balance === :all ? nothing : balance)
                    bw = balance === :none ? nothing : _balanced_weights(data_nt, group_idxs, ab_subset)
                    w_final = _merge_weights(weights, bw, data_nt, group_idxs)
                    df_pred, G_pred = _ape(model, data_nt, engine.compiled, engine.β, engine.Σ; target=target_pred, link=engine.link, rows=group_idxs, weights=w_final)
                    
                    # Add group columns to df_pred
                    for (k, v) in pairs(group_key)
                        df_pred[!, k] = fill(v, nrow(df_pred))
                    end
                    
                    push!(df_parts, df_pred)
                    push!(G_parts, G_pred)
                end
            end
            
            # Stack df and G in identical row order
            df = isempty(df_parts) ? DataFrame(term=String[], estimate=Float64[], se=Float64[], level_from=String[], level_to=String[]) : reduce(vcat, df_parts)
            G_all = isempty(G_parts) ? Matrix{Float64}(undef, 0, length(engine.β)) : vcat(G_parts...)
        end
    end
    
    # Add confidence intervals (update column names)
    dof = _try_dof_residual(model)
    groupcols = Symbol[]
    if over !== nothing
        append!(groupcols, over isa Symbol ? [over] : collect(over))
    end
    # Update _add_ci! for new column names (estimate instead of dydx)
    if !isempty(df) && :se in names(df) && :estimate in names(df)
        df[!, :z] = df.estimate ./ df.se
        if dof === nothing
            # Normal
            df[!, :p] = 2 .* (1 .- cdf.(Distributions.Normal(), abs.(df.z)))
            α = 1 - ci_level
            q = Distributions.quantile(Distributions.Normal(), 1 - α/2)
        else
            # t distribution  
            tdist = Distributions.TDist(Float64(dof))
            df[!, :p] = 2 .* (1 .- cdf.(tdist, abs.(df.z)))
            α = 1 - ci_level
            q = Distributions.quantile(tdist, 1 - α/2)
        end
        df[!, :ci_lo] = df.estimate .- q .* df.se
        df[!, :ci_hi] = df.estimate .+ q .* df.se
    end
    
    # Build result metadata
    target_used = if type === :effects 
        target 
    else 
        scale === :response ? :mu : :eta  # For predictions, target is determined by scale
    end
    
    md = (; 
        mode = type === :effects ? :effects : :predictions,
        dydx = vars, 
        target = target_used, 
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
    
    # Use new _new_result signature
    βnames = Symbol.(StatsModels.coefnames(model))  # Convert to symbols
    computation_type = :population
    
    return _new_result(df, G_all, βnames, computation_type, target_used, backend; md...)
end