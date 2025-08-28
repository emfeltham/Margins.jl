# api.jl

"""
    margins(model, data; kwargs...) -> MarginsResult

Stata-like marginal effects for the JuliaStats stack.

Compat:
- `model`: Fits from GLM.jl (TableRegressionModel) or any StatsModels-compatible model.
- `data`: Tables.jl-compatible table (e.g., DataFrame); Margins respects StatsModels design.
- `target`: `:mu` (response) or `:eta` (link), aligned with GLM predict types.
- `scale`: `:auto|:response|:link` (predictions mode) mirrors GLM.predict scales.
- Covariance: defaults to `vcov(model)`; pass `vcov` matrix/function/estimator for robust/cluster/HAC via CovarianceMatrices.jl.

Examples:
    using CovarianceMatrices
    res = ame(m, df; dydx=[:x], vcov = m->vcov(m, HC1()))
"""
function margins(
    model, data; mode::Symbol=:effects, dydx=:continuous, target::Symbol=:mu,
    at=:none, over=nothing, within=nothing, by=nothing, backend::Symbol=:ad,
    rows=:all, contrasts::Symbol=:pairwise, levels=:all,
    weights=nothing, asbalanced::Union{Bool,Vector{Symbol}}=false, measure::Symbol=:effect, ci_level::Real=0.95,
    mcompare::Symbol=:noadjust,
    vcov::Union{Symbol,AbstractMatrix,Function,Any}=:model,
    scale::Symbol=:auto
)
    engine = _build_engine(model, data, dydx, target)
    # Resolve Σ via vcov spec
    Σ_override = _resolve_vcov(vcov, model, length(engine.β))
    engine = (; engine..., Σ=Σ_override)
    data_nt = engine.data_nt
    # Grouping: build groups if over/within specified and computation is row-based (at=:none)
    groups = _build_groups(data_nt, over, within)
    # Stratification: split by strata (compute per-by subgroup)
    strata = _split_by(data_nt, by)
    if mode === :effects
        # Split vars into continuous and categorical
        cont_vars = [v for v in engine.vars if !_is_categorical(data_nt, v)]
        cat_vars = [v for v in engine.vars if _is_categorical(data_nt, v)]
        df_parts = DataFrame[]
        if (groups === nothing && strata === nothing) || at !== :none
            # No grouping or profile-based effects: compute once
            if !isempty(cont_vars)
                eng_cont = (; engine..., vars=cont_vars)
                if at === :none
                    # merge weights with asbalanced if requested
                    ab_subset = asbalanced === true ? nothing : asbalanced === false ? nothing : asbalanced
                    bw = (asbalanced === false) ? nothing : _balanced_weights(data_nt, rows, ab_subset)
                    w_final = _merge_weights(weights, bw, data_nt, rows)
                    push!(df_parts, _ame_continuous(model, data_nt, eng_cont; target=target, backend=backend, rows=rows, measure=measure, weights=w_final))
                else
                    push!(df_parts, _mem_mer_continuous(model, data_nt, eng_cont, at; target=target, backend=backend, measure=measure))
                end
            end
            if !isempty(cat_vars)
                eng_cat = (; engine..., vars=cat_vars)
                push!(df_parts, _categorical_effects(model, data_nt, eng_cat; target=target, contrasts=contrasts, rows=rows, at=at))
            end
            df = isempty(df_parts) ? DataFrame(term=Symbol[], dydx=Float64[], se=Float64[]) : reduce(vcat, df_parts)
        else
            # Grouped, row-based computation
            all_dfs = DataFrame[]
            # If strata specified, outer loop per strata, inner loop groups
            if strata === nothing
                strata = [(NamedTuple(), rows === :all ? collect(1 : _nrows(data_nt)) : rows)]
            end
            for (bylabels, sidxs) in strata
                local_groups = groups === nothing ? [(NamedTuple(), sidxs)] : _build_groups(data_nt, over, within, sidxs)
                for (labels, idxs) in local_groups
                    if !isempty(cont_vars)
                        eng_cont = (; engine..., vars=cont_vars)
                        ab_subset = asbalanced === true ? nothing : asbalanced === false ? nothing : asbalanced
                        bw = (asbalanced === false) ? nothing : _balanced_weights(data_nt, idxs, ab_subset)
                        w_final = _merge_weights(weights, bw, data_nt, idxs)
                        push!(all_dfs, _ame_continuous(model, data_nt, eng_cont; target=target, backend=backend, rows=idxs, measure=measure, weights=w_final))
                    end
                    if !isempty(cat_vars)
                        eng_cat = (; engine..., vars=cat_vars)
                        push!(all_dfs, _categorical_effects(model, data_nt, eng_cat; target=target, contrasts=contrasts, rows=idxs, at=:none))
                    end
                    # append group and by columns
                    if !isempty(all_dfs)
                        df_last = last(all_dfs)
                        for (k, v) in pairs(bylabels)
                            df_last[!, k] = fill(v, nrow(df_last))
                        end
                        for (k, v) in pairs(labels)
                            df_last[!, k] = fill(v, nrow(df_last))
                        end
                    end
                end
            end
            df = isempty(all_dfs) ? DataFrame(term=Symbol[], dydx=Float64[], se=Float64[]) : reduce(vcat, all_dfs)
        end
    elseif mode === :predictions
        if at === :none
            if groups === nothing && strata === nothing
                target_pred = scale === :auto ? target : (scale === :response ? :mu : :eta)
                ab_subset = asbalanced === true ? nothing : asbalanced === false ? nothing : asbalanced
                bw = (asbalanced === false) ? nothing : _balanced_weights(data_nt, rows, ab_subset)
                w_final = _merge_weights(weights, bw, data_nt, rows)
                val, se = _ape(model, data_nt, engine.compiled, engine.β, engine.Σ; target=target_pred, link=engine.link, rows=rows, weights=w_final)
                df = DataFrame(term=[:prediction], dydx=[val], se=[se])
            else
                all_dfs = DataFrame[]
                strata = strata === nothing ? [(NamedTuple(), rows === :all ? collect(1 : _nrows(data_nt)) : rows)] : strata
                for (bylabels, sidxs) in strata
                    local_groups = groups === nothing ? [(NamedTuple(), sidxs)] : _build_groups(data_nt, over, within, sidxs)
                    for (labels, idxs) in local_groups
                        target_pred = scale === :auto ? target : (scale === :response ? :mu : :eta)
                        ab_subset = asbalanced === true ? nothing : asbalanced === false ? nothing : asbalanced
                        bw = (asbalanced === false) ? nothing : _balanced_weights(data_nt, idxs, ab_subset)
                        w_final = _merge_weights(weights, bw, data_nt, idxs)
                        val, se = _ape(model, data_nt, engine.compiled, engine.β, engine.Σ; target=target_pred, link=engine.link, rows=idxs, weights=w_final)
                        df_g = DataFrame(term=[:prediction], dydx=[val], se=[se])
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
        else
            profs = _build_profiles(at, data_nt)
            target_pred = scale === :auto ? target : (scale === :response ? :mu : :eta)
            df = _ap_profiles(model, data_nt, engine.compiled, engine.β, engine.Σ, profs; target=target_pred, link=engine.link)
            df[!, :term] = fill(:prediction, nrow(df))
        end
    else
        error("Unsupported mode: $mode")
    end
    dof = _try_dof_residual(model)
    groupcols = Symbol[]
    if over !== nothing
        append!(groupcols, over isa Symbol ? [over] : collect(over))
    end
    # At-profile columns are prefixed with at_
    append!(groupcols, [c for c in names(df) if startswith(String(c), "at_")])
    _add_ci!(df; level=ci_level, dof=dof, mcompare=mcompare, groupcols=groupcols)
    md = (; mode, dydx, target, at, backend, rows, contrasts, levels, by, measure,
           n = _nrows(data_nt), link=string(typeof(engine.link)), dof, vcov=vcov, scale=scale)
    return _new_result(df; md...)
end

# Convenience wrappers
ame(model, data; kwargs...) = margins(model, data; mode=:effects, at=:none, kwargs...)
mem(model, data; kwargs...) = margins(model, data; mode=:effects, at=:means, kwargs...)
mer(model, data; at, kwargs...) = margins(model, data; mode=:effects, at=at, kwargs...)
ape(model, data; kwargs...) = margins(model, data; mode=:predictions, at=:none, kwargs...)
apm(model, data; kwargs...) = margins(model, data; mode=:predictions, at=:means, kwargs...)
apr(model, data; at, kwargs...) = margins(model, data; mode=:predictions, at=at, kwargs...)

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
