"""
    margins(model, data; kwargs...) -> MarginsResult

Stata-like interface for marginal effects and adjusted predictions.
"""
function margins(model, data; mode::Symbol=:effects, dydx=:continuous, target::Symbol=:mu,
                 at=:none, over=nothing, backend::Symbol=:ad, link=nothing, vce::Symbol=:delta,
                 rows=:all, contrasts::Symbol=:pairwise, levels=:all, by=nothing,
                 weights=nothing, measure::Symbol=:effect, ci_level::Real=0.95)
    engine = _build_engine(model, data, dydx, target)
    data_nt = engine.data_nt
    # Grouping: build groups if over specified and computation is row-based (at=:none)
    groups = _build_groups(data_nt, over)
    if mode === :effects
        # Split vars into continuous and categorical
        cont_vars = [v for v in engine.vars if !_is_categorical(data_nt, v)]
        cat_vars = [v for v in engine.vars if _is_categorical(data_nt, v)]
        df_parts = DataFrame[]
        if groups === nothing || at !== :none
            # No grouping or profile-based effects: compute once
            if !isempty(cont_vars)
                eng_cont = (; engine..., vars=cont_vars)
                if at === :none
                    push!(df_parts, _ame_continuous(model, data_nt, eng_cont; target=target, backend=backend, rows=rows, measure=measure))
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
            for (labels, idxs) in groups
                if !isempty(cont_vars)
                    eng_cont = (; engine..., vars=cont_vars)
                    push!(all_dfs, _ame_continuous(model, data_nt, eng_cont; target=target, backend=backend, rows=idxs, measure=measure))
                end
                if !isempty(cat_vars)
                    eng_cat = (; engine..., vars=cat_vars)
                    push!(all_dfs, _categorical_effects(model, data_nt, eng_cat; target=target, contrasts=contrasts, rows=idxs, at=:none))
                end
                # append group columns
                if !isempty(all_dfs)
                    df_last = last(all_dfs)
                    for (k, v) in pairs(labels)
                        df_last[!, k] = fill(v, nrow(df_last))
                    end
                end
            end
            df = isempty(all_dfs) ? DataFrame(term=Symbol[], dydx=Float64[], se=Float64[]) : reduce(vcat, all_dfs)
        end
    elseif mode === :predictions
        if at === :none
            if groups === nothing
                val, se = _ape(model, data_nt, engine.compiled, engine.β, engine.Σ; target=target, link=engine.link)
                df = DataFrame(term=[:prediction], dydx=[val], se=[se])
            else
                all_dfs = DataFrame[]
                for (labels, idxs) in groups
                    # APE by group: compute average predictions using subgroup rows
                    # Implement subgroup APE by temporarily subsetting rows via weights (simple approach: recompute mean over idxs)
                    n = length(idxs)
                    xbuf = Vector{Float64}(undef, length(engine.compiled))
                    acc_val = 0.0
                    acc_gβ = zeros(Float64, length(engine.compiled))
                    for row in idxs
                        η = _predict_eta!(xbuf, engine.compiled, data_nt, row, engine.β)
                        if target === :mu
                            μ = GLM.linkinv(engine.link, η)
                            acc_val += μ
                            acc_gβ .+= _dmu_deta_local(engine.link, η) .* xbuf
                        else
                            acc_val += η
                            acc_gβ .+= xbuf
                        end
                    end
                    val = acc_val / n
                    gβ = acc_gβ / n
                    se = FormulaCompiler.delta_method_se(gβ, engine.Σ)
                    df_g = DataFrame(term=[:prediction], dydx=[val], se=[se])
                    for (k, v) in pairs(labels)
                        df_g[!, k] = fill(v, nrow(df_g))
                    end
                    push!(all_dfs, df_g)
                end
                df = reduce(vcat, all_dfs)
            end
        else
            profs = _build_profiles(at, data_nt)
            df = _ap_profiles(model, data_nt, engine.compiled, engine.β, engine.Σ, profs; target=target, link=engine.link)
            df[!, :term] = fill(:prediction, nrow(df))
        end
    else
        error("Unsupported mode: $mode")
    end
    _add_ci!(df; level=ci_level)
    md = (; mode, dydx, target, at, backend, rows, contrasts, levels, by, vce, measure,
           n=_nrows(data_nt), link=string(typeof(engine.link)))
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
    _build_groups(data_nt, over)

Build grouping index from over specification. Returns nothing if over is nothing.
"""
function _build_groups(data_nt::NamedTuple, over)
    if over === nothing
        return nothing
    end
    vars = over isa Symbol ? [over] : collect(over)
    # Build mapping from group label -> row indices
    groups = Dict{NamedTuple, Vector{Int}}()
    n = _nrows(data_nt)
    for i in 1:n
        key = NamedTuple{Tuple(vars)}(Tuple(getproperty(data_nt, v)[i] for v in vars))
        push!(get!(groups, key, Int[]), i)
    end
    return collect(groups)
end
