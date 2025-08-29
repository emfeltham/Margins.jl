# compute_continuous.jl

"""
    _ame_continuous(model, data_nt, engine; target=:mu, backend=:ad, rows=:all, measure=:effect)

Compute AME for continuous vars with delta-method SEs.
"""
function _ame_continuous(model, data_nt, engine; target::Symbol=:mu, backend::Symbol=:ad, rows=:all, measure::Symbol=:effect, weights=nothing)
    (; compiled, de, vars, β, Σ, link) = engine
    n = rows === :all ? _nrows(data_nt) : length(rows)
    idxs = rows === :all ? (1:_nrows(data_nt)) : rows
    # Buffers
    gη = Vector{Float64}(undef, length(vars))
    gβ = Vector{Float64}(undef, length(compiled))
    gβ_sum = Vector{Float64}(undef, length(compiled))
    out = DataFrame(term=Symbol[], dydx=Float64[], se=Float64[])
    # For μ target, we still compute gradient wrt β via accumulate_ame_gradient! (μ chain rule inside)
    w = _resolve_weights(weights, data_nt, idxs)
    for var in vars
        fill!(gβ_sum, 0.0)
        acc_val = 0.0
        if w === nothing
            if target === :eta
                FormulaCompiler.accumulate_ame_gradient!(gβ_sum, de, β, idxs, var; link=GLM.IdentityLink(), backend=backend)
            else
                FormulaCompiler.accumulate_ame_gradient!(gβ_sum, de, β, idxs, var; link=link, backend=backend)
            end
            # AME value via averaging
            g_row = Vector{Float64}(undef, length(vars))
            for row in idxs
                if target === :eta
                    FormulaCompiler.marginal_effects_eta!(g_row, de, β, row; backend=backend)
                else
                    FormulaCompiler.marginal_effects_mu!(g_row, de, β, row; link=link, backend=backend)
                end
                acc_val += g_row[findfirst(==(var), vars)]
            end
            ame_val = acc_val / n
            se = FormulaCompiler.delta_method_se(gβ_sum ./ n, Σ)
        else
            # Weighted average: accumulate per-row gradient and value with weights
            gβ_temp = Vector{Float64}(undef, length(compiled))
            g_row = Vector{Float64}(undef, length(vars))
            for (j, row) in enumerate(idxs)
                if target === :eta
                    FormulaCompiler.me_eta_grad_beta!(gβ_temp, de, β, row, var)
                    FormulaCompiler.marginal_effects_eta!(g_row, de, β, row; backend=backend)
                else
                    FormulaCompiler.me_mu_grad_beta!(gβ_temp, de, β, row, var; link=link)
                    FormulaCompiler.marginal_effects_mu!(g_row, de, β, row; link=link, backend=backend)
                end
                gβ_sum .+= w[j] .* gβ_temp
                acc_val += w[j] .* g_row[findfirst(==(var), vars)]
            end
            ame_val = acc_val
            se = FormulaCompiler.delta_method_se(gβ_sum, Σ)
        end
        val = ame_val
        # Elasticity & semi-elasticities transformations
        if measure != :effect
            # Need average y and average x (rough, but standard):
            # y: average η or μ across rows; x: average x across rows
            y_acc = 0.0
            x_acc = 0.0
            xcol = getproperty(data_nt, var)
            for row in idxs
                x_acc += float(xcol[row])
                if target === :eta
                    xbuf = Vector{Float64}(undef, length(compiled))
                    η = _predict_eta!(xbuf, compiled, data_nt, row, β)
                    y_acc += η
                else
                    xbuf = Vector{Float64}(undef, length(compiled))
                    η = _predict_eta!(xbuf, compiled, data_nt, row, β)
                    y_acc += GLM.linkinv(link, η)
                end
            end
            x̄ = x_acc / n
            ȳ = y_acc / n
            if measure === :elasticity
                val = (x̄ / ȳ) * ame_val
            elseif measure === :semielasticity_x
                val = x̄ * ame_val
            elseif measure === :semielasticity_y
                val = (1 / ȳ) * ame_val
            end
        end
        push!(out, (term=var, dydx=val, se=se))
    end
    return out
end

"""
    _mem_mer_continuous(model, data_nt, engine, at; target=:mu, backend=:ad, measure=:effect)

Compute MEM (at=:means) or MER (profiles Dict/Vector{Dict}) for continuous vars.
"""
function _mem_mer_continuous(model, data_nt, engine, at; target::Symbol=:mu, backend::Symbol=:ad, measure::Symbol=:effect)
    (; compiled, de, vars, β, Σ, link) = engine
    profiles = _build_profiles(at, data_nt)
    out = DataFrame(term=Symbol[], dydx=Float64[], se=Float64[])
    
    # Pre-allocate all profile columns to avoid column count mismatch
    all_profile_keys = Set{Symbol}()
    for prof in profiles
        for k in keys(prof)
            push!(all_profile_keys, Symbol("at_", k))
        end
    end
    for col_name in all_profile_keys
        out[!, col_name] = Union{Missing,Any}[]
    end
    
    gβ = Vector{Float64}(undef, length(compiled))
    # Use row=1 on scenario with overrides to emulate profile evaluation
    for var in vars
        for prof in profiles
            processed_prof = _process_profile_for_scenario(prof, data_nt)
            scen = FormulaCompiler.create_scenario("profile", data_nt, processed_prof)
            # Value
            g_row = Vector{Float64}(undef, length(vars))
            if target === :eta
                FormulaCompiler.marginal_effects_eta!(g_row, de, β, 1; backend=backend)
                FormulaCompiler.me_eta_grad_beta!(gβ, de, β, 1, var)
            else
                FormulaCompiler.marginal_effects_mu!(g_row, de, β, 1; link=link, backend=backend)
                FormulaCompiler.me_mu_grad_beta!(gβ, de, β, 1, var; link=link)
            end
            val = g_row[findfirst(==(var), vars)]
            se = FormulaCompiler.delta_method_se(gβ, Σ)
            # Elasticities
            if measure != :effect
                xcol = getproperty(scen.data, var)
                x̄ = float(xcol[1])
                xbuf = Vector{Float64}(undef, length(compiled))
                η = _predict_eta!(xbuf, compiled, scen.data, 1, β)
                ȳ = target === :mu ? GLM.linkinv(link, η) : η
                if measure === :elasticity
                    val = (x̄ / ȳ) * val
                elseif measure === :semielasticity_x
                    val = x̄ * val
                elseif measure === :semielasticity_y
                    val = (1 / ȳ) * val
                end
            end
            # Create row with profile columns
            row_data = Dict{Symbol,Any}(:term => var, :dydx => val, :se => se)
            for (k,v) in prof
                col_name = Symbol("at_", k)
                row_data[col_name] = v
            end
            # Fill missing profile columns with missing
            for col_name in all_profile_keys
                if !haskey(row_data, col_name)
                    row_data[col_name] = missing
                end
            end
            push!(out, row_data)
        end
    end
    return out
end

"""
    _mem_mer_continuous_from_profiles(model, data_nt, engine, profiles; target=:mu, backend=:ad)

Compute continuous effects from pre-built profiles (bypassing _build_profiles step).
Used by the table-based profile_margins dispatch.
"""
function _mem_mer_continuous_from_profiles(model, data_nt, engine, profiles; target::Symbol=:mu, backend::Symbol=:ad)
    (; compiled, de, vars, β, Σ, link) = engine
    out = DataFrame(term=Symbol[], dydx=Float64[], se=Float64[])
    
    # Pre-allocate all profile columns to avoid column count mismatch
    all_profile_keys = Set{Symbol}()
    for prof in profiles
        for k in keys(prof)
            push!(all_profile_keys, Symbol("at_", k))
        end
    end
    for col_name in all_profile_keys
        out[!, col_name] = Union{Missing,Any}[]
    end
    
    gβ = Vector{Float64}(undef, length(compiled))
    # Use row=1 on scenario with overrides to emulate profile evaluation
    for v in vars
        for prof in profiles
            processed_prof = _process_profile_for_scenario(prof, data_nt)
            scen = FormulaCompiler.create_scenario("profile", data_nt, processed_prof)
            # Value
            g_row = Vector{Float64}(undef, length(vars))
            if target === :eta
                FormulaCompiler.marginal_effects_eta!(g_row, de, β, 1; backend=backend)
                FormulaCompiler.me_eta_grad_beta!(gβ, de, β, 1, v)
            else
                FormulaCompiler.marginal_effects_mu!(g_row, de, β, 1; link=link, backend=backend)
                FormulaCompiler.me_mu_grad_beta!(gβ, de, β, 1, v; link=link)
            end
            val = g_row[findfirst(==(v), vars)]
            se = FormulaCompiler.delta_method_se(gβ, Σ)
            
            # Create row data
            row_data = Dict{Symbol,Any}()
            row_data[:term] = v
            row_data[:dydx] = val
            row_data[:se] = se
            
            # Add profile information
            for (k, pv) in pairs(prof)
                row_data[Symbol("at_", k)] = pv
            end
            
            # Fill missing profile columns with missing
            for col_name in all_profile_keys
                if !haskey(row_data, col_name)
                    row_data[col_name] = missing
                end
            end
            
            push!(out, row_data)
        end
    end
    
    return out
end
