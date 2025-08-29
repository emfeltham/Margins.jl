# predictions.jl

"""
    _predict_eta!(ηbuf, compiled, data_like, row, β)

Compute η = Xβ for a row into ηbuf (X_row), return scalar η.
"""
function _predict_eta!(xbuf::Vector{Float64}, compiled, data_like, row::Int, β::AbstractVector)
    compiled(xbuf, data_like, row)
    return dot(β, xbuf)
end

"""
    _ape(ap, se; target, engine)

Average predictions across rows; returns (value, se).
"""
function _ape(model, data_nt, compiled, β, Σ; target::Symbol=:mu, link=_auto_link(model), rows=:all, weights=nothing)
    idxs = rows === :all ? (1:_nrows(data_nt)) : rows
    n = length(idxs)
    xbuf = Vector{Float64}(undef, length(compiled))
    acc_val = 0.0
    acc_gβ = zeros(Float64, length(compiled))
    w = _resolve_weights(weights, data_nt, idxs)
    for (j, row) in enumerate(idxs)
        η = _predict_eta!(xbuf, compiled, data_nt, row, β)
        if target === :mu
            μ = GLM.linkinv(link, η)
            if w === nothing
                acc_val += μ
                acc_gβ .+= _dmu_deta_local(link, η) .* xbuf
            else
                acc_val += w[j] * μ
                acc_gβ .+= w[j] .* (_dmu_deta_local(link, η) .* xbuf)
            end
        else
            if w === nothing
                acc_val += η
                acc_gβ .+= xbuf
            else
                acc_val += w[j] * η
                acc_gβ .+= w[j] .* xbuf
            end
        end
    end
    val = w === nothing ? (acc_val / n) : acc_val
    gβ = w === nothing ? (acc_gβ / n) : acc_gβ
    se = FormulaCompiler.delta_method_se(gβ, Σ)
    return val, se
end

"""
    _ap_profiles(model, data_nt, compiled, β, Σ, profiles; target)

Compute adjusted predictions at each profile dict.
"""
function _ap_profiles(model, data_nt, compiled, β, Σ, profiles::Vector{<:Dict}; target::Symbol=:mu, link=_auto_link(model), average_profiles::Bool=false)
    xbuf = Vector{Float64}(undef, length(compiled))
    
    # Pre-allocate all columns including profile columns to avoid column count mismatch
    all_profile_keys = Set{Symbol}()
    for prof in profiles
        for k in keys(prof)
            push!(all_profile_keys, Symbol("at_", k))
        end
    end
    
    # Initialize DataFrame with all required columns
    out = DataFrame(dydx=Float64[], se=Float64[])
    for col_name in all_profile_keys
        out[!, col_name] = Union{Missing,Any}[]
    end
    
    if average_profiles
        acc_val = 0.0
        acc_gβ = zeros(Float64, length(compiled))
        n = 0
        for prof in profiles
            processed_prof = _process_profile_for_scenario(prof, data_nt)
            scen = FormulaCompiler.create_scenario("profile", data_nt, processed_prof)
            η = _predict_eta!(xbuf, compiled, scen.data, 1, β)
            if target === :mu
                μ = GLM.linkinv(link, η)
                acc_val += μ
                acc_gβ .+= _dmu_deta_local(link, η) .* xbuf
            else
                acc_val += η
                acc_gβ .+= xbuf
            end
            n += 1
        end
        val = n > 0 ? acc_val / n : acc_val
        gβ = n > 0 ? acc_gβ / n : acc_gβ
        se = FormulaCompiler.delta_method_se(gβ, Σ)
        out = DataFrame(dydx=[val], se=[se])
        return out
    end
    for prof in profiles
        processed_prof = _process_profile_for_scenario(prof, data_nt)
        scen = FormulaCompiler.create_scenario("profile", data_nt, processed_prof)
        η = _predict_eta!(xbuf, compiled, scen.data, 1, β)  # single-row synthetic; value drawn from profile
        # Create row with profile columns
        if target === :mu
            μ = GLM.linkinv(link, η)
            gβ = _dmu_deta_local(link, η) .* xbuf
            se = FormulaCompiler.delta_method_se(gβ, Σ)
            row_data = Dict{Symbol,Any}(:dydx => μ, :se => se)
        else
            gβ = xbuf
            se = FormulaCompiler.delta_method_se(gβ, Σ)
            row_data = Dict{Symbol,Any}(:dydx => η, :se => se)
        end
        
        # Add profile columns to row
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
    return out
end
