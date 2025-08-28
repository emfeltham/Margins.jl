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
function _ape(model, data_nt, compiled, β, Σ; target::Symbol=:mu, link=_auto_link(model))
    n = _nrows(data_nt)
    xbuf = Vector{Float64}(undef, length(compiled))
    acc_val = 0.0
    acc_gβ = zeros(Float64, length(compiled))
    for row in 1:n
        η = _predict_eta!(xbuf, compiled, data_nt, row, β)
        if target === :mu
            μ = GLM.linkinv(link, η)
            acc_val += μ
            acc_gβ .+= _dmu_deta_local(link, η) .* xbuf
        else
            acc_val += η
            acc_gβ .+= xbuf
        end
    end
    val = acc_val / n
    gβ = acc_gβ / n
    se = FormulaCompiler.delta_method_se(gβ, Σ)
    return val, se
end

"""
    _ap_profiles(model, data_nt, compiled, β, Σ, profiles; target)

Compute adjusted predictions at each profile dict.
"""
function _ap_profiles(model, data_nt, compiled, β, Σ, profiles::Vector{<:Dict}; target::Symbol=:mu, link=_auto_link(model))
    xbuf = Vector{Float64}(undef, length(compiled))
    out = DataFrame()
    for prof in profiles
        scen = FormulaCompiler.create_scenario("profile", data_nt, Dict{Symbol,Any}(prof))
        η = _predict_eta!(xbuf, compiled, scen.data, 1, β)  # single-row synthetic; value drawn from profile
        if target === :mu
            μ = GLM.linkinv(link, η)
            gβ = _dmu_deta_local(link, η) .* xbuf
            se = FormulaCompiler.delta_method_se(gβ, Σ)
            push!(out, (; dydx=μ, se))
        else
            gβ = xbuf
            se = FormulaCompiler.delta_method_se(gβ, Σ)
            push!(out, (; dydx=η, se))
        end
        # attach profile columns
        for (k,v) in prof
            out[!, Symbol("at_", k)] = get(out, Symbol("at_", k), fill(v, nrow(out)))
            out[end, Symbol("at_", k)] = v
        end
    end
    return out
end
