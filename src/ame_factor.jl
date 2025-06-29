# ame_factor.jl

###############################################################################
# 4. Categorical AMEs (Bool & CategoricalArray) streamlined
###############################################################################
# one modelmatrix per level, shallow merge, no row loops

function _factor_design(tbl0, fe_form, f::Symbol, lvl)
    col = tbl0[f]
    cat0 = col isa CategoricalVector ? col : categorical(col)
    newcol = categorical(fill(lvl, length(cat0)), levels=levels(cat0))
    tbl2 = merge(tbl0, (f=>newcol,))
    return modelmatrix(fe_form, tbl2)
end

function _ame_factor_pair_matrix(tbl0, fe_form, β, Σβ,
                                 f::Symbol, lvl_j, lvl_k,
                                 invlink, dinvlink)
    Xj = _factor_design(tbl0, fe_form, f, lvl_j)
    Xk = _factor_design(tbl0, fe_form, f, lvl_k)
    ηj, ηk = Xj*β, Xk*β
    μj, μk = invlink.(ηj), invlink.(ηk)
    μpj, μpk = dinvlink.(ηj), dinvlink.(ηk)
    n = length(ηj)
    ame  = mean(μk .- μj)
    grad = (Xk'*(μpk) .- Xj'*(μpj)) ./ n
    se   = sqrt(dot(grad, Σβ * grad))
    return ame, se, grad
end

function _ame_factor_baseline(tbl0, fe_form, β, Σβ,
                              f::Symbol, invlink, dinvlink)
    lvls = levels(categorical(tbl0[f]))
    base = lvls[1]
    ame_d, se_d, g_d = Dict{Tuple,Float64}(), Dict{Tuple,Float64}(), Dict{Tuple,Vector{Float64}}()
    for lvl in lvls[2:end]
        ame,se,grad = _ame_factor_pair_matrix(tbl0, fe_form, β, Σβ,
                                              f, base, lvl,
                                              invlink, dinvlink)
        ame_d[(lvl,)], se_d[(lvl,)], g_d[(lvl,)] = ame, se, grad
    end
    return ame_d, se_d, g_d
end

function _ame_factor_allpairs(tbl0, fe_form, β, Σβ,
                               f::Symbol, invlink, dinvlink)
    lvls = levels(categorical(tbl0[f]))
    ame_d, se_d, g_d = Dict{Tuple,Float64}(), Dict{Tuple,Float64}(), Dict{Tuple,Vector{Float64}}()
    for i in 1:length(lvls)-1, j in i+1:length(lvls)
        ame,se,grad = _ame_factor_pair_matrix(tbl0, fe_form, β, Σβ,
                                              f, lvls[i], lvls[j],
                                              invlink, dinvlink)
        ame_d[(lvls[i],lvls[j])], se_d[(lvls[i],lvls[j])], g_d[(lvls[i],lvls[j])] = ame, se, grad
    end
    return ame_d, se_d, g_d
end
