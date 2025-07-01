###############################################################################
# 4. Categorical AMEs (Bool & CategoricalArray) – allocation-free
###############################################################################

# --- helper: Σβ * g for either a Matrix or a Cholesky ------------------------
_cov_mul(Σ::AbstractMatrix, g) = Σ * g
_cov_mul(C::LinearAlgebra.Cholesky, g) = Matrix(C) * g   # safe, still cheap

function _factor_design(tbl0, fe_form, f::Symbol, lvl)
    col_raw = tbl0[f]
    newcol  = if col_raw isa AbstractVector{Bool}
        fill(lvl, length(col_raw))                    # Vector{Bool}
    elseif col_raw isa CategoricalVector
        categorical(fill(lvl, length(col_raw)),
                    levels = levels(col_raw))
    else
        cat0 = categorical(col_raw)
        categorical(fill(lvl, length(cat0)),
                    levels = levels(cat0))
    end
    tbl2 = merge(tbl0, (f => newcol,))
    return modelmatrix(fe_form, tbl2)
end

function _ame_factor_pair_matrix(
    tbl0, fe_form, β, Σβ_or_chol,
    f::Symbol, lvl_j, lvl_k,
    invlink, dinvlink,
)
    Xj = _factor_design(tbl0, fe_form, f, lvl_j)
    Xk = _factor_design(tbl0, fe_form, f, lvl_k)

    ηj, ηk   = Xj*β, Xk*β
    μj, μk   = invlink.(ηj), invlink.(ηk)
    μpj, μpk = dinvlink.(ηj), dinvlink.(ηk)

    n    = length(ηj)
    ame  = mean(μk .- μj)
    grad = (Xk' * μpk .- Xj' * μpj) ./ n
    se   = sqrt(dot(grad, _cov_mul(Σβ_or_chol, grad)))
    return ame, se, grad
end

function _ame_factor_baseline(
    tbl0, fe_form, β, Σβ_or_chol,
    f::Symbol, invlink, dinvlink,
)
    lvls = levels(categorical(tbl0[f]))
    base = lvls[1]

    ame_d  = Dict{Tuple,Float64}()
    se_d   = Dict{Tuple,Float64}()
    grad_d = Dict{Tuple,Vector{Float64}}()

    for lvl in lvls[2:end]
        ame,se,grad = _ame_factor_pair_matrix(
            tbl0, fe_form, β, Σβ_or_chol,
            f, base, lvl, invlink, dinvlink)
        ame_d[(lvl,)], se_d[(lvl,)], grad_d[(lvl,)] = ame, se, grad
    end
    return ame_d, se_d, grad_d
end

function _ame_factor_allpairs(
    tbl0, fe_form, β, Σβ_or_chol,
    f::Symbol, invlink, dinvlink,
)
    lvls = levels(categorical(tbl0[f]))

    ame_d  = Dict{Tuple,Float64}()
    se_d   = Dict{Tuple,Float64}()
    grad_d = Dict{Tuple,Vector{Float64}}()

    for i in 1:length(lvls)-1, j in i+1:length(lvls)
        ame,se,grad = _ame_factor_pair_matrix(
            tbl0, fe_form, β, Σβ_or_chol,
            f, lvls[i], lvls[j], invlink, dinvlink)
        key = (lvls[i], lvls[j])
        ame_d[key], se_d[key], grad_d[key] = ame, se, grad
    end
    return ame_d, se_d, grad_d
end
