###############################################################################
# 4. Categorical AMEs (Bool & CategoricalArray) – allocation-free
###############################################################################

# --- helper: Σβ * g for either a Matrix or a Cholesky ------------------------
_cov_mul(Σ::AbstractMatrix, g) = Σ * g
_cov_mul(C::LinearAlgebra.Cholesky, g) = Matrix(C) * g   # safe, still cheap

# A new, fast version of the pairwise calculator
function _ame_factor_pair_matrix_fast!(
    Xj::Matrix, Xk::Matrix, workdf::DataFrame, # Pre-allocated buffers
    fe_rhs, β, Σβ_or_chol,
    f::Symbol, lvl_i, lvl_j,
    invlink, dinvlink,
)
    # Build Xj in-place
    fill!(workdf[!, f], lvl_i)
    modelmatrix!(Xj, fe_rhs, workdf)

    # Build Xk in-place
    fill!(workdf[!, f], lvl_j)
    modelmatrix!(Xk, fe_rhs, workdf)

    # --- Calculations are the same as before ---
    ηj, ηk   = Xj*β, Xk*β
    μj, μk   = invlink.(ηj), invlink.(ηk)
    μpj, μpk = dinvlink.(ηj), dinvlink.(ηk)

    n    = length(ηj)
    ame  = mean(μk .- μj)
    grad = (Xk' * μpk .- Xj' * μpj) ./ n
    se   = sqrt(dot(grad, _cov_mul(Σβ_or_chol, grad)))
    return ame, se, grad
end

function _ame_factor_baseline!(
    ame_d, se_d, grad_d,
    tbl0, fe_rhs, β, Σβ_or_chol,
    f::Symbol, invlink, dinvlink,
)
    lvls = levels(categorical(tbl0[f]))
    base = lvls[1]

    # --- Set up reusable work objects ---
    workdf = DataFrame(Tables.columntable(tbl0), copycols=true)
    n, p = nrow(workdf), length(β)
    Xj = Matrix{Float64}(undef, n, p) # Buffer for first level's design
    Xk = Matrix{Float64}(undef, n, p) # Buffer for second level's design
    original_col = copy(workdf[!, f]) # Save original column

    # Loop over pairs, reusing workdf and buffers
    # contrast `lvl` with `base`
    for lvl in lvls[2:end]
        # Re-use buffers Xj and Xk by passing them to the pair-wise function
        ame, se, grad = _ame_factor_pair_matrix_fast!(
            Xj, Xk, workdf, fe_rhs, β, Σβ_or_chol,
            f, base, lvl, invlink, dinvlink
        )
        key = (base, lvl)
        ame_d[key], se_d[key], grad_d[key] = ame, se, grad
    end

    # Restore the original column in the work dataframe
    workdf[!, f] = original_col
end

function _ame_factor_allpairs!(
    ame_d, se_d, grad_d,
    tbl0, fe_rhs, β, Σβ_or_chol,
    f::Symbol, invlink, dinvlink,
)
    lvls = levels(categorical(tbl0[f]))

    # --- Set up reusable work objects ---
    workdf = DataFrame(Tables.columntable(tbl0), copycols=true)
    n, p = nrow(workdf), length(β)
    Xj = Matrix{Float64}(undef, n, p) # Buffer for first level's design
    Xk = Matrix{Float64}(undef, n, p) # Buffer for second level's design
    original_col = copy(workdf[!, f]) # Save original column

    # Loop over pairs, reusing workdf and buffers
    for i in 1:length(lvls)-1, j in i+1:length(lvls)
        lvl_i, lvl_j = lvls[i], lvls[j]

        # Re-use buffers Xj and Xk by passing them to the pair-wise function
        ame, se, grad = _ame_factor_pair_matrix_fast!(
            Xj, Xk, workdf, fe_rhs, β, Σβ_or_chol,
            f, lvl_i, lvl_j, invlink, dinvlink
        )
        key = (lvl_i, lvl_j)
        ame_d[key], se_d[key], grad_d[key] = ame, se, grad
    end

    # Restore the original column in the work dataframe
    workdf[!, f] = original_col
end
