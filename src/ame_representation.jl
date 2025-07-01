###############################################################################
# 3.  AME at representative values  (safe, serial, race-free)
###############################################################################
function _ame_representation(
    df::DataFrame,
    model,
    focal::Symbol,
    repvals::AbstractDict{Symbol,<:AbstractVector},
    fe_form,
    β::Vector{Float64},
    cholΣβ::LinearAlgebra.Cholesky,
    invlink::Function,
    dinvlink::Function,
    d2invlink::Function,
)

    # ---------------------------------------------------------------- rep grid
    repvars = collect(keys(repvals))
    combos  = collect(Iterators.product((repvals[r] for r in repvars)...))

    nr, p = nrow(df), length(β)

    # scratch objects reused each iteration (serial → no races)
    workdf   = DataFrame(df, copycols = true)
    ws       = AMEWorkspace(nr, p)
    X        = Matrix{Float64}(undef, nr, p)
    Xdx      = similar(X)

    ame_dict  = Dict{Tuple,Float64}()
    se_dict   = Dict{Tuple,Float64}()
    grad_dict = Dict{Tuple,Vector{Float64}}()

    for combo in combos
        @inbounds for (rv,val) in zip(repvars, combo)
            fill!(workdf[!, rv], val)
        end

        colT = eltype(workdf[!, focal])

        if colT <: Real && colT != Bool
            # ------------------ continuous focal -----------------------------
            build_continuous_design_single!(workdf, fe_form, focal, X, Xdx)

            ame, se, grad = _ame_continuous!(
                β, cholΣβ, X, Xdx, dinvlink, d2invlink, ws)

            key = Tuple(combo)                    # (rep1,rep2,…)
            ame_dict[key]  = ame
            se_dict[key]   = se
            grad_dict[key] = grad

        elseif colT <: Bool
            # ------------------ Boolean focal → single (false→true) ----------
            tbl = Tables.columntable(workdf)

            ame_b, se_b, grad_b = _ame_factor_baseline(
                tbl, fe_form, β, cholΣβ, focal, invlink, dinvlink)

            pair_key = first(keys(ame_b))         # (false,true) or (true,)
            key      = Tuple(combo)               # rep-values only

            ame_dict[key]  = ame_b[pair_key]
            se_dict[key]   = se_b[pair_key]
            grad_dict[key] = grad_b[pair_key]

        else
            # ------------------ ≥ 3-level categorical focal ------------------
            tbl = Tables.columntable(workdf)

            ame_sub, se_sub, grad_sub = _ame_factor_allpairs(
                tbl, fe_form, β, cholΣβ, focal, invlink, dinvlink)

            repkey = Tuple(combo)
            for pair in keys(ame_sub)             # (lvlᵢ,lvlⱼ)
                fullkey        = (repkey..., pair...)
                ame_dict[fullkey]  = ame_sub[pair]
                se_dict[fullkey]   = se_sub[pair]
                grad_dict[fullkey] = grad_sub[pair]
            end
        end
    end

    return ame_dict, se_dict, grad_dict
end
