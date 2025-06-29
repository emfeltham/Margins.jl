# ame_representation.jl

###############################################################################
# 3. MER helper: marginal effects at representative values
###############################################################################

"""
_ame_representation(
    df::DataFrame,
    model,
    focal::Symbol,
    repvals::Dict{Symbol, Vector},
    fe_form,
    β::AbstractVector,
    invlink,
    dinvlink,
    d2invlink,
    vcov::Function
) -> (Dict{Tuple,Float64}, Dict{Tuple,Float64}, Dict{Tuple,Vector{Float64}})

Compute the marginal effect of `focal` at combinations of representative values
specified in `repvals` (a dict from variable ⇒ vector of values to hold that var at).
Returns three dicts mapping each combo (as a tuple of repvals in key order) to:
- `AME`
- `Std.Err`
- `Δ-method gradient`
"""
function _ame_representation(df::DataFrame, model, focal::Symbol, repvals::AbstractDict,
 fe_form, β, Σβ,
                             invlink, dinvlink, d2invlink)
    repvars = collect(keys(repvals))
    combos  = collect(Iterators.product((repvals[r] for r in repvars)...))
    ame_d, se_d, g_d = Dict{Tuple,Float64}(), Dict{Tuple,Float64}(), Dict{Tuple,Vector{Float64}}()
    tbl0 = Tables.columntable(df)
    for combo in combos
        tbl2 = tbl0
        for (rv,val) in zip(repvars, combo)
            tbl2 = merge(tbl2, (rv => fill(val, nrow(df)),))
        end
        X = modelmatrix(fe_form, tbl2)
        if eltype(df[!,focal]) <: Number
            # continuous focal: use dual-injection on this focal at rep settings
            df_tmp = copy(df)
            for (rv,val) in zip(repvars, combo)
                df_tmp[!, rv] .= val
            end
            # build design+derivatives for the single focal var
            X_rep, Xdx_rep = build_continuous_design(df_tmp, fe_form, [focal])
            ame, se, grad = _ame_continuous(
                β, Σβ,
                X_rep, Xdx_rep[1],
                dinvlink, d2invlink
            )
        else
            ame,se,grad = _ame_factor_baseline(tbl0, fe_form, β, Σβ, focal, invlink, dinvlink)
        end
        ame_d[combo], se_d[combo], g_d[combo] = ame, se, grad
    end
    return ame_d, se_d, g_d
end
