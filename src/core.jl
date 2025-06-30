# core.jl

###############################################################################
# 5. Top-level margins wrapper – *zero* design-matrix allocations on hot path
###############################################################################

"""
    margins(model, vars, df; kwargs...)  ->  MarginsResult

Compute either average marginal effects (AMEs, `type = :dydx`) or average
predicted responses (`type = :predict`) for one or more predictors.

Internally we now

1.  Build the model *schema* **once**  
2.  Allocate a single `X_base` (and a work-buffer `X_buf`) **once**  
3.  Re-fill `X_buf` in-place with [`modelmatrix!`](@ref) whenever the
    DataFrame changes.

This removes every `modelmatrix` heap-allocation inside the main loops.
"""
function margins(
    model,
    vars,
    df::AbstractDataFrame;
    vcov::Function                       = StatsBase.vcov,
    repvals::AbstractDict{Symbol,<:AbstractVector} = Dict{Symbol,Vector{Float64}}(),
    pairs::Symbol                        = :allpairs,
    type::Symbol                         = :dydx,
)

    type ∈ (:dydx, :predict) ||
        throw(ArgumentError("`type` must be :dydx or :predict, got `$type`"))

    # ───────────── one-time setup ───────────────────────────────────────────
    varlist   = isa(vars,Symbol) ? [vars] : collect(vars)
    invlink,
    dinvlink,
    d2invlink = link_functions(model)

    fe_form   = fixed_effects_form(model)

    # -- build *once*: base design + ALL ∂X/∂x for continuous vars ----------
    cts_vars  = filter(v->eltype(df[!,v]) <: Real && eltype(df[!,v]) != Bool,
                       varlist)
    X_base, Xdx_list = build_continuous_design(df, fe_form, cts_vars) # <── NEW
    X_buf     = similar(X_base)          # work buffer for predictions

    n, p      = size(X_base)
    β, Σβ     = coef(model), vcov(model)
    cholΣβ    = cholesky(Σβ)

    # rhs term baked-in (for modelmatrix!)
    mf        = StatsModels.ModelFrame(fe_form, df)
    frhs      = mf.f.rhs

    tbl0      = Tables.columntable(df)

    # result containers -----------------------------------------------------
    result_map = Dict{Symbol,Union{Float64,Dict{Tuple,Float64}}}()
    se_map     = Dict{Symbol,Union{Float64,Dict{Tuple,Float64}}}()
    grad_map   = Dict{Symbol,Union{Vector{Float64},Dict{Tuple,Vector{Float64}}}}()

    ensure_dict!(v) = begin
        result_map[v] isa Dict || (result_map[v] = Dict{Tuple,Float64}())
        se_map[v]     isa Dict || (se_map[v]     = Dict{Tuple,Float64}())
        grad_map[v]   isa Dict || (grad_map[v]   = Dict{Tuple,Vector{Float64}}())
    end

    # ─────────────────────────── :predict branch ───────────────────────────
    if type == :predict
        # (unchanged except for X_buf reuse)
        if isempty(repvals)
            η   = X_base * β
            μ   = invlink.(η)
            pred = mean(μ)
            μp  = dinvlink.(η)
            grad = (X_base' * μp) ./ n
            se   = sqrt(dot(grad, Σβ * grad))
            for v in varlist
                result_map[v] = pred;  se_map[v] = se;  grad_map[v] = grad
            end

        else
            repvars = collect(keys(repvals))
            combos  = collect(Iterators.product((repvals[r] for r in repvars)...))

            workdf = DataFrame(df)
            for rv in repvars; workdf[!,rv] = copy(df[!,rv]); end

            η  = Vector{Float64}(undef,n)
            μ  = Vector{Float64}(undef,n)
            μp = Vector{Float64}(undef,n)
            grad = Vector{Float64}(undef,p)

            for combo in combos
                for (rv,val) in zip(repvars, combo)
                    fill!(workdf[!,rv], val)
                end

                modelmatrix!(X_buf, frhs, Tables.columntable(workdf))  # <── fast rebuild

                mul!(η, X_buf, β)
                @inbounds @simd for i in 1:n
                    μ[i]  = invlink(η[i])
                    μp[i] = dinvlink(η[i])
                end
                pred = sum(μ)/n
                mul!(grad, X_buf', μp);  grad ./= n
                se = sqrt(dot(grad, Σβ * grad))

                for v in varlist
                    ensure_dict!(v)
                    result_map[v][combo] = pred
                    se_map[v][combo]     = se
                    grad_map[v][combo]   = grad
                end
            end
        end

    # ─────────────────────────── :dydx branch ──────────────────────────────
    else
        cat_vars = setdiff(varlist, cts_vars)
        ws       = AMEWorkspace(n,p)   # one Δ-method workspace

        if isempty(repvals)
            # —— continuous AMEs (now zero ForwardDiff inside loop) ————
            for (j,v) in enumerate(cts_vars)
                copy!(X_buf, X_base)          # restore main design
                Xdx = Xdx_list[j]             # ∂X/∂v computed once

                ame, se, grad = _ame_continuous!(
                    β, cholΣβ, X_buf, Xdx,
                    dinvlink, d2invlink, ws)

                result_map[v] = ame;  se_map[v] = se;  grad_map[v] = grad
            end

            # —— categorical AMEs (unchanged) ————————————————
            for v in cat_vars
                ame_d, se_d, g_d = pairs == :baseline ?
                    _ame_factor_baseline(tbl0, fe_form, β, Σβ, v,
                                         invlink, dinvlink) :
                    _ame_factor_allpairs(tbl0, fe_form, β, Σβ, v,
                                         invlink, dinvlink)

                result_map[v] = ame_d;  se_map[v] = se_d;  grad_map[v] = g_d
            end

        else
            # —— AMEs at representative values (unchanged) ———————
            for v in varlist
                ame_d, se_d, g_d = _ame_representation(
                    df, model, v, repvals,
                    fe_form, β, cholΣβ,
                    invlink, dinvlink, d2invlink)

                result_map[v] = ame_d;  se_map[v] = se_d;  grad_map[v] = g_d
            end
        end
    end

    return MarginsResult{type}(
        varlist, repvals, result_map, se_map, grad_map,
        n, dof_residual(model),
        string(family(model).dist),
        string(family(model).link),
    )
end
