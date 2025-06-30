# core.jl

###############################################################################
# 5. Top-level margins wrapper
###############################################################################
"""
    margins(
      model,
      vars::Union{Symbol, AbstractVector{Symbol}},
      df::AbstractDataFrame;
      vcov::Function = StatsBase.vcov,
      repvals::Dict{Symbol,AbstractVector} = Dict{Symbol,AbstractVector}(),
      pairs::Symbol = :allpairs,
      type::Symbol = :dydx    # :dydx for AMEs, :predict for average predictions
    ) -> MarginsResult{type}

Compute either average marginal effects (AME) or average predicted responses
for one or more predictors.

# Arguments
- `model`  : a fitted GLM/GLMM/LinearModel/MixedModel
- `vars`   : either a single `Symbol` or a `Vector{Symbol}` of predictors
- `df`     : the DataFrame used to fit the model
- `vcov`   : function to extract the fixed-effects covariance matrix
              (defaults to `StatsBase.vcov`)
- `repvals`: a `Dict{Symbol, AbstractVector}` of representative values
- `pairs`  : for categorical AMEs, either `:allpairs` or `:baseline`
- `type`   : `:dydx` (default) for AMEs or `:predict` for predictions

# Returns
A `MarginsResult{type}` containing estimates, standard errors, and Δ-method gradients.
"""
function margins(
    model, vars, df::AbstractDataFrame;
    vcov::Function                       = StatsBase.vcov,
    repvals::AbstractDict{Symbol,<:AbstractVector} = Dict{Symbol,Vector{Float64}}(),
    pairs::Symbol                        = :allpairs,
    type::Symbol                         = :dydx
)
    type ∈ (:dydx, :predict) || throw(ArgumentError("`type` must be :dydx or :predict, got `$type`"))

    # ------------------------------------------------ shared setup ------------------------------------------------
    varlist = isa(vars,Symbol) ? [vars] : collect(vars)
    invlink, dinvlink, d2invlink = link_functions(model)
    fe_form = fixed_effects_form(model)
    β, Σβ   = coef(model), vcov(model)
    n       = nrow(df)
    tbl0    = Tables.columntable(df)

    # result containers (scalar OR dict per predictor)
    result_map = Dict{Symbol,Union{Float64,Dict{Tuple,Float64}}}()
    se_map     = Dict{Symbol,Union{Float64,Dict{Tuple,Float64}}}()
    grad_map   = Dict{Symbol,Union{Vector{Float64},Dict{Tuple,Vector{Float64}}}}()

    # helper: ensure dict containers when repvals present ------------------------------------------
    function ensure_dict!(v)
        result_map[v] isa Dict || (result_map[v] = Dict{Tuple,Float64}())
        se_map[v] isa Dict || (se_map[v] = Dict{Tuple,Float64}())
        grad_map[v] isa Dict || (grad_map[v] = Dict{Tuple,Vector{Float64}}())
    end

    # ------------------------------------------------ :predict branch ------------------------------------------------
    if type == :predict
        if isempty(repvals)
            # overall prediction – scalar per var (unchanged)
            X   = modelmatrix(fe_form, df)
            η   = X * β;  μ = invlink.(η)
            pred = mean(μ)
            μp   = dinvlink.(η)
            grad = (X' * μp) ./ n
            se   = sqrt(dot(grad, Σβ * grad))
            for v in varlist
                result_map[v] = pred
                se_map[v] = se
                grad_map[v] = grad
            end
        else
            # prediction at rep-value combinations (optimized in-place + buffer reuse)
            repvars = collect(keys(repvals))
            combos  = collect(Iterators.product((repvals[r] for r in repvars)...))

            # prepare a mutable copy of df with rewritable repvar columns
            workdf = DataFrame(df)
            for rv in repvars
                workdf[!, rv] = copy(df[!, rv])
            end

            # pre-allocate buffers using initial design matrix to get dimensions
            X0 = modelmatrix(fe_form, workdf)
            n, p = size(X0)
            η    = Vector{Float64}(undef, n)
            μ    = Vector{Float64}(undef, n)
            μp   = Vector{Float64}(undef, n)
            grad = Vector{Float64}(undef, p)

            for combo in combos
                # overwrite repvar columns in-place
                for (rv, val) in zip(repvars, combo)
                    fill!(workdf[!, rv], val)
                end

                # rebuild design matrix
                X = modelmatrix(fe_form, workdf)
                mul!(η, X, β)  # η = X*β

                # compute μ and μ' in one @simd loop
                @inbounds @simd for i in 1:n
                    μ[i]  = invlink(η[i])
                    μp[i] = dinvlink(η[i])
                end
                pred = sum(μ) / n

                # gradient = (X' * μp) / n
                mul!(grad, X', μp)
                grad ./= n
                se = sqrt(dot(grad, Σβ * grad))

                # store results
                for v in varlist
                    ensure_dict!(v)
                    result_map[v][combo] = pred
                    se_map[v][combo]     = se
                    grad_map[v][combo]   = grad
                end
            end
        end

    # ------------------------------------------------ :dydx branch --------------------------------------------------
    else  # :dydx
        cts_vars = filter(v->eltype(df[!,v])<:Real && eltype(df[!,v])!=Bool, varlist)
        cat_vars = setdiff(varlist, cts_vars)

        if isempty(repvals)
            # continuous AMEs
            X, Xdx = build_continuous_design(df, fe_form, cts_vars)
            for (j, v) in enumerate(cts_vars)
                ame, se, grad = _ame_continuous(β, Σβ, X, Xdx[j], dinvlink, d2invlink)
                result_map[v] = ame
                se_map[v] = se
                grad_map[v] = grad
            end
            # categorical AMEs
            for v in cat_vars
                ame_d, se_d, g_d = pairs == :baseline ?
                    _ame_factor_baseline(tbl0, fe_form, β, Σβ, v, invlink, dinvlink) :
                    _ame_factor_allpairs(tbl0, fe_form, β, Σβ, v, invlink, dinvlink)
                result_map[v] = ame_d
                se_map[v] = se_d
                grad_map[v] = g_d
            end
        else
            # AMEs at rep‑values (all dicts)
            for v in varlist
                ame_d, se_d, g_d = _ame_representation(
                    df, model, v, repvals,
                    fe_form, β, Σβ,
                    invlink, dinvlink, d2invlink
                )
                result_map[v] = ame_d
                se_map[v]    = se_d
                grad_map[v]  = g_d
            end
        end
    end

    return MarginsResult{type}(
        varlist, repvals, result_map, se_map, grad_map,
        n, dof_residual(model),
        string(family(model).dist),
        string(family(model).link)
    )
end
