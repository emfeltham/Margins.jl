###############################################################################
# 5. Top-level margins wrapper
###############################################################################
"""
    margins(
      model,
      vars::Union{Symbol, AbstractVector{Symbol}},
      df::AbstractDataFrame;
      vcov::Function = StatsBase.vcov,
      repvals::Dict{Symbol, Vector} = Dict{Symbol, Vector}()
    ) -> MarginsResult

Compute average marginal effects (AME) for one or more predictors.

# Arguments
- `model` : a fitted GLM/GLMM/LinearModel/MixedModel
- `vars`  : either a single `Symbol` or a `Vector{Symbol}` of predictors
- `df`    : the DataFrame used to fit the model (required)
- `vcov`  : function to extract the fixed-effects covariance matrix
             (defaults to `StatsBase.vcov`)
- `repvals`: a `Dict{Symbol, Vector}` of representative values for MERs

# Returns
An `MarginsResult` containing AMEs, standard errors, and gradients for each var.
"""
function margins(model, vars, df::AbstractDataFrame;
                 vcov=StatsBase.vcov, repvals=Dict(), pairs::Symbol=:allpairs)

    varlist = isa(vars,Symbol) ? [vars] : collect(vars)
    invlink, dinvlink, d2invlink = link_functions(model)
    fe_form = fixed_effects_form(model)
    β, Σβ   = coef(model), vcov(model)
    n       = nrow(df)
    tbl0    = Tables.columntable(df)

    # Changed here: Bool now treated as categorical
    # continuous = Real but not Bool, categorical = rest
    cts_vars = filter(v -> eltype(df[!,v]) <: Real && eltype(df[!,v]) != Bool, varlist)
    cat_vars = setdiff(varlist, cts_vars)

    ame_map, se_map, grad_map = Dict(), Dict(), Dict()
    
    # continuous
    X, Xdx = build_continuous_design(df, fe_form, cts_vars)
    for (j,v) in enumerate(cts_vars)
        ame,se,grad = _ame_continuous(β, Σβ, X, Xdx[j], dinvlink, d2invlink)
        ame_map[v], se_map[v], grad_map[v] = ame,se,grad
    end

    # MERs
    if !isempty(repvals)
        for v in varlist
            ame,se,grad = _ame_representation(df, model, v, repvals,
                                              fe_form, β, Σβ,
                                              invlink, dinvlink, d2invlink)
            ame_map[v], se_map[v], grad_map[v] = ame,se,grad
        end
    end

    # categorical (including Bool-as-categorical if manually converted)
    for v in cat_vars
        if pairs == :baseline
            ame_d, se_d, g_d = _ame_factor_baseline(tbl0, fe_form, β, Σβ,
                                                    v, invlink, dinvlink)
        else
            ame_d, se_d, g_d = _ame_factor_allpairs(tbl0, fe_form, β, Σβ,
                                                    v, invlink, dinvlink)
        end
        ame_map[v]  = ame_d
        se_map[v]   = se_d
        grad_map[v] = g_d
    end

    return MarginsResult(varlist, repvals, ame_map, se_map, grad_map,
                         n, dof_residual(model),
                         string(family(model).dist), string(family(model).link))
end
