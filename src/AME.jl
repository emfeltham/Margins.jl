"""
    ame(
      model,
      vars::Union{Symbol, AbstractVector{Symbol}},
      df::AbstractDataFrame;
      vcov::Function = StatsBase.vcov,
      repvals::Dict{Symbol, Vector} = Dict{Symbol, Vector}()
    ) -> AMEResult

Compute average marginal effects (AME) for one or more predictors.

# Arguments
- `model` : a fitted GLM/GLMM/LinearModel/MixedModel
- `vars`  : either a single `Symbol` or a `Vector{Symbol}` of predictors
- `df`    : the DataFrame used to fit the model (required)
- `vcov`  : function to extract the fixed-effects covariance matrix
             (defaults to `StatsBase.vcov`)
- `repvals`: a `Dict{Symbol, Vector}` of representative values for MERs

# Returns
An `AMEResult` containing AMEs, standard errors, and gradients for each var.
"""
function ame(
    model,
    vars::Union{Symbol, AbstractVector{Symbol}},
    df::AbstractDataFrame;
    vcov::Function = StatsBase.vcov,
    repvals = Dict()
)
    @assert df isa AbstractDataFrame "You must provide a DataFrame as `df`"
    # normalize vars to a Vector{Symbol}
    varlist = isa(vars, Symbol) ? [vars] : collect(vars)
    for v in string.(varlist)
        @assert v in names(df) "Variable $(v) not found in DataFrame"
    end

    # set up link, design, and parameters
    invlink, dinvlink, d2invlink = link_functions(model)
    fe_form  = fixed_effects_form(model)
    X        = modelmatrix(fe_form, df)
    β        = coef(model)
    n        = size(X, 1)

    # result containers
    ame_map = Dict{Symbol, Any}()
    se_map  = Dict{Symbol, Any}()
    g_map   = Dict{Symbol, Any}()

    for v in varlist
        if !isempty(repvals)
            ame_val, se_val, grad_v = _ame_representation(
                df, model, v, repvals,
                fe_form, β, invlink, dinvlink, d2invlink, vcov
            )
        elseif eltype(df[!, v]) <: Number
            ame_val, se_val, grad_v = _ame_continuous(
                df, model, v,
                fe_form, β, dinvlink, d2invlink, vcov
            )
        else
            ame_val, se_val, grad_v = _ame_factor_allpairs(
                df, model, v,
                fe_form, β, invlink, dinvlink, vcov
            )
        end
        ame_map[v] = ame_val
        se_map[v]  = se_val
        g_map[v]   = grad_v
    end

    fam     = string(family(model).dist)
    linkstr = string(family(model).link)
    
    dofr = dof_residual(model) # get residual degrees of freedom
    return AMEResult(varlist, repvals, ame_map, se_map, g_map, n, dofr, fam, linkstr)
end
