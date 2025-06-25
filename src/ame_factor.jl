# ame_factor.jl

"""
_ame_factor(df, model, f, fe_form, β, invlink, dinvlink, d2invlink, vcov) -> (ame, se, grad)

Compute discrete AME for a categorical predictor `f` by comparing each observation’s level
against the baseline level and applying the Δ-method for standard errors.

# Arguments
- `df::DataFrame`: dataset used for predictions
- `model`: fitted GLM/GLMM/LinearModel/MixedModel
- `f::Symbol`: name of the factor (categorical) variable
- `fe_form`: fixed-effects formula or contrasts for constructing model matrix
- `β::AbstractVector`: coefficient vector of fixed effects
- `invlink`: inverse link function μ(η)
- `dinvlink`: first derivative of inverse link μ′(η)
- `d2invlink`: second derivative of inverse link (unused)
- `vcov`: function to extract covariance matrix of β

# Returns
- `ame_val::Float64`: average of μ(obs)−μ(base) across observations
- `se::Float64`: standard error of the AME via Δ-method
- `grad::Vector{Float64}`: Δ-method gradient of AME w.r.t. β
"""

"""
    _ame_factor_pair(
      df_row, f, j, k, fe_form, β, invlink, dinvlink
    ) -> (δμ, Δgrad)

On a single observation `df_row`, compare factor level `k` vs `j`:

- `δμ = μ(k) - μ(j)`  
- `Δgrad = dμ(k)*X(k) - dμ(j)*X(j)`  
"""

"""
_ame_factor_pair(...) -> (ame, se, grad)
Compute AME and SE for one CategoricalArrays factor-level contrast (j vs k).
"""
function _ame_factor_pair(
    df::DataFrame, model, f::Symbol,
    j, k,
    fe_form, β::AbstractVector,
    invlink, dinvlink, vcov
)
    # f must be a CategoricalArray
    cat_col = df[!, f]
    @assert cat_col isa CategoricalArray "Column $(f) must be a CategoricalArray"
    pool = CategoricalArrays.pool(cat_col)

    X0 = modelmatrix(fe_form, df)
    n, p = size(X0)
    diffs = zeros(n)
    grad_sum = zeros(eltype(β), p)

    for i in 1:n
        df_row = df[i:i, :]
        
        # level j: only use the integer-code constructor
        tmpj = copy(df_row)
        j_idx = findfirst(==(j), levels(pool))
        @assert j_idx !== nothing "level $j not found in pool"
        tmpj[!, f] .= CategoricalValue(j_idx, pool)
        
        Xj = vec(modelmatrix(fe_form, tmpj)); ηj = dot(Xj, β)
        μj = invlink(ηj); dμj = dinvlink(ηj)
        
        # level k: only use the integer-code constructor
        tmpk = copy(df_row)
        k_idx = findfirst(==(k), levels(pool))
        @assert k_idx !== nothing "level $k not found in pool"
        tmpk[!, f] .= CategoricalValue(k_idx, pool)

        Xk = vec(modelmatrix(fe_form, tmpk)); ηk = dot(Xk, β)
        μk = invlink(ηk); dμk = dinvlink(ηk)

        diffs[i] = μk - μj
        @inbounds grad_sum .+= dμk .* Xk .- dμj .* Xj
    end

    ame  = mean(diffs)
    grad = grad_sum ./ n
    se   = sqrt(dot(grad, vcov(model) * grad))
    return ame, se, grad
end

"""
_ame_factor_allpairs(...) -> (Dict, Dict, Dict)
Loop `_ame_factor_pair` over all unique CategoricalArrays level-pairs.
"""
function _ame_factor_allpairs(
    df::DataFrame, model, f::Symbol,
    fe_form, β::AbstractVector,
    invlink, dinvlink, vcov
)
    cat_col = df[!, f]
    @assert cat_col isa CategoricalArray "Column $(f) must be a CategoricalArray"
    lvls  = levels(cat_col)
    pairs = [(j, k) for j in lvls for k in lvls if j < k]

    ame  = Dict{Tuple,Float64}()
    se   = Dict{Tuple,Float64}()
    grad = Dict{Tuple,Vector{Float64}}()

    for (j, k) in pairs
        a, s, g = _ame_factor_pair(
            df, model, f, j, k,
            fe_form, β, invlink, dinvlink, vcov
        )
        ame[(j, k)]  = a
        se[(j, k)]   = s
        grad[(j, k)] = g
    end
    return ame, se, grad
end
