# ame_discrete_constrast.jl

"""
    ame_discrete_contrast(
      df::DataFrame,
      model,
      x::Symbol;
      values::Tuple{<:Real,<:Real} = (0,1),
      vcov         = StatsBase.vcov
    ) -> AMEContrast

Compute the average discrete change in μ when `x` flips from `values[1]` → `values[2]`,
returning both AMEs at each value and their difference, with SEs & gradients.
"""
function ame_discrete_contrast(
    df::DataFrame,
    model,
    x::Symbol;
    values::Tuple{<:Real,<:Real} = (0,1),
    vcov         = StatsBase.vcov
) 

    # Link functions
    lf = link_functions(model)
    invlink, dinvlink = lf.invlink, lf.dinvlink

    # 1) Prep
    n      = nrow(df)
    β      = coef(model)
    Σβ     = vcov(model)
    x_low, x_high = values

    # 2) Two datasets at x_low / x_high
    df_low   = deepcopy(df);  df_low[!,  x] .= x_low
    df_high  = deepcopy(df);  df_high[!, x] .= x_high
    
    X_low    = fixed_X(model, df_low)
    X_high   = fixed_X(model, df_high)

    # 3) Linear predictors & inverse-link
    η_low    = X_low  * β;    η_high   = X_high * β
    μ_low    = invlink.(η_low); μ_high  = invlink.(η_high)

    # 4) AMEs at each
    ame_low  = mean(μ_low)
    ame_high = mean(μ_high)

    # 5) Derivatives dμ/dη
    dlow     = dinvlink.(η_low)
    dhigh    = dinvlink.(η_high)

    # 6) Gradients ∇AME_low and ∇AME_high
    p        = length(β)
    grads_low  = zeros(p, n)
    grads_high = zeros(p, n)
    for i in 1:n
        grads_low[:, i]  = dlow[i]  .* vec(X_low[i, :])
        grads_high[:, i] = dhigh[i] .* vec(X_high[i, :])
    end
    grad_low  = sum(grads_low, dims=2)[:,1]  ./ n
    grad_high = sum(grads_high, dims=2)[:,1] ./ n

    # 7) Difference and delta-method SE
    ame_diff = ame_high - ame_low
    grad_diff = grad_high .- grad_low
    se_low   = sqrt(dot(grad_low,  Σβ * grad_low))
    se_high  = sqrt(dot(grad_high, Σβ * grad_high))
    se_diff  = sqrt(dot(grad_diff, Σβ * grad_diff))

    return AMEContrast(
      ame_low,
      se_low,
      ame_high,
      se_high,
      ame_diff,
      se_diff,
      grad_low,
      grad_high,
      n,
      coefnames(model)
    )
end
