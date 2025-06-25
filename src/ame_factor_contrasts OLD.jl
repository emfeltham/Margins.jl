# ame_factor_contrasts.jl

"""
    ame_factor_contrasts(
      df::DataFrame,
      model,
      x::Symbol;
      vcov = StatsBase.vcov
    ) -> DataFrame

For a categorical `x` with levels ℓ₁,ℓ₂,…,ℓₖ, compute for each pair (ℓᵢ→ℓⱼ), i<j,
the average discrete-change effect
  AMEᵢⱼ =  n⁻¹ ∑[μ(x=ℓⱼ) − μ(x=ℓᵢ)]
and its delta-method SE.  

Returns a DataFrame with columns
  • `from`     — ℓᵢ  
  • `to`       — ℓⱼ  
  • `ame_diff` — AMEᵢⱼ  
  • `se_diff`  — SE(AMEᵢⱼ)
"""
function ame_factor_contrasts(
    df::DataFrame,
    model,
    x::Symbol;
    vcov = StatsBase.vcov
) 

    # Link functions
    lf = link_functions(model)
    invlink, dinvlink = lf.invlink, lf.dinvlink

    # 1) ensure x is categorical
    if !(eltype(df[!,x]) <: CategoricalValue)
        df = copy(df)
        df[!, x] = categorical(df[!, x])
    end
    levels_x = levels(df[!, x])
    k = length(levels_x)

    # 2) common pieces
    n    = nrow(df)
    form = formula(model)
    β    = coef(model)
    Σβ   = vcov(model)

    # 3) prepare output
    out = DataFrame(
      from     = String[],
      to       = String[],
      ame_diff = Float64[],
      se_diff  = Float64[]
    )

    # 4) loop over all i<j
    for i in 1:k-1, j in i+1:k
        ℓ_i = levels_x[i]
        ℓ_j = levels_x[j]

        # build two datasets
        df_i = deepcopy(df); df_i[!, x] .= ℓ_i
        df_j = deepcopy(df); df_j[!, x] .= ℓ_j

        # design matrices
        X_i = fixed_X(model, df_i)
        X_j = fixed_X(model, df_j)

        # linear predictors & μ’s
        η_i = X_i * β; η_j = X_j * β
        μ_i = invlink.(η_i); μ_j = invlink.(η_j)

        # AMEs at each level
        ame_i = mean(μ_i)
        ame_j = mean(μ_j)
        diff  = ame_j - ame_i

        # first-derivative weights
        d_i = dinvlink.(η_i)
        d_j = dinvlink.(η_j)

        # gradients ∇AME_i and ∇AME_j
        p = length(β)
        grads_i = zeros(p, n)
        grads_j = zeros(p, n)
        for t in 1:n
            grads_i[:, t] = d_i[t] .* vec(X_i[t, :])
            grads_j[:, t] = d_j[t] .* vec(X_j[t, :])
        end
        g_i = sum(grads_i, dims=2)[:,1] ./ n
        g_j = sum(grads_j, dims=2)[:,1] ./ n

        # contrast gradient & SE
        g_diff = g_j .- g_i
        se = sqrt(dot(g_diff, Σβ * g_diff))

        push!(out, (
          string(ℓ_i),
          string(ℓ_j),
          diff,
          se
        ))
    end

    return out
end
