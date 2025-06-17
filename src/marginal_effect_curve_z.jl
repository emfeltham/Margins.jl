
"""
    marginal_effect_curve_z(
      df::DataFrame,
      model,
      x::Symbol,
      z::Symbol;
      K::Int = 50,
      δ::Real = 1e-6,
      typical::Function = mean,
      vcov = StatsBase.vcov
    ) -> DataFrame

Compute AME(x | z = z_j) for K equally-spaced values z_j ∈ [min(z), max(z)].  
Returns a DataFrame with columns:
  • `z_grid` — the grid of z values  
  • `ame`    — ∂μ/∂x averaged at each z_j  
  • `se_me`  — delta-method SE of that AME  
"""
function marginal_effect_curve_z(
    df::DataFrame,
    model,
    x::Symbol,
    z::Symbol;
    K::Int           = 50,
    δ::Real          = 1e-6,
    typical::Function= mean,
    vcov             = StatsBase.vcov
)
    # 1) Build the z‐grid
    z_min, z_max = extrema(df[!, z])
    z_grid = collect(range(z_min, z_max; length=K))

    # 2) Storage
    ame_vals = Vector{Float64}(undef, K)
    se_vals  = Vector{Float64}(undef, K)

    # 3) Loop: slice at each z_j and compute AME via your existing function
    for j in 1:K
        out = ame_interaction_continuous(
            df, model, x, z, z_grid[j]; δ=δ, typical=typical, vcov=vcov
        )
        ame_vals[j] = out.ame
        se_vals[j]  = out.se
    end

    # 4) Return as DataFrame
    return DataFrame(
        z_grid = z_grid,
        ame    = ame_vals,
        se_me  = se_vals
    )
end
