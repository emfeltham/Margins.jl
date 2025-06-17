# marginal_effect_curve_discrete_x.jl

"""
    discrete_effect_curve(
      df::DataFrame,
      model,
      x::Symbol,
      values::Vector{<:Real};
      vcov = StatsBase.vcov
    ) -> DataFrame

Compute the average discrete‐change effect of flipping `x` from `values[1]` → each
`values[j]`, j=2…end.  
Returns columns `(x_target, ame_diff, se_diff)`.
"""
function discrete_effect_curve(
    df::DataFrame,
    model,
    x::Symbol,
    values::Vector{<:Real};
    vcov    = StatsBase.vcov
)
    baseline = values[1]

    results = DataFrame(
      x_target = values[2:end],
      ame_diff = Float64[],
      se_diff  = Float64[]
    )

    for v in values[2:end]
        out = ame_discrete_contrast(
            df, model, x;
            values=(baseline, v),
            vcov=vcov
        )
        push!(results, (v, out.ame_diff, out.se_diff))
    end

    return results
end
