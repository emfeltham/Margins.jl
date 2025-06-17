# marginal_effect_curve_x.jl

"""
    marginal_effect_curve_x(
      df::DataFrame,
      model,
      x::Symbol;
      K::Int = 50,
      δ::Real = 1e-6,
      vcov = StatsBase.vcov,
      typical::Function = mean
    ) -> DataFrame

Compute AME(x) at K points from min(x)→max(x), holding *all other* covariates 
(at their sample-mean or user-supplied `typical`) fixed.  
Returns columns `(x_grid, ame, se)`.
"""
function marginal_effect_curve_x(
    df::DataFrame,
    model,
    x::Symbol;
    K::Int           = 50,
    δ::Real          = 1e-6,
    vcov             = StatsBase.vcov,
    typical::Function= mean
)
    # 1) grid on x
    x_min, x_max = extrema(df[!, x])
    x_grid = collect(range(x_min, x_max; length=K))

    # 2) build a “template” snapshot fixing all other covariates
    template = df[1:1, :]  # 1×p
    for col in names(df)
        if col == x
            continue
        elseif eltype(df[!,col]) <: Number
            template[!, col] .= typical(df[!, col])
        elseif eltype(df[!,col]) <: CategoricalValue
            template[!, col] .= categorical(fill(levels(df[!,col])[1], 1),
                                             levels=levels(df[!,col]))
        else
            template[!, col] .= df[1,col]
        end
    end

    # 3) storage
    ame_vals = Vector{Float64}(undef, K)
    se_vals  = Vector{Float64}(undef, K)

    # 4) loop
    for j in 1:K
        # set x = x_grid[j]
        tmp = deepcopy(template)
        tmp[!, x] .= x_grid[j]

        # call ame_continuous on this 1-row “data frame”
        out = ame_continuous(tmp, model, x; δ=δ, vcov=vcov)
        ame_vals[j] = out.ame
        se_vals[j]  = out.se
    end

    return DataFrame(x_grid=x_grid, ame=ame_vals, se=se_vals)
end
