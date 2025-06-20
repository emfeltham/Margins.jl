# effects2.jl

"""
    effects2!(
        reference_grid::DataFrame,
        model::RegressionModel;
        eff_col = nothing,
        err_col = :err,
        typical = mean,
        invlink = identity,
        vcov = StatsBase.vcov
    )

Compute adjusted ("effects at the mean") predictions for each row of a reference grid, 
using a fitted GLM (or other RegressionModel).

# Arguments
- `reference_grid::DataFrame`
  A DataFrame whose columns specify the focal predictor values at which you want predictions.
- `model::RegressionModel`
  A fitted model returned by GLM.jl (e.g. `lm` or `glm`), or any object supporting
  `coef`, `formula`, and `StatsBase.vcov`.

# Keyword Arguments
- `eff_col::Union{Symbol,Nothing}`  
  Name of the column to store the predicted (transformed) responses.  
  If `nothing`, a default name based on the model’s response variable is used.
- `err_col::Symbol`  
  Name of the column to store the standard error of each prediction. Default `:err`.
- `typical::Function`  
  Function to compute “typical” values (e.g. `mean`, `median`) for covariates *not* present
  in `reference_grid`. For categorical variables, level-proportions are computed when `typical==mean`.
- `invlink::Function`  
  Inverse link function for back-transforming the linear predictor (e.g. `identity`, `logistic`).  
  Default is `identity` (no transformation).
- `vcov::Function`  
  Function to extract the coefficient variance-covariance matrix. Default is `StatsBase.vcov`.

# Returns
A mutated copy of `reference_grid` with two new columns:
1. The adjusted predictions on the response scale (column named by `eff_col` or a default).  
2. Their standard errors (column named by `err_col`).

# Example
```julia
using DataFrames, CategoricalArrays, GLM

# Simulate data
df = DataFrame(
    y = randn(100) .+ 2 .* (rand(100) .> 0.5),
    x = rand(100),
    g = categorical(rand(["A","B"], 100))
)

# Fit a linear model
m = lm(@formula(y ~ x + g), df)

# Build a reference grid over x
grid = DataFrame(x = range(extrema(df.x)..., length=5))

# Compute effects at the mean of g
effects2!(grid, m; eff_col=:pred, err_col=:se_pred)
# grid now contains :pred and :se_pred columns
"""
function effects2!(
    reference_grid::DataFrame, model::RegressionModel;
    eff_col=nothing, err_col=:err, typical=mean, invlink=identity,
    vcov=StatsBase.vcov
)

    # this section is different
    # we add the typical value for every model variable not specified in the
    # reference grid (which may vary) to the reference grid, accounting for the
    # special case of categorical variables.
    typicals!(reference_grid, model, typical)
    form = formula(model)
    X = modelcols_alt(form.rhs, columntable(reference_grid))

    eff = X * coef(model)
    err = sqrt.(diag(X * vcov(model) * X'))
    _difference_method!(eff, err, model, invlink)

    reference_grid[!, something(eff_col, _responsename(model))] = eff
    reference_grid[!, err_col] = err
    return reference_grid
    # XXX remove DataFrames dependency
    # this doesn't work for a DataFrame and isn't mutating
    # return (; reference_grid..., depvar => eff, err_col => err)
end

function effects2!(
    reference_grid::DataFrame,
    model::RegressionModel,
    df::AbstractDataFrame;
    eff_col    = nothing,
    err_col    = :err,
    typical    = mean,
    invlink    = AutoInvLink(),  # ← default to the “auto” inverse‐link
    vcov_func  = vcov,
    filtervars = true
)
    # 1) remember what was in the grid before we added anything
    orig_cols = Symbol.(names(reference_grid))

    # 2) fill in “typical” values for any predictors missing from reference_grid
    typicals!(reference_grid, model, typical, df)

    # 3) build fixed‐effects design matrix only
    form        = formula(model)
    fixed_terms = tuple(filter(t -> !(t isa RandomEffectsTerm), form.rhs)...)
    X        = modelcols_alt(fixed_terms, columntable(reference_grid))[1]

    # 4) pull out the fixed‐effects β’s and their variance
    β = fixef(model)
    V = vcov_func(model)

    # 5) compute linear predictor & SE
    η  = X * β
    se = sqrt.(diag(X * V * X'))

    # 6) get the right inverse‐link from AutoInvLink
    _difference_method!(η, se, model, invlink)

    # 7) write results back into the grid
    outname = something(eff_col, _responsename(model))
    reference_grid[!, outname] = η
    reference_grid[!, err_col]   = se

    # 8) optionally drop everything except the original grid + eff + err
    if filtervars
        select!(reference_grid, [orig_cols... , Symbol(outname), err_col])
    end

    return reference_grid
end

# function effects2!(
#     reference_grid::DataFrame, model::RegressionModel, df::AbstractDataFrame;
#     eff_col=nothing, err_col=:err, typical=mean, invlink=identity,
#     vcov=StatsBase.vcov
# )

#     # this section is different
#     # we add the typical value for every model variable not specified in the
#     # reference grid (which may vary) to the reference grid, accounting for the
#     # special case of categorical variables.
#     typicals!(reference_grid, model, typical, df)
#     form = formula(model)
#     X = modelcols_alt(form.rhs, columntable(reference_grid))

#     eff = X * coef(model)
#     err = sqrt.(diag(X * vcov(model) * X'))
#     _difference_method!(eff, err, model, invlink)

#     reference_grid[!, something(eff_col, _responsename(model))] = eff
#     reference_grid[!, err_col] = err
#     return reference_grid
#     # XXX remove DataFrames dependency
#     # this doesn't work for a DataFrame and isn't mutating
#     # return (; reference_grid..., depvar => eff, err_col => err)
# end

## example

# df = DataFrame(y = rand(10), a = rand(10), b = rand(10), c = categorical(rand(["x", "y", "z"], 10)),)

# fo = @formula(y ~ b + c)
# fo = @formula(y ~ b * c)
# fo = @formula(y ~ log(b) + c + a&c) # target case


# m1 = lm(fo, df; contrasts = Dict(:b => ZScore(), :a => ZScore()))
# m2 = lm(fo, df)

# mf = m1.mf
# mf.schema
# mf.f

# m2.mf.schema

# reference_grid = expand_grid(
#     Dict(
#         :b => range(extrema(df.b)..., 3)
#     )
# );

# effects2!(reference_grid, m1)

# reference_grid2 = expand_grid(
#     Dict(
#         :b => range(extrema(df.b)..., 3)
#     )
# );

# effects2!(reference_grid2, m2)
