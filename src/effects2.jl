# effects2.jl

# N.B., one way to make this more general would be allowing
# the specification of different typical functions for each variable
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
    X = modelcols(form.rhs, columntable(reference_grid))

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

#%% MEM
#=
instead of reference_grid, we really want something more like `margins`
we want to specify the dydx variable
really the refgrid should just have _two_ rows for the contrast, whatever it is
could feed or write a special function(s) to generate the referencegrids typically
desired
=#

"""
        effectsΔyΔx(
            reference_grid::AbstractDataFrame, model::RegressionModel;
            eff_col=nothing, err_col=:err,
            typical=mean, invlink=identity, vcov=StatsBase.vcov,
            digits = 3
        )

## Description

Calculate the discrete marginal effect on a two-row dataframe that represents
a reference grid.

- N.B., This appears to be the correct way to calculate contrasts (as opposed to empairs).
- As it stands, this is like `empairs()` but correctly adjusts the standard error
(it doesn't assume independence).
"""
function effectsΔyΔx(
    reference_grid::AbstractDataFrame, model::RegressionModel;
    eff_col=nothing, err_col=:err,
    typical=mean, invlink=identity, vcov=StatsBase.vcov,
    typicals = false,
    digits = 3
)
    
    if !typicals
        typicals!(reference_grid, model, typical)
    end

    form = formula(model)
    X = modelcols(form.rhs, columntable(reference_grid))
    
    η = X * coef(model)  # Linear predictor (η)
    Σ = vcov(model)      # Variance-covariance matrix

    # Check for binary contrast (exactly two rows)
    if nrow(reference_grid) != 2
        error("must be pairwise contrast (nrow(refgrid) should be 2)")
    end

    # Compute Marginal Effect at the Mean (MEM)
    η1, η0 = η[1], η[2]   # η for treatment (1) and control (0)
    X1, X0 = X[1,:], X[2,:]  # Design vectors

    # Transform to probabilities (p1, p0) and derivatives
    p1, dp1dη = _invlink_and_deriv(invlink, η1)
    p0, dp0dη = _invlink_and_deriv(invlink, η0)
    mem = p1 - p0

    # Compute gradients for MEM (∇p1 - ∇p0)
    ∇p1 = dp1dη .* X1  # Gradient for p1: p1(1-p1) * X1
    ∇p0 = dp0dη .* X0  # Gradient for p0: p0(1-p0) * X0
    ∇mem = ∇p1 - ∇p0   # Gradient of MEM

    # Compute variance and standard error
    var_mem = ∇mem' * Σ * ∇mem
    se_mem = sqrt(var_mem)

    # Update reference_grid with MEM and SE
    # this will be a DataFrame with 1 row
    p1n = typeof(p1) <: AbstractFloat ? string(round(p1; digits)) : p1
    p0n = typeof(p0) <: AbstractFloat ? string(round(p0; digits)) : p0

    diffrg = DataFrame(
        :response_contrast => string(p1n) * " - " * string(p0n),
        Symbol(something(eff_col, :mem)) => mem,
        Symbol("mem_" * string(err_col)) => se_mem,
        :∇mem => [∇mem]
    )

    return diffrg
end
