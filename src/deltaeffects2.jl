
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

    # probably should sub this in
    # X = modelcols_alt(form.rhs, columntable(reference_grid))
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

function effectsΔyΔx(
    x::Symbol,
    reference_grid::AbstractDataFrame,
    model::RegressionModel,
    df::AbstractDataFrame;
    err_col   = :err,
    typical   = mean,
    invlink   = AutoInvLink(),
    vcov_func = vcov,
    digits    = 3
)
    # 1) fill in “typical” values for predictors missing from reference_grid
    typicals!(reference_grid, model, typical, df)

    # 2) build fixed‐effects design matrix only
    form        = formula(model)
    fixed_terms = tuple(filter(t -> !(t isa RandomEffectsTerm), form.rhs)...)
    X           = modelcols_alt(fixed_terms, columntable(reference_grid))[1]

    # 3) pull coefficients and variance
    β = fixef(model)
    V = vcov_func(model)

    # 4) linear predictor and inverse‐link
    η    = X * β
    p    = similar(η)
    dpdη = similar(η)
    for i in eachindex(η)
        p[i], dpdη[i] = _invlink_and_deriv(invlink, η[i])
    end

    # 5) ensure two rows
    if size(X,1) != 2
        error("effectsΔyΔx!: reference_grid must have exactly 2 rows")
    end

    # 6) compute contrast and SE via delta‐method
    mem   = p[1] - p[2]
    ∇mem  = dpdη[1] * X[1,:] .- dpdη[2] * X[2,:]
    var_m = ∇mem' * V * ∇mem
    se_m  = sqrt(var_m)

    # 7) format response contrast string
    p1n = typeof(p[1]) <: AbstractFloat ? string(round(p[1]; digits=digits)) : p[1]
    p0n = typeof(p[2]) <: AbstractFloat ? string(round(p[2]; digits=digits)) : p[2]
    contrast_str = string(p1n, " - ", p0n)

    # format variable constrast

    nn = string.(reference_grid[!, x])
    variable_str = string(nn[1], " - ", nn[2])

    # 8) assemble output DataFrame
    eff_col = Symbol(:Δ, model.LMM.formula.lhs.sym)
    outname = Symbol(something(eff_col, :mem))
    gradname = :∇mem
    return DataFrame(
        :variable_contrast => variable_str,
        :response_contrast => contrast_str,
        outname             => mem,
        err_col             => se_m,
        gradname             => [∇mem]
    )
end

"""
    group_effectsΔyΔx(
        reference_grid::AbstractDataFrame,
        group_var::Union{Symbol, Vector{Symbol}},
        model::RegressionModel,
        df::AbstractDataFrame;
        eff_col   = nothing,
        err_col   = :err,
        typical   = mean,
        invlink   = AutoInvLink(),
        vcov_func = vcov,
        digits    = 3
    ) -> DataFrame

Apply `effectsΔyΔx!` to each two-row subset of `reference_grid` grouped by `group_var`, returning a combined DataFrame with one row per group.

N.B., the grouped dataframe must split into two-row dataframes.
"""
function group_effectsΔyΔx(
    x::Symbol,
    reference_grid::AbstractDataFrame,
    group_var::Union{Symbol, Vector{Symbol}},
    model::RegressionModel,
    df::AbstractDataFrame;
    err_col   = :err,
    typical   = mean,
    invlink   = AutoInvLink(),
    vcov_func = vcov,
    digits    = 3
)
    gdf = groupby(reference_grid, group_var);
    outdf = DataFrame();
    for (k, a) in pairs(gdf)
        u = effectsΔyΔx(
            x,
            DataFrame(a), model, df;
            err_col=err_col,
            typical=typical,
            invlink=invlink,
            vcov_func=vcov_func,
            digits = digits
        )
        for (k1, k2) in zip(names(k), k)
            u[!, k1] .= k2
        end
        append!(outdf, u)
    end
    outdf.variable .= x
    return outdf
end
