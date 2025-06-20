# typicals.jl

"""
        typicals!(reference_grid, model::RegressionModel, typical::Function)

## Description

Add typical values `reference_grid` for variables that are in the model, but
not specified in `reference_grid` already.

N.B., when `typical` refers to mean, categorical variables will take the value
of a vector, at their percentage in the data for each level. Note that all K levels
are included in the vector, in the order of `levels()`. This works with a modified version of `StatsBase.modelcols()`.

"""
function typicals!(reference_grid, model::RegressionModel, typical::Function)

    dx = if typeof(model) == GeneralizedLinearMixedModel
    elseif typeof(model) == LinearMixedModel
    end

    dx = model.mf.data
    # all model variables (without response)
    # as they exist in the underlying data (e.g., neither interactions nor transformations)
    modelvariables = fieldnames(typeof(dx));
    modelvariables = setdiff(modelvariables, [model.mf.f.lhs.sym]);

    for v in modelvariables
        # add typical values for variables not
        # present in the reference grid
        # straightforward for binary or continous variables
        if v ∉ Symbol.(names(reference_grid))

            # categorical is distinct case when typical function is mean
            reference_grid[!, v] = if typeof(dx[v]) <: CategoricalArray
                if typical == mean
                    pcts = [sum(l .== dx[v])/length(dx[v]) for l in levels(dx[v])]
                    [pcts for _ in 1:nrow(reference_grid)]
                else fill(typical(dx[v]), nrow(reference_grid))
                end
            else
                fill(typical(dx[v]), nrow(reference_grid))
            end
        end
    end
end

"""
    typicals!(
        reference_grid::AbstractDataFrame,
        model::RegressionModel,
        typical::Function,
        dx
    ) → reference_grid

Mutates `reference_grid` by adding “typical” values for every fixed-effect predictor in `model` that isn’t already present.

# Arguments
- `reference_grid::AbstractDataFrame`  
  A table (e.g. `DataFrame`) to which new columns will be added in place.
- `model::RegressionModel`  
  A fitted model object (`LinearModel`, `GLM`, or `MixedModel`).
- `typical::Function`  
  A summary function (e.g. `mean`, `median`) used to compute the default value for each predictor.
- `dx`  
  The original data used to fit `model`.  It will be converted internally via `columntable(dx)`.

# Returns
- The same `reference_grid`, mutated in place (also returned for convenience).

# Behavior
1. **Collect fixed-effect names**:  
   Uses `StatsModels.termvars(formula(model).rhs)` to extract exactly the symbols of the predictors on the right-hand side of the formula.  
2. **Exclude random-effect grouping factors**:  
   If `model isa MixedModel`, any grouping factors (from `(…|G)`) as returned by `fnames(model)` are removed from the list.  
3. **Fill in “typical” values**:  
   For each remaining predictor `v` not already present in `reference_grid`:
   - If `dx[v] isa CategoricalArray` and `typical === mean`, computes the marginal class proportions of `dx[v]` and repeats that vector in each row.  
   - Otherwise, computes `typical(skipmissing(dx[v]))` and fills the column with that constant.

Missing values in `dx` are skipped when computing summaries.

# Example
```julia
using DataFrames, MixedModels, StatsModels

# Simulate data
df = DataFrame(
    y  = randn(100),
    x1 = rand(100),
    x2 = rand(100),
    g  = repeat([:A, :B], inner=50)
)

# Fit a mixed-model
model = fit(MixedModel, @formula(y ~ x1 + x2 + (1|g)), df)

# Create a small reference grid with only x1
ref = DataFrame(x1 = [0.0, 1.0])

# Populate ref with “typical” x2 (mean) and omit g
typicals!(ref, model, mean, df)

# ref now has two columns: x1 (unchanged) and x2 (filled with mean(df.x2))
"""
function typicals!(
    reference_grid::AbstractDataFrame,
    model::RegressionModel,
    typical::Function,
    dx
)
    dx = columntable(dx)

    # 1) get RHS predictor symbols
    all_vars = StatsModels.termvars(formula(model).rhs)

    # 2) drop any grouping factors
    exclude = Symbol[]
    if model isa MixedModel
        append!(exclude, Symbol.(fnames(model)))
    end
    fixed_vars = setdiff(all_vars, exclude)

    # precompute which columns are already present
    existing = Set(Symbol.(names(reference_grid)))
    n = nrow(reference_grid)

    for v in fixed_vars
        if v ∉ existing
            col = dx[v]
            if col isa CategoricalArray
                if typical === mean
                    # total non-missing count
                    N = count(!ismissing, col)
                    # per-level counts over non-missings
                    pcts = [count(==(l), skipmissing(col)) / N for l in levels(col)]
                    reference_grid[!, v] = [pcts for _ in 1:n]
                else
                    reference_grid[!, v] = fill(typical(skipmissing(col)), n)
                end
            else
                reference_grid[!, v] = fill(typical(skipmissing(col)), n)
            end
        end
    end

    return reference_grid
end


"""

typicals!(reference_grid, tbl::Tables.ColumnTable, typical::Function)

## Description

Assumes that `tbl` only contains the relevant columns (the model's fixed effects).
"""
function typicals!(reference_grid, tbl::Tables.ColumnTable, typical::Function)
    # all model variables (without response)
    # as they exist in the underlying data (e.g., neither interactions nor transformations)
    modelvariables = fieldnames(typeof(tbl));
    modelvariables = setdiff(modelvariables, [model.mf.f.lhs.sym]);

    for v in modelvariables
        # add typical values for variables not
        # present in the reference grid
        # straightforward for binary or continous variables
        if v ∉ Symbol.(names(reference_grid))
            # categorical is distinct case when typical function is mean
            reference_grid[!, v] = if typeof(tbl[v]) <: CategoricalArray
                if typical == mean
                    pcts = [sum(l .== tbl[v])/length(tbl[v]) for l in levels(tbl[v])]
                    [pcts for _ in 1:nrow(reference_grid)]
                else fill(typical(tbl[v]), nrow(reference_grid))
                end
            else
                fill(typical(tbl[v]), nrow(reference_grid))
            end
        end
    end
end