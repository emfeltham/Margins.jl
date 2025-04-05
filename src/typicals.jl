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

function typicals!(reference_grid, model::RegressionModel, typical::Function, dx)
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