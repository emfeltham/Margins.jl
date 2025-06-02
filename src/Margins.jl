module Margins

using CategoricalArrays
using DataFrames
using StatsBase, StatsModels
using StatsModels: formula, termvars
# import StatsModels.modelcols

using Effects: TypicalTerm  # Import the type only
using StatsModels: AbstractTerm, width, termvars
# export MarginsTypicalTerm, margins_typical

# needed from Effects.jl to compute effects!:
import Effects:diag,vcov,_difference_method!,_responsename,something
import Effects:_invlink_and_deriv
import Effects:expand_grid

import DataFrames.Tables.columntable

import StatsModels:TupleTerm,ColumnTable

using StandardizedPredictors

include("reference grid.jl")
include("modelcols_alt.jl") # new function based on modelcols
include("modelcols.jl")
include("typicals.jl")
include("effects2.jl")
include("standardized.jl")

export modelvariables
export effects2!, effectsΔyΔx

export get_typicals, typicals!

# methods for stan/bayes model
import LinearAlgebra.mul!

include("hpdi.jl")
include("bayes.jl")

export setup_refgrid, setup_contrast_grid

# average marginal effects
import LinearAlgebra.dot

include("ame.jl")
include("ame_contrast.jl")
include("marginal_effect_curve_x.jl")

export AME, ame_numeric
export AMEContrast, ame_contrast_numeric
export marginal_effect_curve_x

import StatsBase.confint

function confint(c::AMEContrast)
    ame_diff = c.ame_diff
    se_diff = c.se_diff
    return tuple(sort([ame_diff + 1.96 * se_diff, ame_diff - 1.96 * se_diff])...)
end

function confint(a::AME)
    ame = a.ame
    se = a.se
    return tuple(sort([ame + 1.96 * se, ame - 1.96 * se])...)
end

export confint

function zvalues(df::AbstractDataFrame, z; type = "10-90")
    v = df[!, z]
    return if type == "1SD"
        (low = mean(v) - std(v), high = mean(v) + std(v))
    elseif type == "10-90"
        (high = quantile(v, 0.9), low = quantile(v, 0.1))
    else "error"
    end
end

export zvalues

end
