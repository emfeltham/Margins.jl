module Margins

using CategoricalArrays
using DataFrames
using StatsBase, StatsModels
using StatsModels: formula, termvars
import StatsBase.confint

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
include("typicals get.jl")

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
using GLM: linkinv, mueta
using MixedModels: RandomEffectsTerm

import Base: show
using Statistics, Printf, Distributions

# Core type definitions
include("fixed_helpers.jl")
include("mueta2.jl")
include("linkhandling.jl")
include("AME.jl")
include("ame_continuous.jl")           # defines struct AME and ame_continuous
include("ame_interaction_continuous.jl")      # defines ame_interaction_continuous
include("ame_discrete_contrast.jl")# defines struct AMEContrast and ame_discrete_contrast
include("marginal_effect_curve_z.jl")
include("marginal_effect_curve_x.jl")
include("marginal_effect_curve_discrete_x.jl")
include("ame_factor_contrasts.jl")

# Exported types and functions
export AME,
       AMEContrast,
       ame_continuous,
       ame_interaction_continuous,
       ame_discrete_contrast,
       marginal_effect_curve_z,
       marginal_effect_curve_x,
       discrete_effect_curve,
       ame_factor_contrasts
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
