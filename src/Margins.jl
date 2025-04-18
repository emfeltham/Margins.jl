module Margins

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

include("reference grid.jl")
include("modelcols_alt.jl") # new function based on modelcols
include("modelcols.jl")
include("typicals.jl")
include("effects2.jl")

export modelvariables
export effects2!, effectsΔyΔx

export get_typicals, typicals!

# methods for stan model
import LinearAlgebra.mul!

include("hpdi.jl")
include("bayes.jl")

export setup_refgrid, setup_contrast_grid

end
