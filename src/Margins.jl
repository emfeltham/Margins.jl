module Margins

using StatsModels
import StatsModels:modelcols
using StatsModels: formula, termvars

# needed from Effects.jl to compute effects!:
import Effects:diag,vcov,_difference_method!,_responsename,something
import Effects:_invlink_and_deriv

include("reference grid.jl")
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
