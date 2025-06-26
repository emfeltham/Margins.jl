module Margins # begin module

using CategoricalArrays
import CategoricalArrays.pool

using DataFrames
import DataFrames.Tables.columntable

using Statistics
using StatsBase, StatsModels
using Distributions

using StatsModels
using StatsModels: formula, termvars
import StatsModels: TupleTerm,ColumnTable

using StatsBase
import StatsBase.confint
using StatsBase: vcov
using StatsModels: AbstractTerm, width, termvars

# Needed from Effects.jl to compute effects!:
using Effects: TypicalTerm  # Import the type only
import Effects: diag,vcov, _difference_method!, _responsename, something
import Effects: _invlink_and_deriv, AutoInvLink
import Effects: expand_grid

using GLM
import GLM: LinearModel, GeneralizedLinearModel
import MixedModels
import MixedModels: LinearMixedModel, GeneralizedLinearMixedModel
using MixedModels: MixedModel, fnames, RandomEffectsTerm
using MixedModels: fixef, fixefnames

using StandardizedPredictors

include("family.jl")
export Family, family

# APMs and MEMs
include("reference grid.jl")
include("modelcols_alt.jl") # new function based on modelcols
include("modelcols.jl")
include("typicals.jl")
include("effects2.jl")
include("deltaeffects2.jl")
include("standardized.jl")
include("typicals get.jl")

export setup_refgrid, setup_contrast_grid
export get_typicals, typicals!
export modelvariables

export effects2!
export effectsΔyΔx, effectsΔyΔx, group_effectsΔyΔx

# AMEs
using ForwardDiff, LinearAlgebra
import LinearAlgebra.dot
using GLM: linkinv, mueta
using MixedModels: RandomEffectsTerm

# Core type definitions
import Base: show
using Printf, Distributions

include("AMER.jl")
include("confint.jl")
export AME, AMEResult
export confint

# Helper functions for AMEs
include("fixed_helpers.jl")
include("mueta2.jl")
include("link.jl")

# Core AME functions
include("ame.jl")
include("ame_continuous.jl")
include("ame_factor.jl")
include("ame_representation.jl")

# Exported types and functions
export
    ame, _ame_continuous

# AME contrasts
include("contrasts.jl")
export ContrastResult, contrast, contrast_levels

# Bayesian methods

# Methods for Bayesian models (very rough)
import LinearAlgebra.mul!

include("hpdi.jl")
include("bayes.jl")

end # end module
