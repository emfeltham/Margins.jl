module Margins # begin module

using CategoricalArrays
import CategoricalArrays.pool

using DataFrames, Tables
import DataFrames.DataFrame
import Tables.columntable

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
import MixedModels: sdest
using MixedModels: MixedModel, fnames, RandomEffectsTerm
using MixedModels: fixef, fixefnames

using StandardizedPredictors

## formula helpers
# Logical negation on Bool → Bool, so modelmatrix sees a Bool dummy
not(x::Bool) = !x

# Numeric negation on any Real (Float64, Dual, etc.) → one(x) - x
# Calculates complement
not(x::T) where {T<:Real} = one(x) - x
import StatsModels: term
# whenever you write !term, turn it into function‐term not(term)
Base.:!(t::StatsModels.Term) = term(not, t)

export not
##

include("modelmatrix!.jl")

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

include("workspace.jl")

# Exported types
include("marginsresult.jl")
export MarginsResult
include("margins_to_df.jl")

# Helper functions for AMEs
include("fixed_helpers.jl")
include("mueta2.jl")
include("link.jl")

# Core AME functions
include("core.jl")
include("build_continuous_design.jl")
include("ame_continuous.jl")
include("ame_factor.jl")
include("ame_representation.jl")

include("confint.jl")
export confint

# Exported types and functions
export margins

# AME contrasts
include("contrastresult.jl")
include("contrasts.jl")
include("contrast_to_df.jl")

export DataFrame

# Export type; functions
export ContrastResult, contrast, contrast_levels

# Bayesian methods (very early stage; may be removed)

# Methods for Bayesian models
import LinearAlgebra.mul!

include("hpdi.jl")
include("bayes.jl")

end # end module
