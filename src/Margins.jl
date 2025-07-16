module Margins

using BenchmarkTools: @belapsed

using FormulaCompiler
using FormulaCompiler: compile_formula, compile_derivative_formula, CompiledFormula, CompiledDerivativeFormula
using FormulaCompiler: create_scenario, DataScenario

using DataFrames, CategoricalArrays, Tables
import DataFrames.DataFrame
using Tables: ColumnTable, columntable

using StandardizedPredictors
import StandardizedPredictors: ZScoredTerm

using LinearAlgebra: norm, dot
using Statistics: mean, std, var
using StatsModels
using StatsModels: TupleTerm

using GLM: 
    linkinv, mueta, IdentityLink, LogLink, LogitLink, ProbitLink, 
    CloglogLink, InverseLink, InverseSquareLink, SqrtLink, PowerLink, Link
using Distributions: Normal, pdf
using ForwardDiff


# Needed from Effects.jl to compute APM/MEM effects!:
using Effects: TypicalTerm  # Import the type only
import Effects: diag, vcov, _difference_method!, _responsename, something
import Effects: _invlink_and_deriv, AutoInvLink
import Effects: expand_grid

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

# AMEs with FormulaCompiler integration
using ForwardDiff, LinearAlgebra
using LinearAlgebra.BLAS
import LinearAlgebra.dot
using GLM: linkinv, mueta

# Core type definitions
import Base: show
using Printf, Distributions

# Bayesian methods (very early stage; may be removed)
include("hpdi.jl")
include("bayes.jl")

####

include("workspace.jl")

include("link_functions.jl")
include("marginal_effects_results.jl")

include("marginal_effects_core.jl")
include("representative_effects.jl")
export compute_marginal_effects
include("continuous_variable_effects.jl")
include("categorical_variable_effects.jl")
# include("categorical_variable_effects_utilities.jl")

# contrasts and export
# include("margins_to_df.jl")
# AME contrasts
# include("contrastresult.jl")
# include("contrasts.jl")
# include("contrast_to_df.jl")
# Export type; functions
# export ContrastResult, contrast, contrast_levels
# Exported types

export DataFrame

end # end module
