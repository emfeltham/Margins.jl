# Margins.jl
# Clean module with merged core

module Margins

# Dependencies (unchanged)
using Random
using ForwardDiff
using StatsModels, GLM, CategoricalArrays, Tables, DataFrames
using LinearAlgebra


import MixedModels
using MixedModels: MixedModel, LinearMixedModel, GeneralizedLinearMixedModel, RandomEffectsTerm

using StandardizedPredictors: ZScoredTerm, ScaledTerm, CenteredTerm
using Statistics

# ________________________________________ Utility functions
not(x::Bool) = !x
not(x::T) where {T<:Real} = one(x) - x
export not

# # FormulaCompiler foundation components
# using FormulaCompiler: 
#     compile_formula, compile_derivative_formula,
#     CompiledFormula, CompiledDerivativeFormula, 
#     modelrow!, create_scenario, DataScenario,
#     create_scenario_grid, ScenarioCollection,
#     OverrideVector

# # Include files in dependency order
# include("workspace.jl")                    # Enhanced workspace
# include("core.jl")                         # Merged core implementation  
# include("continuous_variable_effects.jl")  # Analytical derivatives
# include("categorical_variable_effects.jl") # Cleaned categorical effects with scenarios
# include("representative_effects.jl")       # Cleaned representative values utilities
# include("marginal_effects_results.jl")     # Result types and display
# include("link_functions.jl")               # GLM utilities
# include("mueta2.jl")                       # f`` for invlink

# # CLEAN EXPORTS - Focus on user API
# export margins                            # Main user function
# export not                               # Utility for formulas

# # Result types
# export MarginalEffectsResult

# # Advanced exports (for package developers/power users)
# export MarginalEffectsWorkspace           # For custom workflows
# export extract_link_functions, mueta2     # GLM utilities

# # Representative values utilities (for advanced use cases)
# export create_representative_value_grid, validate_representative_values

# # Note: Removed deprecated exports and redundant scenario functions
# # Note: FormulaCompiler scenario functions (create_scenario, etc.) are imported but not re-exported
# #       Users should import FormulaCompiler directly for advanced scenario work


# ________________________________________ APM and MEM estimation
using Effects: expand_grid, AutoInvLink
using StatsModels: TupleTerm
using Tables: ColumnTable

include("modelcols.jl")
include("modelcols_alt.jl")
include("standardized.jl")
include("typicals.jl")
include("typicals get.jl")

include("effects2.jl")
include("deltaeffects2.jl")

export effects2!, effectsΔyΔx

include("reference grid.jl")
export setup_refgrid, setup_contrast_grid

end # module
