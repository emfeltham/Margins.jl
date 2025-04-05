# modelvariables.jl

# need to extract the data with the relevant model variables
import StatsModels
using StatsModels: formula, termvars

"""

response, fixedeffects, randomeffects = model_variables(m::MixedModel)

## Description

Extract all variables from a MixedModel. Useful for constructing model matrix,
and calculating typical values.
"""
function model_variables(m::MixedModel)
    # Initialize with response variable
    y = m.formula.lhs.sym

    fixed_terms = StatsModels.termvars(m.formula.rhs[1])

    random_terms = []
    for re in m.reterms
        # Random effects terms (left of |)
        append!(random_terms, StatsModels.termvars(re.trm))
        # this doesn't work, and requires more thought
        # Grouping variable (right of |)
        # push!(res, Symbol(re.grpadef.name))
    end
    
    fixed_terms = unique(fixed_terms)
    random_terms = unique(random_terms)
    return y, fixed_terms, random_terms
end

function modelvariables(f::FormulaTerm)
    y = f.lhs.sym
    rhs = f.rhs
    
    # Extract fixed and random effects terms from formula
    fixed_terms = termvars(rhs)
    random_terms = []
    
    # Traverse formula structure to find random effects terms
    for term in StatsModels.terms(rhs)
        if is_random_effect(term)
            append!(random_terms, termvars(term.trm))
        end
    end
    
    return y, unique(fixed_terms), unique(random_terms)
end

# Helper function to detect random effects terms
is_random_effect(term) = term isa StatsModels.Term && 
    occursin("|", string(term))  # Works for MixedModels syntax

"""

response, fixedeffects, randomeffects = model_variables(ml)

## Description

Extract all variables from a RegressionModel. Useful for constructing model matrix,
and calculating typical values.
"""
function modelvariables(m)
    f = formula(m)
    rhs = f.rhs
    y = f.lhs.sym
    
    # Extract fixed effects terms from formula
    fixed_terms = StatsModels.termvars(rhs[1])  # f[1] gets RHS of formula
    
    # Initialize random effects terms array
    random_terms = []
    
    # Only process random effects if model contains them
    if isdefined(m, :reterms)  # Check for MixedModels structure
        for re in m.reterms
            append!(random_terms, StatsModels.termvars(re.trm))
        end
    end
    
    return y, unique(fixed_terms), unique(random_terms)
end
