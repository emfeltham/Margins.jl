# computation/engine.jl - FormulaCompiler integration

"""
    _build_engine(model, data, dydx, target)

Build FormulaCompiler engine: compiled evaluator, derivative evaluator, β, Σ, link, vars, and data NT.
"""
function _build_engine(model, data, dydx, target)
    data_nt = Tables.columntable(data)
    compiled = FormulaCompiler.compile_formula(model, data_nt)
    vars = dydx === :continuous ? FormulaCompiler.continuous_variables(compiled, data_nt) : collect(dydx)
    
    # Only pass continuous variables to build_derivative_evaluator
    # Categorical variables don't need derivatives and are handled via contrast_modelrow!
    continuous_vars = FormulaCompiler.continuous_variables(compiled, data_nt)
    vars_for_de = filter(v -> v in continuous_vars, vars)
    
    # Build derivative evaluator only if there are continuous variables
    de = if !isempty(vars_for_de)
        FormulaCompiler.build_derivative_evaluator(compiled, data_nt; vars=vars_for_de)
    else
        nothing  # No continuous variables, no derivative evaluator needed
    end
    
    β = StatsModels.coef(model)
    Σ = _vcov_model(model, length(β))
    link = target === :mu ? _auto_link(model) : GLM.IdentityLink()
    return (; compiled, de, vars, β, Σ, link, data_nt)
end
