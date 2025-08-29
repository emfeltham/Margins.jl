# computation/engine.jl - FormulaCompiler integration

"""
    _build_engine(model, data, dydx, target)

Build FormulaCompiler engine: compiled evaluator, derivative evaluator, β, Σ, link, vars, and data NT.
"""
function _build_engine(model, data, dydx, target)
    data_nt = Tables.columntable(data)
    compiled = FormulaCompiler.compile_formula(model, data_nt)
    vars = dydx === :continuous ? FormulaCompiler.continuous_variables(compiled, data_nt) : collect(dydx)
    de = FormulaCompiler.build_derivative_evaluator(compiled, data_nt; vars=vars)
    β = StatsModels.coef(model)
    Σ = _vcov_model(model, length(β))
    link = target === :mu ? _auto_link(model) : GLM.IdentityLink()
    return (; compiled, de, vars, β, Σ, link, data_nt)
end
