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
    Σ = try
        StatsBase.vcov(model)
    catch
        Matrix{Float64}(I, length(β), length(β))
    end
    link = target === :mu ? _auto_link(model) : GLM.IdentityLink()
    return (; compiled, de, vars, β, Σ, link, data_nt)
end

"""
    _nrows(data_nt)
"""
_nrows(data_nt::NamedTuple) = length(first(data_nt))

"""
    _is_categorical(data_nt, var::Symbol)

Detect categorical variable by column type.
"""
function _is_categorical(data_nt::NamedTuple, var::Symbol)
    col = getproperty(data_nt, var)
    return (Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)) || (eltype(col) <: Bool)
end
