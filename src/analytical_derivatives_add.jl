
"""
    analytical_derivative(term::ZScoredTerm{<:ContinuousTerm}, variable::Symbol, data::NamedTuple)

For a standardized continuous predictor, treat the standardized variable itself as the focal
variable (∂z/∂z = 1).  Other variables remain zero.
"""
function analytical_derivative(term::ZScoredTerm{<:ContinuousTerm}, variable::Symbol, data::NamedTuple)
    n = length(data[variable])
    # only the standardized version of `variable` depends on `variable`
    if term.term.sym === variable
        return ones(Float64, n)
    else
        return zeros(Float64, n)
    end
end
