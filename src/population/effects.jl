# population/effects.jl - Zero-allocation AME/AAP computation

using Distributions: Normal, cdf

"""
    _population_predictions(engine, data_nt; target, kwargs...) -> (DataFrame, Matrix{Float64})

Compute average adjusted predictions (AAP) with delta-method standard errors.

This function computes the population average of predicted values across the sample,
providing a measure of the expected outcome for the population.
"""
function _population_predictions(engine::MarginsEngine{L}, data_nt::NamedTuple; target=:mu, kwargs...) where L
    n_obs = length(first(data_nt))
    n_params = length(engine.β)
    
    # Use pre-allocated η_buf as working buffer; size to n_obs
    work = view(engine.η_buf, 1:n_obs)
    G = zeros(1, n_params)  # Single row for population average

    # Delegate hot loop to a helper with concrete arguments to ensure zero allocations
    mean_prediction = _population_predictions_impl!(G, work, engine.compiled, engine.row_buf,
                                                    engine.β, engine.link, data_nt, target)
    
    # Delta-method SE (G is 1×p, Σ is p×p)
    se = sqrt((G * engine.Σ * G')[1, 1])
    
    # Create results DataFrame
    df = DataFrames.DataFrame(
        term = ["AAP"],
        estimate = [mean_prediction],
        se = [se],
        t_stat = [mean_prediction / se],
        p_value = [2 * (1 - cdf(Normal(), abs(mean_prediction / se)))]
    )
    
    return df, G
end

# Helper with concrete arguments to enable full specialization and avoid per-iteration allocations
function _population_predictions_impl!(G::AbstractMatrix{<:Float64}, work, compiled, row_buf::Vector{Float64},
                                       β::Vector{Float64}, link, data_nt::NamedTuple, target::Symbol)
    n_obs = length(first(data_nt))
    mean_acc = 0.0
    if target === :mu
        for i in 1:n_obs
            FormulaCompiler.modelrow!(row_buf, compiled, data_nt, i)
            η = dot(row_buf, β)
            μ = GLM.linkinv(link, η)
            dμ_dη = GLM.mueta(link, η)
            mean_acc += μ
            @inbounds for j in 1:length(row_buf)
                G[1, j] += (dμ_dη * row_buf[j]) / n_obs
            end
            work[i] = μ
        end
    else
        for i in 1:n_obs
            FormulaCompiler.modelrow!(row_buf, compiled, data_nt, i)
            η = dot(row_buf, β)
            mean_acc += η
            @inbounds for j in 1:length(row_buf)
                G[1, j] += row_buf[j] / n_obs
            end
            work[i] = η
        end
    end
    return mean_acc / n_obs
end
