# population/effects.jl - Zero-allocation AME/AAP computation

using Distributions: Normal, cdf, ccdf

"""
    _population_predictions(engine, data_nt; scale, weights) -> (DataFrame, Matrix{Float64})

Compute average adjusted predictions (AAP) with delta-method standard errors.

This function computes the population average of predicted values across the sample,
providing a measure of the expected outcome for the population.

# Arguments
- `engine::MarginsEngine`: Pre-built computation engine
- `data_nt::NamedTuple`: Data in columntable format
- `scale::Symbol`: Target scale (:response or :link)
- `weights::Union{Vector{Float64}, Nothing}`: Observation weights (nothing for equal weights)

# Returns
- `DataFrame`: Results with estimate, se, t_stat, p_value, n columns
- `Matrix{Float64}`: Gradient matrix G for delta-method standard errors
"""
function _population_predictions(
    engine::MarginsEngine{L}, data_nt::NamedTuple;
    scale=:response, weights=nothing
) where L

    n_obs = length(first(data_nt))

    # Average prediction and its gradient (1×p) via the prediction kernel. Weighted
    # averages are normalized by the sum of all weights (rows with weight ≤ 0 skipped).
    g = Vector{Float64}(undef, length(engine.β))
    total_weight = isnothing(weights) ? Float64(n_obs) : sum(weights)
    mean_prediction = average_prediction!(
        g, engine.row_buf, engine.compiled, data_nt, engine.β,
        predscale(scale, engine.link), 1:n_obs, weights, total_weight
    )
    G = reshape(g, 1, :)  # Single row for population average

    # Delta-method SE (G is 1×p, Σ is p×p)
    se = sqrt((G * engine.Σ * G')[1, 1])
    
    # Create results DataFrame (no variable/contrast for predictions)
    df = DataFrame(
        estimate = [mean_prediction],
        se = [se],
        t_stat = [mean_prediction / se],
        p_value = [2 * ccdf(Normal(), abs(mean_prediction / se))],
        n = [n_obs]  # Add sample size
    )
    
    return df, G
end
