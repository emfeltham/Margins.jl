# population/effects.jl - Zero-allocation AME/AAP computation

using Distributions: Normal, cdf

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
    n_params = length(engine.β)
    
    # Use pre-allocated η_buf as working buffer; size to n_obs
    work = view(engine.η_buf, 1:n_obs)
    G = zeros(1, n_params)  # Single row for population average

    # Delegate hot loop to a helper with concrete arguments to ensure zero allocations
    mean_prediction = _compute_population_predictions!(
        G, work, engine, data_nt, scale, weights
    )
    
    # Delta-method SE (G is 1×p, Σ is p×p)
    se = sqrt((G * engine.Σ * G')[1, 1])
    
    # Create results DataFrame
    df = DataFrame(
        term = ["AAP"],
        estimate = [mean_prediction],
        se = [se],
        t_stat = [mean_prediction / se],
        p_value = [2 * (1 - cdf(Normal(), abs(mean_prediction / se)))],
        n = [n_obs]  # Add sample size
    )
    
    return df, G
end

# Helper with concrete arguments to enable full specialization and avoid per-iteration allocations
function _compute_population_predictions!(
    G::AbstractMatrix{<:Float64}, work, engine::MarginsEngine, 
    data_nt::NamedTuple, scale::Symbol, weights::Union{Vector{Float64}, Nothing}
)
    n_obs = length(first(data_nt))
    
    # Extract engine fields once for cleaner code
    (; compiled, row_buf, β, link) = engine
    
    if isnothing(weights)
        # Unweighted case (original implementation)
        mean_acc = 0.0
        if scale === :response
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
    else
        # Weighted case
        total_weight = sum(weights)
        weighted_acc = 0.0
        
        if scale === :response
            for i in 1:n_obs
                w = weights[i]
                if w > 0  # Skip zero-weight observations
                    FormulaCompiler.modelrow!(row_buf, compiled, data_nt, i)
                    η = dot(row_buf, β)
                    μ = GLM.linkinv(link, η)
                    dμ_dη = GLM.mueta(link, η)
                    weighted_acc += w * μ
                    @inbounds for j in 1:length(row_buf)
                        G[1, j] += (w * dμ_dη * row_buf[j]) / total_weight
                    end
                    work[i] = μ
                else
                    work[i] = 0.0  # Zero for zero-weight observations
                end
            end
        else
            for i in 1:n_obs
                w = weights[i]
                if w > 0  # Skip zero-weight observations
                    FormulaCompiler.modelrow!(row_buf, compiled, data_nt, i)
                    η = dot(row_buf, β)
                    weighted_acc += w * η
                    @inbounds for j in 1:length(row_buf)
                        G[1, j] += (w * row_buf[j]) / total_weight
                    end
                    work[i] = η
                else
                    work[i] = 0.0  # Zero for zero-weight observations
                end
            end
        end
        return weighted_acc / total_weight
    end
end
