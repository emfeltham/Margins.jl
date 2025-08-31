# population/effects.jl - Zero-allocation AME/AAP computation

using Distributions: Normal, cdf

"""
    _population_predictions(engine, data_nt; target, kwargs...) -> (DataFrame, Matrix{Float64})

Compute average adjusted predictions (AAP) with delta-method standard errors.

This function computes the population average of predicted values across the sample,
providing a measure of the expected outcome for the population.
"""
function _population_predictions(engine::MarginsEngine, data_nt::NamedTuple; target=:mu, kwargs...)
    n_obs = length(first(data_nt))
    n_params = length(engine.β)
    
    # Preallocate arrays
    predictions = Vector{Float64}(undef, n_obs)
    G = zeros(1, n_params)  # Single row for population average
    
    # Compute predictions and gradients
    for i in 1:n_obs
        # Use FormulaCompiler modelrow to get design matrix row for row i
        X_row = FormulaCompiler.modelrow(engine.compiled, data_nt, i)
        eta_i = dot(X_row, engine.β)
        
        if target === :mu
            # Transform to response scale
            predictions[i] = GLM.linkinv(engine.link, eta_i)
            
            # Compute gradient with chain rule: d/dβ[linkinv(Xβ)] = linkinv'(Xβ) * X
            link_deriv = GLM.mueta(engine.link, eta_i)
            G .+= (link_deriv .* X_row') ./ n_obs
        else # target === :eta
            # Keep on link scale
            predictions[i] = eta_i
            
            # Gradient is just the design matrix row
            G .+= X_row' ./ n_obs
        end
    end
    
    # Compute population average
    mean_prediction = mean(predictions)
    
    # Compute delta-method SE (G is 1 x p, Σ is p x p)
    se = sqrt((G * engine.Σ * G')[1, 1])
    
    # Create results DataFrame
    df = DataFrame(
        term = ["AAP"],
        estimate = [mean_prediction],
        se = [se],
        t_stat = [mean_prediction / se],
        p_value = [2 * (1 - cdf(Normal(), abs(mean_prediction / se)))]
    )
    
    return df, G
end