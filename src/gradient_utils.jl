# gradient_utils.jl - Post-calculation utilities for GradientMatrix storage

"""
    get_gradients(result::MarginsResult)

Extract the gradient matrix from a MarginsResult for direct access.
Returns the GradientMatrix object with row-aligned gradients.
"""
function get_gradients(result::MarginsResult)
    return result.gradients
end

"""
    contrast(result::MarginsResult, contrast_vector::Vector{Float64}; Σ=nothing)

Compute linear contrasts over result rows using row-aligned gradient matrix.
Efficient contrast calculation using pre-computed gradient information.

# Arguments
- `result::MarginsResult`: Result with row-aligned gradient matrix
- `contrast_vector::Vector{Float64}`: Contrast weights (length must match number of result rows)
- `Σ=nothing`: Optional covariance matrix override for standard error computation

# Returns
Named tuple with:
- `estimate`: Contrast estimate (c' * estimates)
- `gradient`: Combined gradient (c' * G)  
- `se`: Standard error using delta method
- `contrast`: Original contrast vector
"""
function contrast(result::MarginsResult, contrast_vector::Vector{Float64}; Σ=nothing)
    G = result.gradients.G
    estimates = result.estimate
    
    if length(contrast_vector) != length(estimates)
        error("Contrast vector length ($(length(contrast_vector))) must match number of result rows ($(length(estimates)))")
    end
    
    # Compute linear contrast
    estimate = dot(contrast_vector, estimates)
    combined_gradient = vec(contrast_vector' * G)  # Row vector * matrix = row vector, then vectorize
    
    # Compute standard error using delta method
    if Σ === nothing
        error("Standard error computation requires Σ parameter - pass the model's covariance matrix")
    end
    se = FormulaCompiler.delta_method_se(combined_gradient, Σ)
    
    return (
        estimate = estimate,
        gradient = combined_gradient,
        se = se,
        contrast = contrast_vector,
        coefficients = result.gradients.βnames
    )
end

"""
    bootstrap_effects(result::MarginsResult, β_samples::Matrix{Float64})

Perform bootstrap resampling using stored gradients and coefficient samples.
Much faster than recomputing marginal effects for each bootstrap sample.

# Arguments  
- `result::MarginsResult`: Result with row-aligned gradient matrix
- `β_samples::Matrix{Float64}`: Bootstrap samples of coefficients (n_samples × n_coeffs)

# Returns
Named tuple with bootstrap estimates for each result row.
"""
function bootstrap_effects(result::MarginsResult, β_samples::Matrix{Float64})
    G = result.gradients.G
    n_samples, n_coeffs = size(β_samples)
    n_rows = size(G, 1)
    
    if size(G, 2) != n_coeffs
        error("Coefficient sample dimension ($(n_coeffs)) must match gradient matrix columns ($(size(G, 2)))")
    end
    
    # Compute bootstrap estimates: each row of G dotted with each β sample
    # G is (n_rows × n_coeffs), β_samples is (n_samples × n_coeffs)
    # Result: (n_rows × n_samples)
    bootstrap_estimates = G * β_samples'  # (n_rows × n_coeffs) * (n_coeffs × n_samples)
    
    # Compute statistics for each row
    bootstrap_stats = map(1:n_rows) do i
        row_samples = bootstrap_estimates[i, :]
        (
            mean = mean(row_samples),
            std = std(row_samples),
            quantiles = quantile(row_samples, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]),
            samples = row_samples
        )
    end
    
    return (
        row_stats = bootstrap_stats,
        n_samples = n_samples,
        n_rows = n_rows,
        computation_type = result.gradients.computation_type,
        target = result.gradients.target,
        backend = result.gradients.backend
    )
end

"""
    effect_heterogeneity(result::MarginsResult)

Analyze heterogeneity in effects using row-aligned gradient matrix.
Provides insights into variation in marginal effects across result rows.

# Returns
Named tuple with heterogeneity statistics:
- `estimate_stats`: Statistics on effect estimates
- `gradient_norms`: L2 norms of gradient rows  
- `gradient_stats`: Statistics on gradient magnitudes
- `row_diagnostics`: Per-row gradient diagnostics
"""
function effect_heterogeneity(result::MarginsResult)
    G = result.gradients.G
    estimates = result.estimate
    n_rows, n_coeffs = size(G)
    
    # Compute gradient norms for each row
    gradient_norms = [norm(G[i, :]) for i in 1:n_rows]
    
    # Analyze estimate heterogeneity
    estimate_stats = (
        mean = mean(estimates),
        std = std(estimates), 
        min = minimum(estimates),
        max = maximum(estimates),
        range = maximum(estimates) - minimum(estimates),
        cv = std(estimates) / abs(mean(estimates))  # coefficient of variation
    )
    
    # Analyze gradient heterogeneity  
    gradient_stats = (
        mean_norm = mean(gradient_norms),
        std_norm = std(gradient_norms),
        min_norm = minimum(gradient_norms),
        max_norm = maximum(gradient_norms),
        range_norm = maximum(gradient_norms) - minimum(gradient_norms),
        cv_norm = std(gradient_norms) / mean(gradient_norms)
    )
    
    # Per-row diagnostics
    row_diagnostics = map(1:n_rows) do i
        (
            estimate = estimates[i],
            gradient_norm = gradient_norms[i],
            max_coeff_sensitivity = maximum(abs.(G[i, :])),  # Which coefficient has largest impact
            gradient_direction = G[i, :] ./ gradient_norms[i]  # Unit gradient vector
        )
    end
    
    return (
        estimate_stats = estimate_stats,
        gradient_norms = gradient_norms,
        gradient_stats = gradient_stats,
        row_diagnostics = row_diagnostics,
        n_rows = n_rows,
        n_coefficients = n_coeffs,
        computation_type = result.gradients.computation_type,
        target = result.gradients.target,
        coefficients = result.gradients.βnames
    )
end

"""
    gradient_summary(result::MarginsResult)

Provide a summary of the gradient matrix information.
Useful for understanding gradient storage and computational diagnostics.

# Returns
Named tuple with gradient matrix summary and diagnostics.
"""
function gradient_summary(result::MarginsResult)
    grads = result.gradients
    G = grads.G
    n_rows, n_coeffs = size(G)
    
    # Basic matrix properties
    matrix_stats = (
        dimensions = (n_rows, n_coeffs),
        density = count(!iszero, G) / length(G),  # Proportion of non-zero elements
        condition_number = cond(G'G),  # Condition number of G'G for numerical stability
        rank = rank(G),  # Matrix rank
        norm_frobenius = norm(G, 2),  # Frobenius norm
        max_element = maximum(abs.(G)),
        mean_abs_element = mean(abs.(G))
    )
    
    # Row-wise statistics
    row_norms = [norm(G[i, :]) for i in 1:n_rows]
    row_stats = (
        mean_norm = mean(row_norms),
        std_norm = std(row_norms),
        min_norm = minimum(row_norms),
        max_norm = maximum(row_norms),
        zero_rows = count(==(0.0), row_norms)
    )
    
    # Column-wise statistics (coefficient sensitivities)
    col_norms = [norm(G[:, j]) for j in 1:n_coeffs]
    col_stats = (
        mean_norm = mean(col_norms),
        std_norm = std(col_norms),
        min_norm = minimum(col_norms),
        max_norm = maximum(col_norms),
        zero_cols = count(==(0.0), col_norms),
        coeff_names = grads.βnames
    )
    
    return (
        gradient_type = typeof(grads),
        computation_type = grads.computation_type,
        target = grads.target,
        backend = grads.backend,
        matrix_stats = matrix_stats,
        row_stats = row_stats,
        column_stats = col_stats,
        coefficient_names = grads.βnames,
        storage_efficient = true,  # Row-aligned storage is efficient
        ready_for_contrasts = n_rows > 0,
        ready_for_bootstrap = n_coeffs > 0
    )
end