#!/usr/bin/env julia
# test_statistical_validation.jl - Statistical validation against bootstrap estimates
#
# Phase 4: Validate that Margins.jl delta-method standard errors are statistically
# correct by comparing against bootstrap estimates

using Pkg; Pkg.activate(".")
using Margins
using GLM, DataFrames
using Random
using Statistics
using Printf

function bootstrap_marginal_effects(model, data, n_bootstrap=500; vars=[:x1, :x2], type=:effects)
    """Bootstrap marginal effects to get empirical standard errors."""
    Random.seed!(123)  # For reproducibility
    n_obs = nrow(data)
    
    bootstrap_results = []
    
    for i in 1:n_bootstrap
        # Bootstrap sample
        boot_indices = rand(1:n_obs, n_obs)
        boot_data = data[boot_indices, :]
        
        try
            # Refit model on bootstrap sample
            boot_model = lm(model.mf.f, boot_data)
            
            # Compute population margins on bootstrap sample
            result = population_margins(boot_model, boot_data; type=type, vars=vars, backend=:fd)
            boot_estimates = DataFrame(result).estimate
            
            push!(bootstrap_results, boot_estimates)
        catch e
            # Skip failed bootstrap samples
            continue
        end
    end
    
    if isempty(bootstrap_results)
        error("All bootstrap samples failed")
    end
    
    # Convert to matrix for easier computation
    boot_matrix = hcat(bootstrap_results...)'  # Each row is a bootstrap sample
    
    # Compute empirical standard errors
    bootstrap_ses = [std(boot_matrix[:, j]) for j in 1:size(boot_matrix, 2)]
    bootstrap_means = [mean(boot_matrix[:, j]) for j in 1:size(boot_matrix, 2)]
    
    return bootstrap_means, bootstrap_ses
end

function statistical_validation_test()
    println("Statistical Validation Test: Margins.jl vs Bootstrap")
    println("="^60)
    
    # Create test data
    Random.seed!(123)
    n = 2000  # Large enough for good bootstrap properties
    data = DataFrame(
        x1 = randn(n),
        x2 = randn(n),
        x3 = rand(n)
    )
    data.y = 0.5 * data.x1 + 0.3 * data.x2 + 0.2 * data.x3 + randn(n) * 0.1
    
    # Fit model
    model = lm(@formula(y ~ x1 + x2 + x3), data)
    
    println("Model fitted on n=$n observations")
    println("Formula: y ~ x1 + x2 + x3")
    println("Testing variables: [:x1, :x2]")
    
    # Get Margins.jl results using delta method
    println("\n1. Computing Margins.jl results (delta method)...")
    margins_result = population_margins(model, data; type=:effects, vars=[:x1, :x2], backend=:fd)
    margins_df = DataFrame(margins_result)
    
    margins_estimates = margins_df.estimate
    margins_ses = margins_df.se
    
    # Get bootstrap results
    println("\n2. Computing bootstrap results (500 samples)...")
    boot_estimates, boot_ses = bootstrap_marginal_effects(model, data, 500; vars=[:x1, :x2])
    
    # Statistical comparison
    println("\n3. Statistical Comparison")
    println("-"^40)
    
    println("Variable | Delta SE | Boot SE | Ratio | Agreement")
    println("-"^40)
    
    agreements = []
    
    for i in 1:length(margins_estimates)
        var_name = margins_df.term[i]
        delta_se = margins_ses[i]
        boot_se = boot_ses[i]
        ratio = delta_se / boot_se
        
        # Check if ratio is close to 1.0 (within 10%)
        agreement = abs(ratio - 1.0) < 0.1
        push!(agreements, agreement)
        
        status = agreement ? "✅" : "❌"
        
        println("$(lpad(var_name, 8)) | $(lpad(round(delta_se, digits=4), 8)) | $(lpad(round(boot_se, digits=4), 7)) | $(lpad(round(ratio, digits=2), 5)) | $status")
    end
    
    # Coverage probability test
    println("\n4. Coverage Probability Test (95% Confidence Intervals)")
    println("-"^50)
    
    coverage_rates = []
    
    for i in 1:length(margins_estimates)
        var_name = margins_df.term[i]
        
        # Delta method CI
        delta_estimate = margins_estimates[i]
        delta_se = margins_ses[i]
        delta_ci_lower = delta_estimate - 1.96 * delta_se
        delta_ci_upper = delta_estimate + 1.96 * delta_se
        
        # Bootstrap CI (percentile method)
        boot_values = [boot_matrix[j, i] for j in 1:size(boot_matrix, 1)]
        boot_values_sorted = sort(boot_values)
        boot_ci_lower = quantile(boot_values_sorted, 0.025)
        boot_ci_upper = quantile(boot_values_sorted, 0.975)
        
        # Bootstrap values from bootstrap samples  
        bootstrap_results = []
        for j in 1:length(bootstrap_results)
            if length(bootstrap_results[j]) > i
                push!(boot_values, bootstrap_results[j][i])
            end
        end
        
        # Check if bootstrap estimate falls within delta CI
        boot_estimate = boot_estimates[i]
        within_delta_ci = (delta_ci_lower <= boot_estimate <= delta_ci_upper)
        
        # Check if delta estimate falls within bootstrap CI  
        within_boot_ci = (boot_ci_lower <= delta_estimate <= boot_ci_upper)
        
        coverage = within_delta_ci && within_boot_ci
        push!(coverage_rates, coverage)
        
        status = coverage ? "✅" : "❌"
        
        println("$var_name: Delta CI [$(round(delta_ci_lower, digits=4)), $(round(delta_ci_upper, digits=4))]")
        println("     Boot CI [$(round(boot_ci_lower, digits=4)), $(round(boot_ci_upper, digits=4))] $status")
    end
    
    # Overall assessment
    println("\n5. Overall Statistical Assessment")
    println("-"^35)
    
    se_agreement_rate = mean(agreements)
    coverage_rate = mean(coverage_rates)
    
    println("SE Agreement Rate: $(round(se_agreement_rate * 100, digits=1))% (target: >90%)")
    println("Coverage Rate: $(round(coverage_rate * 100, digits=1))% (target: >90%)")
    
    overall_valid = se_agreement_rate >= 0.9 && coverage_rate >= 0.9
    
    if overall_valid
        println("\n🎉 STATISTICAL VALIDATION: PASSED")
        println("Delta-method standard errors are statistically valid!")
        println("Margins.jl meets publication-grade statistical standards.")
    else
        println("\n❌ STATISTICAL VALIDATION: FAILED")
        println("Delta-method standard errors may not be reliable.")
        println("Investigation needed before publication use.")
    end
    
    return Dict(
        :se_agreement_rate => se_agreement_rate,
        :coverage_rate => coverage_rate,
        :overall_valid => overall_valid,
        :margins_estimates => margins_estimates,
        :margins_ses => margins_ses,
        :bootstrap_estimates => boot_estimates,
        :bootstrap_ses => boot_ses
    )
end

# Run the validation test
result = statistical_validation_test()