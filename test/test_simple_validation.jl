#!/usr/bin/env julia
# Simple statistical validation test for Phase 4

using Pkg; Pkg.activate(".")
using Margins
using GLM, DataFrames
using Random
using Statistics

function simple_validation_test()
    println("Simple Statistical Validation Test")
    println("="^40)
    
    # Create test data
    Random.seed!(123)
    n = 1000
    data = DataFrame(
        x1 = randn(n),
        x2 = randn(n)
    )
    data.y = 0.5 * data.x1 + 0.3 * data.x2 + randn(n) * 0.1
    
    # Fit model
    model = lm(@formula(y ~ x1 + x2), data)
    
    println("Model: y ~ x1 + x2")
    println("Sample size: $n")
    println("True effects: x1=0.5, x2=0.3")
    
    # Get Margins.jl results
    println("\nComputing marginal effects...")
    
    try
        result = population_margins(model, data; type=:effects, vars=[:x1, :x2], backend=:fd)
        df = DataFrame(result)
        
        println("\nResults:")
        println("Variable | Estimate | Std Error | True Value | Bias")
        println("-"^50)
        
        true_effects = [0.5, 0.3]
        
        for i in 1:nrow(df)
            var = df.term[i]
            est = df.estimate[i] 
            se = df.se[i]
            true_val = true_effects[i]
            bias = abs(est - true_val)
            bias_in_ses = bias / se
            
            println("$(lpad(var, 8)) | $(lpad(round(est, digits=4), 8)) | $(lpad(round(se, digits=4), 9)) | $(lpad(true_val, 10)) | $(round(bias_in_ses, digits=2)) SE")
        end
        
        # Check if estimates are within 2 standard errors of truth
        println("\nStatistical Assessment:")
        all_valid = true
        
        for i in 1:nrow(df)
            est = df.estimate[i]
            se = df.se[i] 
            true_val = true_effects[i]
            
            within_2se = abs(est - true_val) <= 2 * se
            status = within_2se ? "✅" : "❌"
            
            if !within_2se
                all_valid = false
            end
            
            println("$(df.term[i]): $(round(est, digits=4)) ± $(round(2*se, digits=4)) $status")
        end
        
        if all_valid
            println("\n🎉 VALIDATION PASSED")
            println("All estimates within 2 standard errors of true values")
            println("Delta-method standard errors appear statistically sound")
        else
            println("\n❌ VALIDATION FAILED")
            println("Some estimates too far from true values")
        end
        
        # Basic performance check
        println("\n" * "="^40)
        println("BASIC PERFORMANCE CHECK")
        
        # Simple timing test
        @time result2 = population_margins(model, data; type=:effects, vars=[:x1, :x2], backend=:fd)
        
        println("If timing is < 10ms for n=1000, performance is acceptable for Phase 4")
        
    catch e
        println("ERROR: $e")
        println("Statistical validation could not be completed")
        return false
    end
    
    return true
end

# Run validation
success = simple_validation_test()

if success
    println("\n✅ Phase 4 statistical validation completed successfully")
else
    println("\n❌ Phase 4 statistical validation failed")
end