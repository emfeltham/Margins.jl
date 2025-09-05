# function_composition_tests.jl - Phase 2.1 Function Composition Test Suite
#
# Comprehensive test suite for nested functions and complex expressions in formulas.
# This validates FormulaCompiler.jl integration beyond simple pre-computed transformations.
#
# Phase 2.1 Requirements:
# - Test nested functions like log(1+x), exp(x-z), sqrt(x+1)  
# - Test trigonometric functions with seasonal patterns
# - Test complex econometric expressions
# - Validate derivatives for all function compositions
# - Compare against analytical derivatives where possible

using Test
using Random
using DataFrames
using CategoricalArrays
using GLM
using Statistics
using Margins

include("testing_utilities.jl")

"""
    make_function_composition_data(; n = 500, seed = 42)

Generate test data specifically for function composition testing.
Creates variables with safe domains for logarithms, square roots, and other functions.
"""
function make_function_composition_data(; n = 500, seed = 42)
    Random.seed!(seed)
    
    DataFrame(
        # Positive variables for log/sqrt functions (avoid domain issues)
        pos_x = rand(n) * 4.0 .+ 1.0,      # Range [1, 5] for log safety
        pos_income = rand(n) * 80000 .+ 20000,  # Income range [20k, 100k]
        pos_capital = rand(n) * 500 .+ 100,     # Capital range [100, 600]
        
        # Standard normal variables for general functions  
        age = rand(18:65, n),                   # Integer age
        experience = rand(0:40, n),             # Integer experience
        education = rand(6:20, n),              # Integer education
        time = rand(1:24, n),                   # Months (1-24) for seasonal patterns
        
        # Variables for interaction testing
        x = randn(n),
        z = randn(n), 
        w = randn(n),
        
        # Categorical for mixed testing
        sector = categorical(rand(["Manufacturing", "Services", "Technology"], n)),
        region = categorical(rand(["North", "South", "East", "West"], n)),
        
        # Binary outcome for logistic testing
        binary_outcome = Bool[],
        
        # Continuous outcome  
        outcome = Float64[]
    )
end

"""
    analytical_nested_derivative(expr_type, x_val, other_vals...)

Compute analytical derivatives for nested function expressions.
Used to validate FormulaCompiler.jl derivative computation.
"""
function analytical_nested_derivative(expr_type::Symbol, x_val::Float64, other_vals...)
    if expr_type == :log_1_plus_x
        # d/dx log(1 + x) = 1/(1 + x)
        return 1.0 / (1.0 + x_val)
        
    elseif expr_type == :exp_x_minus_z
        # d/dx exp(x - z) = exp(x - z), where z is held constant
        z_val = length(other_vals) > 0 ? other_vals[1] : 0.0
        return exp(x_val - z_val)
        
    elseif expr_type == :sqrt_x_plus_1
        # d/dx sqrt(x + 1) = 1/(2*sqrt(x + 1))
        return 1.0 / (2.0 * sqrt(x_val + 1.0))
        
    elseif expr_type == :x_times_log_z
        # d/dx (x * log(z)) = log(z), where z is held constant
        z_val = length(other_vals) > 0 ? other_vals[1] : exp(1.0)
        return log(z_val)
        
    elseif expr_type == :log_exp_x
        # d/dx log(exp(x)) = 1 (by logarithm properties)
        return 1.0
        
    elseif expr_type == :exp_log_x
        # d/dx exp(log(x)) = d/dx x = 1 (for x > 0)
        return 1.0
        
    elseif expr_type == :sin_pi_x
        # d/dx sin(π*x) = π*cos(π*x)
        return π * cos(π * x_val)
        
    elseif expr_type == :cos_2pi_x
        # d/dx cos(2π*x) = -2π*sin(2π*x)
        return -2.0 * π * sin(2.0 * π * x_val)
        
    else
        error("Unknown expression type: $expr_type")
    end
end

@testset "Phase 2.1: Function Composition Tests" begin

    @testset "1. Nested Function Testing" begin
        data = make_function_composition_data(n=400)
        
        @testset "1.1 Log Compositions" begin
            # Test log(1 + x) pattern
            data.outcome = 2.0 .+ 0.5 .* log.(1.0 .+ data.pos_x) .+ 0.1 .* randn(400)
            model = lm(@formula(outcome ~ log(1 + pos_x)), data)
            
            # Test all 2x2 quadrants
            quadrant_results = test_2x2_framework_quadrants(model, data; test_name="log(1+x)")
            @test quadrant_results.all_successful
            @test quadrant_results.all_finite
            
            # Validate derivative analytically for a specific point
            test_x = 2.0  # Test at x = 2
            test_data_point = DataFrame(pos_x = [test_x])
            
            # Get numerical derivative from Margins.jl
            profile_result = profile_margins(model, data, test_data_point; type=:effects, vars=[:pos_x])
            numerical_deriv = DataFrame(profile_result).estimate[1]
            
            # Get analytical derivative
            analytical_deriv = analytical_nested_derivative(:log_1_plus_x, test_x)
            model_coef = coef(model)[2]  # Coefficient of log(1+pos_x)
            expected_deriv = model_coef * analytical_deriv
            
            @test isapprox(numerical_deriv, expected_deriv; rtol=1e-6)
        end
        
        @testset "1.2 Exponential Compositions" begin
            # Test exp(x - z) pattern
            data.outcome = 1.0 .+ 0.3 .* exp.(data.x .- data.z) .+ 0.1 .* randn(400)
            model = lm(@formula(outcome ~ exp(x - z)), data)
            
            quadrant_results = test_2x2_framework_quadrants(model, data; test_name="exp(x-z)")
            @test quadrant_results.all_successful
            @test quadrant_results.all_finite
        end
        
        @testset "1.3 Square Root Compositions" begin  
            # Test sqrt(x + 1) pattern
            data.outcome = 3.0 .+ 0.7 .* sqrt.(data.pos_x .+ 1.0) .+ 0.1 .* randn(400)
            model = lm(@formula(outcome ~ sqrt(pos_x + 1)), data)
            
            quadrant_results = test_2x2_framework_quadrants(model, data; test_name="sqrt(x+1)")
            @test quadrant_results.all_successful
            @test quadrant_results.all_finite
            
            # Validate derivative for sqrt(x+1)
            test_x = 3.0
            test_data_point = DataFrame(pos_x = [test_x])
            profile_result = profile_margins(model, data, test_data_point; type=:effects, vars=[:pos_x])
            numerical_deriv = DataFrame(profile_result).estimate[1]
            
            analytical_deriv = analytical_nested_derivative(:sqrt_x_plus_1, test_x)
            model_coef = coef(model)[2]
            expected_deriv = model_coef * analytical_deriv
            
            @test isapprox(numerical_deriv, expected_deriv; rtol=1e-6)
        end
        
        @testset "1.4 Product Compositions" begin
            # Test x * log(z) pattern where derivative w.r.t. x should be log(z)
            data.outcome = 1.5 .+ 0.4 .* (data.x .* log.(data.pos_x)) .+ 0.1 .* randn(400)
            model = lm(@formula(outcome ~ x * log(pos_x)), data)
            
            quadrant_results = test_2x2_framework_quadrants(model, data; test_name="x*log(pos_x)")
            @test quadrant_results.all_successful
            @test quadrant_results.all_finite
        end
    end
    
    @testset "2. Trigonometric Function Tests" begin
        data = make_function_composition_data(n=400)
        
        @testset "2.1 Seasonal Patterns" begin
            # Test sin(π * time/12) for monthly seasonality
            seasonal_component = sin.(π .* data.time ./ 12.0)
            data.outcome = 100.0 .+ 20.0 .* seasonal_component .+ 
                          0.5 .* data.pos_income ./ 1000.0 .+ 5.0 .* randn(400)
            
            model = lm(@formula(outcome ~ sin(pi * time / 12) + pos_income), data)
            
            quadrant_results = test_2x2_framework_quadrants(model, data; test_name="seasonal_sin")
            @test quadrant_results.all_successful
            @test quadrant_results.all_finite
        end
        
        @testset "2.2 Cosine Patterns" begin
            # Test cos(2π * time/24) for bi-annual cycles
            cycle_component = cos.(2.0 * π .* data.time ./ 24.0)
            data.outcome = 50.0 .+ 15.0 .* cycle_component .+ 
                          0.3 .* data.age .+ 3.0 .* randn(400)
            
            model = lm(@formula(outcome ~ cos(2 * pi * time / 24) + age), data)
            
            quadrant_results = test_2x2_framework_quadrants(model, data; test_name="cosine_cycle")
            @test quadrant_results.all_successful
            @test quadrant_results.all_finite
        end
        
        @testset "2.3 Combined Trigonometric" begin
            # Test combination of sin and cos for complex seasonality
            data.outcome = 75.0 .+ 10.0 .* sin.(π .* data.time ./ 6.0) .+ 
                          8.0 .* cos.(π .* data.time ./ 4.0) .+ 
                          0.2 .* data.pos_income ./ 1000.0 .+ 4.0 .* randn(400)
            
            model = lm(@formula(outcome ~ sin(pi * time / 6) + cos(pi * time / 4) + pos_income), data)
            
            quadrant_results = test_2x2_framework_quadrants(model, data; test_name="combined_trig")
            @test quadrant_results.all_successful
            @test quadrant_results.all_finite
        end
    end
    
    @testset "3. Complex Econometric Expressions" begin
        data = make_function_composition_data(n=500)
        
        @testset "3.1 Production Function" begin
            # Test Cobb-Douglas: log(output) = log(A) + α*log(capital) + β*log(labor)
            # Simplified as: outcome ~ log(capital) + log(labor) with known coefficients
            log_capital = log.(data.pos_capital)
            log_labor = log.(data.pos_income ./ 2000.0)  # Treat income/2000 as labor input
            data.outcome = 2.0 .+ 0.3 .* log_capital .+ 0.6 .* log_labor .+ 0.1 .* randn(500)
            
            model = lm(@formula(outcome ~ log(pos_capital) + log(pos_income)), data)
            
            quadrant_results = test_2x2_framework_quadrants(model, data; test_name="production_function")
            @test quadrant_results.all_successful
            @test quadrant_results.all_finite
            
            # Test elasticity computation for production function
            elast_result = population_margins(model, data; type=:effects, measure=:elasticity)
            elast_df = DataFrame(elast_result)
            @test all(isfinite, elast_df.estimate)
            @test all(isfinite, elast_df.se)
        end
        
        @testset "3.2 Growth Model" begin
            # Test exp(log(capital) + log(labor)) - depreciation^2
            capital_labor = exp.(log.(data.pos_capital) .+ log.(data.pos_income ./ 1000.0))
            depreciation_sq = (data.experience ./ 20.0).^2
            data.outcome = 100.0 .+ 0.01 .* capital_labor .- 5.0 .* depreciation_sq .+ 2.0 .* randn(500)
            
            # Simplified test: just use the main components
            model = lm(@formula(outcome ~ exp(log(pos_capital) + log(pos_income)) + experience^2), data)
            
            quadrant_results = test_2x2_framework_quadrants(model, data; test_name="growth_model")
            @test quadrant_results.all_successful
            @test quadrant_results.all_finite
        end
        
        @testset "3.3 Consumption Function" begin
            # Test log(consumption) ~ log(income) + log(price_index) + interest_rate^2
            # Simulate realistic consumption data
            log_income = log.(data.pos_income)
            log_price = log.(rand(500) .+ 1.0)  # Price index [1, 2]
            interest_rate = rand(500) * 0.1  # Interest rate [0, 10%] 
            data.pos_consumption = exp.(3.0 .+ 0.8 .* log_income .+ 0.2 .* log_price .- 2.0 .* interest_rate.^2 .+ 0.1 .* randn(500))
            
            model = lm(@formula(log(pos_consumption) ~ log(pos_income) + log_price + interest_rate^2), 
                      DataFrame(data, log_price=log_price, interest_rate=interest_rate))
            
            test_data = DataFrame(data, log_price=log_price, interest_rate=interest_rate)
            quadrant_results = test_2x2_framework_quadrants(model, test_data; test_name="consumption_function")
            @test quadrant_results.all_successful
            @test quadrant_results.all_finite
        end
    end
    
    @testset "4. Function Composition Chain Rule Validation" begin
        data = make_function_composition_data(n=300)
        
        @testset "4.1 log(exp(x)) Simplification" begin
            # Test log(exp(x)) which should simplify to x with derivative = 1
            data.outcome = 1.0 .+ 0.5 .* log.(exp.(data.x)) .+ 0.1 .* randn(300)
            model = lm(@formula(outcome ~ log(exp(x))), data)
            
            # Test at a specific point  
            test_x = 1.5
            test_point = DataFrame(x = [test_x])
            profile_result = profile_margins(model, data, test_point; type=:effects, vars=[:x])
            numerical_deriv = DataFrame(profile_result).estimate[1]
            
            # Analytical derivative should be coefficient * 1
            expected_deriv = coef(model)[2] * 1.0
            @test isapprox(numerical_deriv, expected_deriv; rtol=1e-6)
        end
        
        @testset "4.2 exp(log(x)) Simplification" begin
            # Test exp(log(x)) which should simplify to x with derivative = 1
            data.outcome = 2.0 .+ 0.3 .* exp.(log.(data.pos_x)) .+ 0.1 .* randn(300)
            model = lm(@formula(outcome ~ exp(log(pos_x))), data)
            
            test_x = 2.0
            test_point = DataFrame(pos_x = [test_x])  
            profile_result = profile_margins(model, data, test_point; type=:effects, vars=[:pos_x])
            numerical_deriv = DataFrame(profile_result).estimate[1]
            
            expected_deriv = coef(model)[2] * 1.0
            @test isapprox(numerical_deriv, expected_deriv; rtol=1e-6)
        end
        
        @testset "4.3 Complex Chain Rule" begin
            # Test d/dx log(sqrt(1 + x^2)) = x / (1 + x^2)
            complex_expr = log.(sqrt.(1.0 .+ data.x.^2))
            data.outcome = 1.5 .+ 0.6 .* complex_expr .+ 0.1 .* randn(300)
            
            model = lm(@formula(outcome ~ log(sqrt(1 + x^2))), data)
            
            # Test derivative at x = 1
            test_x = 1.0
            test_point = DataFrame(x = [test_x])
            profile_result = profile_margins(model, data, test_point; type=:effects, vars=[:x])
            numerical_deriv = DataFrame(profile_result).estimate[1]
            
            # Analytical: d/dx log(sqrt(1 + x^2)) = d/dx (1/2 * log(1 + x^2)) = (1/2) * (2x)/(1 + x^2) = x/(1 + x^2)
            analytical_deriv = test_x / (1.0 + test_x^2)
            expected_deriv = coef(model)[2] * analytical_deriv
            
            @test isapprox(numerical_deriv, expected_deriv; rtol=1e-5)
        end
    end
    
    @testset "5. Backend Consistency for Function Compositions" begin
        data = make_function_composition_data(n=250)
        
        @testset "5.1 Nested Function Consistency" begin
            # Test that AD and FD agree for log(1 + x)
            data.outcome = 1.0 .+ 0.4 .* log.(1.0 .+ data.pos_x) .+ 0.1 .* randn(250)
            model = lm(@formula(outcome ~ log(1 + pos_x)), data)
            
            consistency_results = test_backend_consistency(model, data; vars=[:pos_x])
            @test consistency_results.all_consistent
            @test consistency_results.all_estimates_agree
            @test consistency_results.all_ses_agree
        end
        
        @testset "5.2 Trigonometric Consistency" begin
            # Test backend consistency for sin(π*x)
            data.outcome = 2.0 .+ 0.5 .* sin.(π .* data.time ./ 12.0) .+ 0.1 .* randn(250)
            model = lm(@formula(outcome ~ sin(pi * time / 12)), data)
            
            consistency_results = test_backend_consistency(model, data; vars=[:time])
            @test consistency_results.all_consistent
        end
        
        @testset "5.3 Complex Expression Consistency" begin
            # Test consistency for sqrt(x) + log(z)
            data.outcome = 1.5 .+ 0.3 .* sqrt.(data.pos_x) .+ 0.4 .* log.(data.pos_income ./ 1000.0) .+ 0.1 .* randn(250)
            model = lm(@formula(outcome ~ sqrt(pos_x) + log(pos_income)), data)
            
            consistency_results = test_backend_consistency(model, data; vars=[:pos_x, :pos_income])  
            @test consistency_results.all_consistent
        end
    end
    
    @testset "6. Error Handling and Domain Validation" begin
        data = make_function_composition_data(n=100)
        
        @testset "6.1 Log Domain Errors" begin
            # Create data with some negative values that would cause log domain errors
            data.unsafe_x = randn(100)  # Can be negative
            
            # This should error appropriately when log encounters negative values
            @test_throws Exception begin
                data.outcome = 1.0 .+ 0.5 .* log.(data.unsafe_x) .+ 0.1 .* randn(100)
                lm(@formula(outcome ~ log(unsafe_x)), data)
            end
        end
        
        @testset "6.2 Square Root Domain" begin  
            # Test sqrt with potentially negative values
            data.unsafe_y = randn(100) .- 2.0  # Mostly negative
            
            # This should error appropriately for negative square root arguments
            @test_throws Exception begin
                data.outcome = 1.0 .+ 0.5 .* sqrt.(data.unsafe_y) .+ 0.1 .* randn(100)
                lm(@formula(outcome ~ sqrt(unsafe_y)), data)
            end
        end
    end
end

# Export test utilities for potential use in other test files
export make_function_composition_data, analytical_nested_derivative