#!/usr/bin/env julia
# Test automatic variable detection logic to prevent regression of dependent variable bug

using Test
using Margins
using GLM, DataFrames
using CategoricalArrays
using Random

@testset "Automatic Variable Detection" begin
    Random.seed!(12345)
    
    @testset "Continuous dependent variable should be excluded" begin
        # This is the exact scenario that was failing before the fix
        df = DataFrame(
            y = randn(100),      # Continuous dependent variable
            x1 = randn(100),     # Continuous explanatory
            x2 = randn(100),     # Continuous explanatory
            group = categorical(repeat(["A", "B"], 50))  # Categorical explanatory
        )
        
        model = lm(@formula(y ~ x1 + x2 + group), df)
        
        # Test population_margins with automatic variable detection
        result = population_margins(model, df; type=:effects)
        df_result = DataFrame(result)
        
        # Should include both continuous variables (x1, x2) AND categorical baseline contrasts (group)
        @test nrow(df_result) == 3  # x1, x2, and group baseline contrast
        @test "x1" ∈ df_result.variable
        @test "x2" ∈ df_result.variable  
        @test "group" ∈ df_result.variable  # Categorical included with baseline contrast
        @test "y" ∉ df_result.variable  # Dependent variable should be excluded
    end
    
    @testset "Profile margins with automatic detection" begin
        df = DataFrame(
            y = randn(50),
            x1 = randn(50),
            x2 = randn(50)
        )
        
        model = lm(@formula(y ~ x1 + x2), df)
        
        # Should work without error - was failing before fix  
        # Create reference grid excluding dependent variable y
        df_explanatory = select(df, Not(:y))
        result = profile_margins(model, df, means_grid(df_explanatory); type=:effects)
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 2
        @test any(occursin.("x1", df_result.variable))
        @test any(occursin.("x2", df_result.variable))
        @test all(.!occursin.("y", df_result.variable))
    end
    
    @testset "Mixed variable types with automatic detection" begin
        df = DataFrame(
            revenue = randn(100) .+ 1000,    # Continuous dependent
            price = randn(100) .+ 50,        # Continuous explanatory  
            quantity = rand(1:100, 100),     # Int64 explanatory (should be included)
            available = rand(Bool, 100),     # Bool explanatory (categorical)
            category = categorical(repeat(["X", "Y", "Z"], 34)[1:100])  # Categorical
        )
        
        model = lm(@formula(revenue ~ price + quantity + available + category), df)
        result = population_margins(model, df; type=:effects)
        df_result = DataFrame(result)
        
        # Should include: price (Float64), quantity (Int64), available (Bool baseline), category (baseline contrasts)
        # Should exclude: revenue (dependent variable)
        @test nrow(df_result) == 5  # price, quantity, available, category(Y vs X), category(Z vs X)
        @test "price" ∈ df_result.variable      # Float64 continuous
        @test "quantity" ∈ df_result.variable   # Int64 treated as continuous  
        @test "available" ∈ df_result.variable  # Bool with baseline contrast
        @test "category" ∈ df_result.variable   # Categorical with baseline contrasts
        @test "revenue" ∉ df_result.variable    # Dependent variable excluded
    end
    
    @testset "All categorical explanatory variables" begin
        df = DataFrame(
            y = randn(100),
            cat1 = categorical(repeat(["A", "B"], 50)),
            cat2 = rand(Bool, 100),
            cat3 = categorical(repeat(["X", "Y", "Z", "W"], 25))  # Clearly categorical
        )
        
        model = lm(@formula(y ~ cat1 + cat2 + cat3), df)
        
        # Should work fine and return baseline contrasts for all categorical variables
        result = population_margins(model, df; type=:effects)
        df_result = DataFrame(result)
        
        # Should have baseline contrasts for all categorical variables
        @test nrow(df_result) >= 3  # At least one contrast per categorical variable
        @test "cat1" ∈ df_result.variable
        @test "cat2" ∈ df_result.variable  
        @test "cat3" ∈ df_result.variable
    end
    
    @testset "Explicit vars parameter bypasses automatic detection" begin
        df = DataFrame(
            y = randn(50),
            x1 = randn(50),
            x2 = randn(50)
        )
        
        model = lm(@formula(y ~ x1 + x2), df)
        
        # Even with y in the data, explicit vars should work
        result = population_margins(model, df; type=:effects, vars=[:x1])
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 1
        @test df_result.variable[1] == "x1"
        
        # Should also work with explicit vars including non-existent vars (should error appropriately)
        @test_throws Margins.MarginsError population_margins(model, df; type=:effects, vars=[:nonexistent])
    end
    
    @testset "Standard error consistency after fix" begin
        # Test the original issue from the bug report
        Random.seed!(06515)
        df = DataFrame(
            y = randn(1000),
            x1 = randn(1000),
            x2 = randn(1000),
            group = categorical(repeat(["A", "B", "C"], 334)[1:1000])
        )
        df.y = 0.5 * df.x1 + 0.3 * df.x2 + randn(1000) * 0.1
        
        model = lm(@formula(y ~ x1 + x2 + group), df)
        
        # Get coefficient standard errors
        coef_table = DataFrame(coeftable(model))
        x1_coef_se = coef_table[coef_table[!, "Name"] .== "x1", "Std. Error"][1]
        x2_coef_se = coef_table[coef_table[!, "Name"] .== "x2", "Std. Error"][1]
        
        # Get marginal effect standard errors (should be same for linear model)
        result = population_margins(model, df; type=:effects)
        df_result = DataFrame(result)
        
        x1_row = df_result[df_result.variable .== "x1", :]
        x2_row = df_result[df_result.variable .== "x2", :]
        
        # Standard errors should be approximately equal (not 1000x different!)
        @test isapprox(x1_row.se[1], x1_coef_se, rtol=1e-10)  # High precision tolerance for linear model specification
        @test isapprox(x2_row.se[1], x2_coef_se, rtol=1e-10)
        
        # Verify estimates match coefficients too
        x1_coef_est = coef_table[coef_table[!, "Name"] .== "x1", "Coef."][1]
        x2_coef_est = coef_table[coef_table[!, "Name"] .== "x2", "Coef."][1]
        
        @test isapprox(x1_row.estimate[1], x1_coef_est, rtol=1e-10)
        @test isapprox(x2_row.estimate[1], x2_coef_est, rtol=1e-10)
    end
end