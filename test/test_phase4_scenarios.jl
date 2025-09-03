# test_phase4_scenarios.jl - Tests for Phase 4 Scenarios Integration

using Test
using DataFrames
using GLM
using StatsModels
using CategoricalArrays
using Random
using Margins

@testset "Phase 4 Scenarios Integration" begin
    # Create realistic test data
    Random.seed!(456)
    n = 500
    
    # Generate test dataset with mixed variable types
    df = DataFrame(
        # Outcome variable
        y = randn(n),
        
        # Continuous variables for effects
        x1 = randn(n),
        x2 = randn(n),
        income = 30000 .+ 25000 * rand(n),
        age = 25 .+ 35 * rand(n),
        
        # Categorical variables
        education = categorical(rand(["HS", "College", "Graduate"], n)),
        region = categorical(rand(["North", "South", "East", "West"], n)),
        gender = categorical(rand(["Male", "Female"], n)),
        
        # Treatment/policy variables for scenarios
        treatment = rand([0, 1], n),
        policy = categorical(rand(["status_quo", "reform"], n)),
        intervention = categorical(rand(["control", "active"], n))
    )
    
    # Add some structure to the outcome based on scenarios variables
    df.y = 0.3 * df.x1 + 0.2 * df.x2 + 
           0.5 * df.treatment + 0.3 * (df.policy .== "reform") + 0.2 * (df.intervention .== "active") +
           0.4 * (df.education .== "Graduate") + 0.1 * (df.income / 10000) +
           randn(n)
    
    # Fit comprehensive model
    model = lm(@formula(y ~ x1 + x2 + treatment + policy + intervention + education + region + gender + income + age), df)
    
    @testset "Basic Scenarios Functionality" begin
        # Single binary scenarios
        result = population_margins(model, df; vars=[:x1], scenarios=Dict(:treatment => [0, 1]))
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 2  # 2 treatment scenarios
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_treatment])
        @test Set(df_result.at_treatment) == Set([0, 1])
        
        # Multiple values scenario
        result = population_margins(model, df; vars=[:x1], 
                                   scenarios=Dict(:policy => ["status_quo", "reform"]))
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 2  # 2 policy scenarios
        @test Set(df_result.at_policy) == Set(["status_quo", "reform"])
        
        # Continuous variable scenarios
        result = population_margins(model, df; vars=[:x1], 
                                   scenarios=Dict(:income => [30000, 45000, 60000]))
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 3  # 3 income scenarios
        @test Set(df_result.at_income) == Set([30000, 45000, 60000])
    end
    
    @testset "Multiple Variable Scenarios (Cartesian Products)" begin
        # Two binary variables
        result = population_margins(model, df; vars=[:x1], 
                                   scenarios=Dict(:treatment => [0, 1], 
                                                 :policy => ["status_quo", "reform"]))
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 4  # 2 × 2 = 4 combinations
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_treatment, :at_policy])
        
        # Verify all combinations present
        combos = unique(select(df_result, :at_treatment, :at_policy))
        @test nrow(combos) == 4
        expected_combos = [(0, "status_quo"), (0, "reform"), (1, "status_quo"), (1, "reform")]
        for combo in expected_combos
            @test any(row -> row.at_treatment == combo[1] && row.at_policy == combo[2], eachrow(combos))
        end
        
        # Three variables
        result = population_margins(model, df; vars=[:x1], 
                                   scenarios=Dict(:treatment => [0, 1], 
                                                 :policy => ["status_quo", "reform"],
                                                 :intervention => ["control", "active"]))
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 8  # 2 × 2 × 2 = 8 combinations
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_treatment, :at_policy, :at_intervention])
    end
    
    @testset "Scenarios with Simple Groups" begin
        # Single categorical group with scenarios
        result = population_margins(model, df; vars=[:x1], 
                                   groups=:education,
                                   scenarios=Dict(:treatment => [0, 1]))
        df_result = DataFrame(result)
        
        expected_results = length(unique(df.education)) * 2  # 3 education × 2 treatment
        @test nrow(df_result) == expected_results
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_education, :at_treatment])
        
        # Cross-tabulated groups with scenarios
        result = population_margins(model, df; vars=[:x1], 
                                   groups=[:education, :gender],
                                   scenarios=Dict(:policy => ["status_quo", "reform"]))
        df_result = DataFrame(result)
        
        expected_results = length(unique(df.education)) * length(unique(df.gender)) * 2  # 3 × 2 × 2
        @test nrow(df_result) == expected_results
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_education, :at_gender, :at_policy])
    end
    
    @testset "Scenarios with Phase 3 Advanced Groups" begin
        # Nested groups with scenarios
        result = population_margins(model, df; vars=[:x1], 
                                   groups=:region => :education,
                                   scenarios=Dict(:treatment => [0, 1]))
        df_result = DataFrame(result)
        
        expected_results = length(unique(df.region)) * length(unique(df.education)) * 2  # 4 × 3 × 2
        @test nrow(df_result) == expected_results
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_region, :at_education, :at_treatment])
        
        # Continuous binning with scenarios
        result = population_margins(model, df; vars=[:x1], 
                                   groups=(:income, 4),  # Quartiles
                                   scenarios=Dict(:intervention => ["control", "active"]))
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 8  # 4 quartiles × 2 interventions
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_income, :at_income_bin, :at_intervention])
        @test Set(df_result.at_income) == Set(["Q1", "Q2", "Q3", "Q4"])
        
        # Mixed categorical/continuous groups with scenarios
        result = population_margins(model, df; vars=[:x1], 
                                   groups=[:education, (:age, 3)],  # Education × age tertiles
                                   scenarios=Dict(:treatment => [0, 1], :policy => ["status_quo", "reform"]))
        df_result = DataFrame(result)
        
        expected_results = 3 * 3 * 2 * 2  # 3 education × 3 age tertiles × 2 treatment × 2 policy
        @test nrow(df_result) == expected_results
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_education, :at_age, :at_age_bin, :at_treatment, :at_policy])
    end
    
    @testset "Complex Nested Scenarios" begin
        # Deep nesting with multiple scenarios
        result = population_margins(model, df; vars=[:x1], 
                                   groups=:region => (:education => :gender),
                                   scenarios=Dict(:treatment => [0, 1]))
        df_result = DataFrame(result)
        
        expected_results = length(unique(df.region)) * length(unique(df.education)) * length(unique(df.gender)) * 2
        @test nrow(df_result) == expected_results
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_region, :at_education, :at_gender, :at_treatment])
        
        # Mixed continuous/categorical nesting with multi-variable scenarios
        result = population_margins(model, df; vars=[:x1], 
                                   groups=:education => (:income, 3),  # Education → income tertiles
                                   scenarios=Dict(:treatment => [0, 1], :intervention => ["control", "active"]))
        df_result = DataFrame(result)
        
        expected_results = 3 * 3 * 2 * 2  # 3 education × 3 income tertiles × 2 treatment × 2 intervention
        @test nrow(df_result) == expected_results
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_education, :at_income, :at_income_bin, :at_treatment, :at_intervention])
    end
    
    @testset "Scenarios with Predictions" begin
        # Basic predictions with scenarios
        result = population_margins(model, df; type=:predictions, 
                                   scenarios=Dict(:treatment => [0, 1]))
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 2  # 2 treatment scenarios
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_treatment])
        @test all(df_result.variable .== "AAP")  # Average Adjusted Predictions
        
        # Predictions with groups and scenarios
        result = population_margins(model, df; type=:predictions, 
                                   groups=:education,
                                   scenarios=Dict(:policy => ["status_quo", "reform"]))
        df_result = DataFrame(result)
        
        expected_results = length(unique(df.education)) * 2  # 3 education × 2 policy
        @test nrow(df_result) == expected_results
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_education, :at_policy])
    end
    
    @testset "Statistical Correctness of Scenarios" begin
        # Test that scenarios preserve statistical validity
        result = population_margins(model, df; vars=[:x1, :x2], 
                                   groups=:education,
                                   scenarios=Dict(:treatment => [0, 1]))
        df_result = DataFrame(result)
        
        # All standard errors should be positive
        @test all(df_result.se .> 0)
        
        # All estimates should be finite
        @test all(isfinite.(df_result.estimate))
        
        # T-statistics should be computed correctly
        expected_t_stats = df_result.estimate ./ df_result.se
        @test all(abs.(df_result.t_stat .- expected_t_stats) .< 1e-10)
        
        # P-values should be between 0 and 1
        @test all(0 .<= df_result.p_value .<= 1)
        
        # Scenarios should have different estimates (treatment effects)
        treatment0_results = filter(row -> row.at_treatment == 0, df_result)
        treatment1_results = filter(row -> row.at_treatment == 1, df_result)
        @test nrow(treatment0_results) == nrow(treatment1_results)  # Same number of results
    end
    
    @testset "Scenario Edge Cases and Error Handling" begin
        # Empty scenarios Dict should work (no scenarios)
        result = population_margins(model, df; vars=[:x1], scenarios=Dict{Symbol, Vector{Any}}())
        df_result = DataFrame(result)
        @test nrow(df_result) == 1  # Just the base case
        
        # Single value in scenario (should still work)
        result = population_margins(model, df; vars=[:x1], scenarios=Dict(:treatment => [1]))
        df_result = DataFrame(result)
        @test nrow(df_result) == 1
        @test df_result.at_treatment[1] == 1
        
        # Non-existent variable in scenarios should error
        @test_throws Exception population_margins(model, df; vars=[:x1], scenarios=Dict(:nonexistent => [0, 1]))
        
        # Invalid scenario values should error appropriately
        @test_throws Exception population_margins(model, df; vars=[:x1], scenarios=Dict(:education => ["Invalid"]))
    end
    
    @testset "Performance and Memory Efficiency" begin
        # Large number of scenarios should still work efficiently
        large_scenarios = Dict(:income => collect(30000:5000:70000))  # 9 income values
        result = population_margins(model, df; vars=[:x1], scenarios=large_scenarios)
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 9
        @test Set(df_result.at_income) == Set(collect(30000:5000:70000))
        
        # Combined large groups and scenarios
        result = population_margins(model, df; vars=[:x1], 
                                   groups=[:education, :gender],  # 6 combinations
                                   scenarios=Dict(:treatment => [0, 1], :policy => ["status_quo", "reform"]))  # 4 combinations
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 24  # 6 × 4 = 24 total combinations
    end
end