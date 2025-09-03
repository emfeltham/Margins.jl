# test_phase3_grouping.jl - Tests for Phase 3 Population Grouping Framework

using Test
using DataFrames
using GLM
using StatsModels
using CategoricalArrays
using Random
using Margins

@testset "Phase 3 Population Grouping Framework" begin
    # Create realistic test data
    Random.seed!(123)
    n = 1000
    
    # Generate test dataset with mixed variable types
    df = DataFrame(
        # Outcome variable
        y = randn(n),
        
        # Continuous variables
        income = 30000 .+ 20000 * rand(n),
        age = 25 .+ 35 * rand(n),
        
        # Categorical variables  
        education = categorical(rand(["High School", "College", "Graduate"], n)),
        region = categorical(rand(["North", "South", "East", "West"], n)),
        gender = categorical(rand(["Male", "Female"], n)),
        
        # Treatment variable for scenarios
        treatment = rand([0, 1], n)
    )
    
    # Add some structure to the outcome
    df.y = 0.5 * (df.income / 10000) + 
           0.3 * df.age + 
           0.4 * (df.education .== "Graduate") + 
           0.2 * df.treatment + 
           randn(n)
    
    # Fit a simple linear model
    model = lm(@formula(y ~ income + age + education + region + gender + treatment), df)
    
    @testset "Basic Grouping (Phase 1/2 - should still work)" begin
        # Simple categorical grouping
        result = population_margins(model, df; vars=[:income], groups=:education)
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 3  # 3 education levels
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_education])
        @test Set(df_result.at_education) == Set(["High School", "College", "Graduate"])
        
        # Cross-tabulation
        result = population_margins(model, df; vars=[:income], groups=[:education, :gender])
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 6  # 3 education × 2 gender
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_education, :at_gender])
    end
    
    @testset "Phase 3: Nested Grouping with => Operator" begin
        # Simple nested grouping: region => education
        result = population_margins(model, df; vars=[:income], groups=:region => :education)
        df_result = DataFrame(result)
        
        # Should have region×education combinations
        expected_combinations = length(unique(df.region)) * length(unique(df.education))
        @test nrow(df_result) == expected_combinations
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_region, :at_education])
        
        # Verify all region-education combinations are present
        unique_combos = unique(select(df_result, :at_region, :at_education))
        @test nrow(unique_combos) == expected_combinations
        
        # Nested with cross-tabulation: region => [:education, :gender]
        result = population_margins(model, df; vars=[:income], groups=:region => [:education, :gender])
        df_result = DataFrame(result)
        
        expected_combinations = length(unique(df.region)) * length(unique(df.education)) * length(unique(df.gender))
        @test nrow(df_result) == expected_combinations
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_region, :at_education, :at_gender])
    end
    
    @testset "Phase 3: Continuous Binning" begin
        # Quartiles
        result = population_margins(model, df; vars=[:age], groups=(:income, 4))
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 4  # 4 quartiles
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_income_bin])
        @test Set(df_result.at_income_bin) == Set([1, 2, 3, 4])
        
        # Tertiles
        result = population_margins(model, df; vars=[:age], groups=(:income, 3))
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 3  # 3 tertiles
        @test Set(df_result.at_income_bin) == Set([1, 2, 3])
        
        # Custom thresholds
        thresholds = [35000, 45000, 55000]
        result = population_margins(model, df; vars=[:age], groups=(:income, thresholds))
        df_result = DataFrame(result)
        
        @test nrow(df_result) == 4  # 4 groups from 3 thresholds
        @test Set(df_result.at_income_bin) == Set([1, 2, 3, 4])
    end
    
    @testset "Phase 3: Mixed Categorical and Continuous Patterns" begin
        # Categorical × continuous binning
        result = population_margins(model, df; vars=[:age], groups=[:education, (:income, 4)])
        df_result = DataFrame(result)
        
        expected_combinations = length(unique(df.education)) * 4  # 3 education × 4 quartiles
        @test nrow(df_result) == expected_combinations
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_education, :at_income_bin])
        
        # Nested: categorical => continuous
        result = population_margins(model, df; vars=[:age], groups=:education => (:income, 3))
        df_result = DataFrame(result)
        
        expected_combinations = length(unique(df.education)) * 3  # 3 education × 3 tertiles
        @test nrow(df_result) == expected_combinations
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_education, :at_income_bin])
        
        # Nested: continuous => categorical
        result = population_margins(model, df; vars=[:income], groups=(:age, 3) => :gender)
        df_result = DataFrame(result)
        
        expected_combinations = 3 * length(unique(df.gender))  # 3 age tertiles × 2 genders
        @test nrow(df_result) == expected_combinations
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_age_bin, :at_gender])
    end
    
    @testset "Phase 3: Complex Nested Patterns" begin
        # Three-level nesting: region => education => gender
        result = population_margins(model, df; vars=[:income], groups=:region => (:education => :gender))
        df_result = DataFrame(result)
        
        expected_combinations = length(unique(df.region)) * length(unique(df.education)) * length(unique(df.gender))
        @test nrow(df_result) == expected_combinations
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_region, :at_education, :at_gender])
        
        # Mixed nesting with continuous binning
        result = population_margins(model, df; vars=[:age], groups=:region => [(:income, 3), :education])
        df_result = DataFrame(result)
        
        # Should have region × (income_tertiles + education_levels) combinations
        expected_combinations = length(unique(df.region)) * (3 + length(unique(df.education)))
        @test nrow(df_result) == expected_combinations
    end
    
    @testset "Statistical Correctness" begin
        # Test that results are statistically valid
        result = population_margins(model, df; vars=[:income], groups=:education => (:age, 4))
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
    end
    
    @testset "Error Handling" begin
        # Invalid nested syntax
        @test_throws ArgumentError population_margins(model, df; vars=[:income], groups=(:invalid, :syntax))
        
        # Invalid continuous specification
        @test_throws ArgumentError population_margins(model, df; vars=[:income], groups=(:income, 0))
        @test_throws ArgumentError population_margins(model, df; vars=[:income], groups=(:income, -1))
        
        # Non-existent variables
        @test_throws Exception population_margins(model, df; vars=[:income], groups=:nonexistent_var)
    end
    
    @testset "Scenarios Integration (Phase 4 Preview)" begin
        # Basic scenarios with grouping
        result = population_margins(model, df; 
                                  vars=[:income], 
                                  groups=:education,
                                  scenarios=Dict(:treatment => [0, 1]))
        df_result = DataFrame(result)
        
        # Should have education × treatment scenarios
        expected_combinations = length(unique(df.education)) * 2
        @test nrow(df_result) == expected_combinations
        @test all(col -> col in names(df_result), [:variable, :estimate, :se, :at_education, :at_treatment])
    end
end