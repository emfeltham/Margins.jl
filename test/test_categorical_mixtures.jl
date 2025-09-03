using Test
using Random
using DataFrames, GLM, StatsModels, CategoricalArrays
using Margins

@testset "CategoricalMixture Construction and Validation" begin
    @testset "Valid Construction" begin
        # Basic construction
        m1 = mix("A" => 0.3, "B" => 0.7)
        @test m1.levels == ["A", "B"]
        @test m1.weights == [0.3, 0.7]
        @test m1 isa CategoricalMixture{String}
        
        # Symbol levels
        m2 = mix(:urban => 0.6, :rural => 0.4)
        @test m2.levels == [:urban, :rural]
        @test m2.weights == [0.6, 0.4]
        @test m2 isa CategoricalMixture{Symbol}
        
        # Bool levels
        m3 = mix(true => 0.8, false => 0.2)
        @test m3.levels == [true, false]
        @test m3.weights == [0.8, 0.2]
        @test m3 isa CategoricalMixture{Bool}
        
        # Multiple levels
        m4 = mix("low" => 0.2, "medium" => 0.5, "high" => 0.3)
        @test length(m4.levels) == 3
        @test sum(m4.weights) ≈ 1.0
        
        # Direct constructor
        m5 = CategoricalMixture(["X", "Y"], [0.4, 0.6])
        @test m5.levels == ["X", "Y"]
        @test m5.weights == [0.4, 0.6]
    end
    
    @testset "Invalid Construction" begin
        # Weights don't sum to 1
        @test_throws ArgumentError mix("A" => 0.3, "B" => 0.8)
        @test_throws ArgumentError mix("A" => 0.1, "B" => 0.1)
        
        # Negative weights
        @test_throws ArgumentError mix("A" => -0.1, "B" => 1.1)
        @test_throws ArgumentError mix("A" => 0.5, "B" => -0.5)
        
        # Duplicate levels
        @test_throws ArgumentError mix("A" => 0.3, "A" => 0.7)
        @test_throws ArgumentError CategoricalMixture(["X", "X"], [0.5, 0.5])
        
        # Length mismatch
        @test_throws ArgumentError CategoricalMixture(["A", "B"], [0.3])
        @test_throws ArgumentError CategoricalMixture(["A"], [0.3, 0.7])
        
        # Empty mixture
        @test_throws ArgumentError mix()
    end
    
    @testset "Display and Iteration" begin
        m = mix("high" => 0.3, "low" => 0.7)
        
        # String representation
        str = string(m)
        @test occursin("mix(", str)
        @test occursin("\"high\" => 0.3", str)
        @test occursin("\"low\" => 0.7", str)
        
        # Length and indexing
        @test length(m) == 2
        @test m[1] == ("high", 0.3)
        @test m[2] == ("low", 0.7)
        
        # Iteration
        pairs = collect(m)
        @test pairs == [("high", 0.3), ("low", 0.7)]
    end
end

@testset "Mixture Validation Against Data" begin
    @testset "CategoricalArray Validation" begin
        df = DataFrame(
            education = categorical(["high_school", "college", "graduate"]),
            y = [1, 0, 1]
        )
        
        # Valid mixture
        valid_mix = mix("high_school" => 0.4, "college" => 0.6)
        @test Margins._validate_mixture_against_data(valid_mix, df.education, :education) == true
        
        # Invalid mixture - level not in data
        invalid_mix = mix("high_school" => 0.4, "phd" => 0.6)
        @test_throws ArgumentError Margins._validate_mixture_against_data(invalid_mix, df.education, :education)
        
        # All levels valid
        complete_mix = mix("high_school" => 0.3, "college" => 0.4, "graduate" => 0.3)
        @test Margins._validate_mixture_against_data(complete_mix, df.education, :education) == true
    end
    
    @testset "Bool Validation" begin
        df = DataFrame(
            treated = [true, false, true, false],
            y = [1, 0, 1, 0]
        )
        
        # Valid Bool mixtures
        bool_mix1 = mix(true => 0.6, false => 0.4)
        @test Margins._validate_mixture_against_data(bool_mix1, df.treated, :treated) == true
        
        bool_mix2 = mix("true" => 0.3, "false" => 0.7)
        @test Margins._validate_mixture_against_data(bool_mix2, df.treated, :treated) == true
        
        # Invalid Bool mixture
        invalid_bool = mix("yes" => 0.5, "no" => 0.5)
        @test_throws ArgumentError Margins._validate_mixture_against_data(invalid_bool, df.treated, :treated)
    end
    
    @testset "Generic String Validation" begin
        df = DataFrame(
            region = ["north", "south", "east", "west"],
            y = [1, 0, 1, 0]
        )
        
        # Valid string mixture
        region_mix = mix("north" => 0.25, "south" => 0.75)
        @test Margins._validate_mixture_against_data(region_mix, df.region, :region) == true
        
        # Invalid string mixture
        invalid_region = mix("north" => 0.5, "central" => 0.5)
        @test_throws ArgumentError Margins._validate_mixture_against_data(invalid_region, df.region, :region)
    end
end

@testset "Mixture to Scenario Value Conversion" begin
    @testset "CategoricalArray Conversion" begin
        education_levels = ["high_school", "college", "graduate"]  
        df = DataFrame(education = categorical(education_levels))
        
        # Test weighted average computation
        # Levels are alphabetically ordered: college=1, graduate=2, high_school=3
        edu_mix = mix("high_school" => 0.5, "college" => 0.3, "graduate" => 0.2)
        scenario_val = Margins._mixture_to_scenario_value(edu_mix, df.education)
        
        expected = 0.5 * 3 + 0.3 * 1 + 0.2 * 2  # 1.5 + 0.3 + 0.4 = 2.2
        @test scenario_val ≈ expected
        
        # Edge case: single level
        single_mix = mix("college" => 1.0)
        single_val = Margins._mixture_to_scenario_value(single_mix, df.education)
        @test single_val ≈ 1.0  # college is index 1 (alphabetically first)
    end
    
    @testset "Bool Conversion" begin
        df = DataFrame(treated = [true, false, true])
        
        # Test probability of true
        bool_mix1 = mix(true => 0.3, false => 0.7)
        prob_true = Margins._mixture_to_scenario_value(bool_mix1, df.treated)
        @test prob_true ≈ 0.3
        
        bool_mix2 = mix("true" => 0.8, "false" => 0.2)  
        prob_true2 = Margins._mixture_to_scenario_value(bool_mix2, df.treated)
        @test prob_true2 ≈ 0.8
        
        # Edge cases
        all_true = mix(true => 1.0, false => 0.0)
        @test Margins._mixture_to_scenario_value(all_true, df.treated) ≈ 1.0
        
        all_false = mix(true => 0.0, false => 1.0)
        @test Margins._mixture_to_scenario_value(all_false, df.treated) ≈ 0.0
    end
    
    @testset "Generic String Conversion" begin
        df = DataFrame(region = ["A", "B", "C", "A", "B"])
        
        # Test weighted average with sorted levels: A=1, B=2, C=3 (alphabetical order)
        region_mix = mix("A" => 0.2, "B" => 0.3, "C" => 0.5)
        scenario_val = Margins._mixture_to_scenario_value(region_mix, df.region)
        
        expected = 0.2 * 1 + 0.3 * 2 + 0.5 * 3  # 0.2 + 0.6 + 1.5 = 2.3
        @test scenario_val ≈ expected
    end
end

@testset "Integration with Margins Workflow" begin
    @testset "Basic Functionality Test" begin
        # This is a smoke test to ensure the mixtures module loads and basic operations work
        # Full integration testing will be in Phase 2
        
        Random.seed!(12345)
        n = 50
        df = DataFrame(
            education = categorical(rand(["high_school", "college", "graduate"], n)),
            income = randn(n) * 10000 .+ 50000,
            y = rand(Bool, n)
        )
        
        # Test mixture creation with real data
        edu_mix = mix("high_school" => 0.4, "college" => 0.4, "graduate" => 0.2)
        @test edu_mix isa CategoricalMixture
        
        # Test validation against real data
        @test Margins._validate_mixture_against_data(edu_mix, df.education, :education) == true
        
        # Test scenario value computation
        scenario_val = Margins._mixture_to_scenario_value(edu_mix, df.education)
        @test isa(scenario_val, Real)
        @test isfinite(scenario_val)
    end
    
    @testset "profile_margins() Integration" begin
        Random.seed!(42)
        n = 100
        df = DataFrame(
            education = categorical(rand(["high_school", "college", "graduate"], n)),
            region = categorical(rand(["urban", "rural"], n)),
            age = rand(25:65, n),
            income = randn(n) * 10000 .+ 50000,
            employed = rand(Bool, n),
            outcome = randn(n) * 2 .+ 10
        )
        
        # Add some realistic relationships
        df.outcome += 0.001 * (df.income .- 50000) + 0.02 * (df.age .- 45)
        
        # Fit models
        simple_model = lm(@formula(outcome ~ education + age), df)
        interaction_model = lm(@formula(outcome ~ education * region + age + income + employed), df)
        
        @testset "Single Categorical Mixture" begin
            edu_mix = mix("high_school" => 0.3, "college" => 0.5, "graduate" => 0.2)
            
            # Test predictions
            result_pred = profile_margins(simple_model, df; at=Dict(:education => edu_mix), type=:predictions)
            @test result_pred isa MarginsResult
            @test nrow(DataFrame(result_pred)) == 1
            @test "estimate" in names(DataFrame(result_pred))
            @test isfinite(DataFrame(result_pred).estimate[1])
            
            # Test effects
            result_eff = profile_margins(simple_model, df; at=Dict(:education => edu_mix), type=:effects)
            @test result_eff isa MarginsResult
            @test nrow(DataFrame(result_eff)) >= 1  # At least age effect
            @test all(isfinite, DataFrame(result_eff).estimate)
        end
        
        @testset "Weighted Contrast Accuracy" begin
            edu_mix = mix("high_school" => 0.4, "college" => 0.4, "graduate" => 0.2)
            
            # Get mixture result
            mixture_result = profile_margins(simple_model, df; at=Dict(:education => edu_mix), type=:predictions)
            mixture_pred = DataFrame(mixture_result).estimate[1]
            
            # Get individual level results
            hs_result = profile_margins(simple_model, df; at=Dict(:education => "high_school"), type=:predictions)
            col_result = profile_margins(simple_model, df; at=Dict(:education => "college"), type=:predictions)
            grad_result = profile_margins(simple_model, df; at=Dict(:education => "graduate"), type=:predictions)
            
            # Manual weighted combination
            expected = 0.4 * DataFrame(hs_result).estimate[1] + 0.4 * DataFrame(col_result).estimate[1] + 0.2 * DataFrame(grad_result).estimate[1]
            
            # Should match exactly (within floating point precision)
            @test abs(mixture_pred - expected) < 1e-12
        end
        
        @testset "Boolean Mixture Support" begin
            # Test with CategoricalArray{Bool}
            df_cat_bool = copy(df)
            df_cat_bool.employed = categorical(df_cat_bool.employed)
            model_cat_bool = lm(@formula(outcome ~ employed + age), df_cat_bool)
            
            bool_mix = mix(true => 0.7, false => 0.3)
            result_cat = profile_margins(model_cat_bool, df_cat_bool; at=Dict(:employed => bool_mix), type=:predictions)
            @test result_cat isa MarginsResult
            @test isfinite(DataFrame(result_cat).estimate[1])
            
            # Test with regular Vector{Bool}
            bool_result = profile_margins(interaction_model, df; at=Dict(:employed => bool_mix), type=:predictions)
            @test bool_result isa MarginsResult
            @test isfinite(DataFrame(bool_result).estimate[1])
            
            # Verify weighted combination for regular Bool
            emp_true = profile_margins(interaction_model, df; at=Dict(:employed => true), type=:predictions).estimate[1]
            emp_false = profile_margins(interaction_model, df; at=Dict(:employed => false), type=:predictions).estimate[1]
            expected_bool = 0.7 * emp_true + 0.3 * emp_false
            @test abs(DataFrame(bool_result).estimate[1] - expected_bool) < 1e-12
        end
        
        @testset "Multiple Categorical Mixtures" begin
            edu_mix = mix("high_school" => 0.3, "college" => 0.5, "graduate" => 0.2)
            region_mix = mix("urban" => 0.6, "rural" => 0.4)
            emp_mix = mix(true => 0.8, false => 0.2)
            
            # Test multiple mixtures together
            result = profile_margins(interaction_model, df; 
                at=Dict(:education => edu_mix, :region => region_mix, :employed => emp_mix),
                type=:predictions
            )
            @test result isa MarginsResult
            @test nrow(DataFrame(result)) == 1
            @test isfinite(DataFrame(result).estimate[1])
            
            # Test with mixture + continuous overrides
            result_mixed = profile_margins(interaction_model, df;
                at=Dict(
                    :education => edu_mix,
                    :age => 40,
                    :income => 60000
                ),
                type=:predictions
            )
            @test result_mixed isa MarginsResult
            @test isfinite(DataFrame(result_mixed).estimate[1])
        end
        
        @testset "Effects with Mixtures" begin
            edu_mix = mix("high_school" => 0.4, "college" => 0.4, "graduate" => 0.2)
            
            result = profile_margins(interaction_model, df; at=Dict(:education => edu_mix), type=:effects)
            @test result isa MarginsResult
            @test nrow(DataFrame(result)) >= 1  # At least one effect (age, income, or employed)
            @test "term" in names(DataFrame(result))
            @test "estimate" in names(DataFrame(result))
            @test all(isfinite, DataFrame(result).estimate)
            @test all(isfinite, DataFrame(result).se)
        end
        
        @testset "Error Handling" begin
            # Invalid mixture levels
            invalid_mix = mix("high_school" => 0.3, "phd" => 0.7)  # "phd" not in levels
            @test_throws ArgumentError profile_margins(simple_model, df; at=Dict(:education => invalid_mix), type=:predictions)
            
            # Invalid weights (don't sum to 1)
            @test_throws ArgumentError mix("high_school" => 0.3, "college" => 0.8)
            
            # Empty mixture
            @test_throws ArgumentError mix()
        end
    end
end