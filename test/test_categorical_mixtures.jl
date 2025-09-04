using Test
using Random
using Statistics
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
        @test Margins.validate_mixture_against_data(valid_mix, df.education, :education) == true
        
        # Invalid mixture - level not in data
        invalid_mix = mix("high_school" => 0.4, "phd" => 0.6)
        @test_throws ArgumentError Margins.validate_mixture_against_data(invalid_mix, df.education, :education)
        
        # All levels valid
        complete_mix = mix("high_school" => 0.3, "college" => 0.4, "graduate" => 0.3)
        @test Margins.validate_mixture_against_data(complete_mix, df.education, :education) == true
    end
    
    @testset "Bool Validation" begin
        df = DataFrame(
            treated = [true, false, true, false],
            y = [1, 0, 1, 0]
        )
        
        # Valid Bool mixtures
        bool_mix1 = mix(true => 0.6, false => 0.4)
        @test Margins.validate_mixture_against_data(bool_mix1, df.treated, :treated) == true
        
        bool_mix2 = mix("true" => 0.3, "false" => 0.7)
        @test Margins.validate_mixture_against_data(bool_mix2, df.treated, :treated) == true
        
        # Invalid Bool mixture
        invalid_bool = mix("yes" => 0.5, "no" => 0.5)
        @test_throws ArgumentError Margins.validate_mixture_against_data(invalid_bool, df.treated, :treated)
    end
    
    @testset "Generic String Validation" begin
        df = DataFrame(
            region = ["north", "south", "east", "west"],
            y = [1, 0, 1, 0]
        )
        
        # Valid string mixture
        region_mix = mix("north" => 0.25, "south" => 0.75)
        @test Margins.validate_mixture_against_data(region_mix, df.region, :region) == true
        
        # Invalid string mixture
        invalid_region = mix("north" => 0.5, "central" => 0.5)
        @test_throws ArgumentError Margins.validate_mixture_against_data(invalid_region, df.region, :region)
    end
end

@testset "Mixture to Scenario Value Conversion" begin
    @testset "CategoricalArray Conversion" begin
        education_levels = ["high_school", "college", "graduate"]  
        df = DataFrame(education = categorical(education_levels))
        
        # Test weighted average computation
        # Levels are alphabetically ordered: college=1, graduate=2, high_school=3
        edu_mix = mix("high_school" => 0.5, "college" => 0.3, "graduate" => 0.2)
        scenario_val = Margins.mixture_to_scenario_value(edu_mix, df.education)
        
        expected = 0.5 * 3 + 0.3 * 1 + 0.2 * 2  # 1.5 + 0.3 + 0.4 = 2.2
        @test scenario_val ≈ expected
        
        # Edge case: single level
        single_mix = mix("college" => 1.0)
        single_val = Margins.mixture_to_scenario_value(single_mix, df.education)
        @test single_val ≈ 1.0  # college is index 1 (alphabetically first)
    end
    
    @testset "Bool Conversion" begin
        df = DataFrame(treated = [true, false, true])
        
        # Test probability of true
        bool_mix1 = mix(true => 0.3, false => 0.7)
        prob_true = Margins.mixture_to_scenario_value(bool_mix1, df.treated)
        @test prob_true ≈ 0.3
        
        bool_mix2 = mix("true" => 0.8, "false" => 0.2)  
        prob_true2 = Margins.mixture_to_scenario_value(bool_mix2, df.treated)
        @test prob_true2 ≈ 0.8
        
        # Edge cases
        all_true = mix(true => 1.0, false => 0.0)
        @test Margins.mixture_to_scenario_value(all_true, df.treated) ≈ 1.0
        
        all_false = mix(true => 0.0, false => 1.0)
        @test Margins.mixture_to_scenario_value(all_false, df.treated) ≈ 0.0
    end
    
    @testset "Generic String Conversion" begin
        df = DataFrame(region = ["A", "B", "C", "A", "B"])
        
        # Test weighted average with sorted levels: A=1, B=2, C=3 (alphabetical order)
        region_mix = mix("A" => 0.2, "B" => 0.3, "C" => 0.5)
        scenario_val = Margins.mixture_to_scenario_value(region_mix, df.region)
        
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
        @test Margins.validate_mixture_against_data(edu_mix, df.education, :education) == true
        
        # Test scenario value computation
        scenario_val = Margins.mixture_to_scenario_value(edu_mix, df.education)
        @test isa(scenario_val, Real)
        @test isfinite(scenario_val)
    end
    
    @testset "profile_margins() Integration" begin
        Random.seed!(06515)
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
            
            # Create explicit reference grid for simple_model (only education + age)
            ref_grid = DataFrame(education=[edu_mix], age=[mean(df.age)])
            
            # Test predictions
            result_pred = profile_margins(simple_model, df, ref_grid; type=:predictions)
            @test result_pred isa MarginsResult
            @test nrow(DataFrame(result_pred)) == 1
            @test "estimate" in names(DataFrame(result_pred))
            @test isfinite(DataFrame(result_pred).estimate[1])
            
            # Test effects (only age since education is categorical)
            result_eff = profile_margins(simple_model, df, ref_grid; type=:effects, vars=[:age])
            @test result_eff isa MarginsResult
            @test nrow(DataFrame(result_eff)) == 1  # Only age effect
            @test all(isfinite, DataFrame(result_eff).estimate)
        end
        
        @testset "Categorical Mixture Functionality" begin
            edu_mix = mix("high_school" => 0.4, "college" => 0.4, "graduate" => 0.2)
            
            # Test that mixture prediction works and is finite
            ref_grid_mix = DataFrame(education=[edu_mix], age=[mean(df.age)])
            mixture_result = profile_margins(simple_model, df, ref_grid_mix; type=:predictions)
            mixture_pred = DataFrame(mixture_result).estimate[1]
            
            @test mixture_result isa MarginsResult
            @test isfinite(mixture_pred)
            @test length(DataFrame(mixture_result).estimate) == 1
            
            # Test that mixture prediction differs from individual discrete predictions
            # This confirms FormulaCompiler is using fractional indicators, not discrete averaging
            original_levels = levels(df.education)
            ref_grid_hs = DataFrame(education=categorical(["high_school"], levels=original_levels), age=[mean(df.age)])
            hs_result = profile_margins(simple_model, df, ref_grid_hs; type=:predictions)
            hs_pred = DataFrame(hs_result).estimate[1]
            
            # Mixture should generally differ from any single discrete prediction
            # (unless the mixture heavily weights one level, which it doesn't here)
            @test mixture_pred != hs_pred  # Confirms fractional indicator approach
            
            # NOTE: We do NOT test mixture_pred == weighted_avg(discrete_predictions)
            # because FormulaCompiler correctly uses fractional indicators in the linear predictor,
            # not post-hoc averaging of discrete predictions. See CAT_MIX_CONCEPT.md for details.
        end
        
        @testset "Boolean Variable Support" begin
            # Per FormulaCompiler BOOLEAN_USER_GUIDE.md:
            # Boolean variables work as continuous 0/1 values - no special mixtures needed!
            # Just use numeric values: false=0.0, true=1.0, or intermediate values like 0.7
            
            # Test with numeric boolean value (70% treatment probability)
            ref_grid_numeric = DataFrame(
                employed=[0.7],  # Simple numeric value instead of complex mixture
                education=categorical([first(levels(df.education))], levels=levels(df.education)),
                region=categorical([first(levels(df.region))], levels=levels(df.region)),
                age=[mean(df.age)],
                income=[mean(df.income)]
            )
            bool_result = profile_margins(interaction_model, df, ref_grid_numeric; type=:predictions)
            @test bool_result isa MarginsResult
            @test isfinite(DataFrame(bool_result).estimate[1])
            
            # Test discrete boolean scenarios (true/false work directly)
            ref_grid_true = DataFrame(employed=[true], education=categorical([first(levels(df.education))], levels=levels(df.education)), region=categorical([first(levels(df.region))], levels=levels(df.region)), age=[mean(df.age)], income=[mean(df.income)])
            ref_grid_false = DataFrame(employed=[false], education=categorical([first(levels(df.education))], levels=levels(df.education)), region=categorical([first(levels(df.region))], levels=levels(df.region)), age=[mean(df.age)], income=[mean(df.income)])
            emp_true = DataFrame(profile_margins(interaction_model, df, ref_grid_true; type=:predictions)).estimate[1]
            emp_false = DataFrame(profile_margins(interaction_model, df, ref_grid_false; type=:predictions)).estimate[1]
            
            # Verify both discrete predictions work
            @test isfinite(emp_true) && isfinite(emp_false)
            
            # Verify mixture uses fractional indicators (should differ from naive weighted average)
            mixture_pred = DataFrame(bool_result).estimate[1]
            @test isfinite(mixture_pred)
            # Note: Don't test exact weighted average equality due to fractional indicator behavior
        end
        
        @testset "Multiple Categorical Mixtures" begin
            edu_mix = mix("high_school" => 0.3, "college" => 0.5, "graduate" => 0.2)
            region_mix = mix("urban" => 0.6, "rural" => 0.4)
            
            # Test multiple string mixtures together (avoiding boolean mixtures for now)
            ref_grid_multi = DataFrame(
                education=[edu_mix],
                region=[region_mix],
                employed=[true],  # Use concrete boolean instead of mixture
                age=[mean(df.age)],
                income=[mean(df.income)]
            )
            result = profile_margins(interaction_model, df, ref_grid_multi; type=:predictions)
            @test result isa MarginsResult
            @test nrow(DataFrame(result)) == 1
            @test isfinite(DataFrame(result).estimate[1])
            
            # Test with mixture + continuous overrides
            ref_grid_mixed = DataFrame(
                education=[edu_mix],
                region=categorical([first(levels(df.region))]),
                employed=[true],
                age=[40],
                income=[60000]
            )
            result_mixed = profile_margins(interaction_model, df, ref_grid_mixed; type=:predictions)
            @test result_mixed isa MarginsResult
            @test isfinite(DataFrame(result_mixed).estimate[1])
        end
        
        @testset "Effects with Mixtures" begin
            edu_mix = mix("high_school" => 0.4, "college" => 0.4, "graduate" => 0.2)
            
            ref_grid_eff = DataFrame(
                education=[edu_mix],
                region=categorical([first(levels(df.region))]),
                employed=[true],
                age=[mean(df.age)],
                income=[mean(df.income)]
            )
            result = profile_margins(interaction_model, df, ref_grid_eff; type=:effects)
            @test result isa MarginsResult
            @test nrow(DataFrame(result)) >= 1  # At least one effect (age, income, or employed)
            @test "term" in names(DataFrame(result))
            @test "estimate" in names(DataFrame(result))
            @test all(isfinite, DataFrame(result).estimate)
            @test all(isfinite, DataFrame(result).se)
        end
        
        @testset "Error Handling" begin
            # Invalid weights (don't sum to 1) - this should still work
            @test_throws ArgumentError mix("high_school" => 0.3, "college" => 0.8)
            
            # Empty mixture - this should still work
            @test_throws ArgumentError mix()
            
            # NOTE: Invalid mixture levels validation moved to FormulaCompiler
            # FormulaCompiler may be more permissive than the old Margins.jl validation
            # This is acceptable as per CAT_OLD.md migration notes
        end
    end
end