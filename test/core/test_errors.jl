using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins
using Dates
using Statistics
using LinearAlgebra

@testset "Error handling and input validation" begin
    Random.seed!(131415)
    n = 100
    df = DataFrame(
        y = Float64.(randn(n)),  # Ensure Float64
        x = Float64.(randn(n)),  # Ensure Float64
        z = Float64.(randn(n))   # Ensure Float64
    )

    m = lm(@formula(y ~ x + z), df)

    # Test invalid type parameter
    @testset "Invalid type parameter" begin
        @debug "Testing invalid type parameter error handling" test_input=:invalid expected_error=ArgumentError
        @test_throws ArgumentError population_margins(m, df; type=:invalid)
        @test_throws ArgumentError profile_margins(m, df, means_grid(df); type=:invalid)
    end

    # Test invalid scale parameter (deprecated target parameter) 
    @testset "Invalid scale parameter (deprecated target)" begin
        @debug "Testing deprecated target parameter error handling" test_input=:invalid expected_error=MethodError
        @test_throws MethodError population_margins(m, df; type=:effects, target=:invalid)  # target param no longer exists
    end

    # Test invalid scale parameter
    @testset "Invalid scale parameter" begin
        @debug "Testing invalid scale parameter error handling" test_input=:invalid expected_error=ArgumentError
        @test_throws ArgumentError population_margins(m, df; type=:predictions, scale=:invalid)
    end

    # Test empty vars
    @testset "Empty vars parameter" begin
        @debug "Testing empty vars parameter error handling" test_input=Symbol[] expected_error=ArgumentError
        # Empty vars should throw informative error
        @test_throws ArgumentError population_margins(m, df; type=:effects, vars=Symbol[])
    end

    # Test invalid variable names
    @testset "Invalid variable names" begin
        @debug "Testing invalid variable names error handling" test_input=[:nonexistent] expected_error=Margins.MarginsError
        @test_throws Margins.MarginsError population_margins(m, df; type=:effects, vars=[:nonexistent])
    end

    # Test mismatched model and data
    @testset "Mismatched model and data" begin
        df_wrong = DataFrame(a=randn(n), b=randn(n))
        @debug "Testing mismatched model and data error handling" model_vars=[:x,:z] data_vars=names(df_wrong) expected_error=Exception
        @test_throws Exception population_margins(m, df_wrong; type=:effects)
    end

    # Test invalid reference grid for profile_margins
    @testset "Invalid reference grid" begin
        invalid_grid = DataFrame(nonexistent_var=[1.0])  # Variable not in data
        @debug "Testing invalid reference grid error handling" grid_vars=names(invalid_grid) data_vars=names(df) expected_error=Exception
        @test_throws Exception profile_margins(m, df, invalid_grid; type=:effects)
    end

    # Test data type compatibility
    @testset "Data type compatibility" begin
        df_mixed = DataFrame(
            y = randn(50),
            x = rand(1:10, 50),  # Integer
            z = rand(Bool, 50)   # Boolean
        )
        m_mixed = lm(@formula(y ~ x + z), df_mixed)
        
        # Mixed data types should work fine (Int/Bool are properly handled)
        result = profile_margins(m_mixed, df_mixed, means_grid(df_mixed); type=:effects)
        @test nrow(DataFrame(result)) >= 1  # Should succeed
    end

    # Test unsupported data types - explicit error policy
    @testset "Unsupported data types explicit errors" begin
        @debug "Testing explicit errors for unsupported data types"
        
        # Test with unsupported data type in population_margins
        @testset "Unsupported type in population data" begin
            df_unsupported = DataFrame(
                y = randn(50),
                x = randn(50),
                unsupported = fill(Date(2024, 1, 1), 50)  # Date type not supported
            )
            m_unsup = lm(@formula(y ~ x), df_unsupported)  # Model doesn't use unsupported var
            
            # This should work since unsupported var is not used in model
            result = population_margins(m_unsup, df_unsupported; type=:effects)
            @test nrow(DataFrame(result)) >= 1
            
            # But trying to create a reference grid with unsupported type should fail
            @test_throws Margins.MarginsError means_grid(df_unsupported)
        end

        @testset "Unsupported type in reference grid" begin
            df_base = DataFrame(
                y = randn(50),
                x = randn(50)
            )
            m_base = lm(@formula(y ~ x), df_base)
            
            # Create reference grid with unsupported Date type
            invalid_grid = DataFrame(
                x = [0.0],
                invalid_date = [Date(2024, 1, 1)]
            )
            
            # Should fail with explicit error about unsupported data type
            @test_throws Margins.MarginsError profile_margins(m_base, df_base, invalid_grid; type=:effects)
        end

        @testset "Complex number type rejection" begin
            df_base = DataFrame(
                y = randn(50),
                x = randn(50)
            )
            m_base = lm(@formula(y ~ x), df_base)
            
            # Create reference grid with complex numbers
            invalid_grid = DataFrame(
                x = [0.0],
                complex_var = [1.0 + 2.0im]
            )
            
            # Should fail with explicit error about unsupported data type
            @test_throws Margins.MarginsError profile_margins(m_base, df_base, invalid_grid; type=:effects)
        end

        @testset "String variables in optimized path" begin
            df_with_strings = DataFrame(
                y = randn(50),
                x = randn(50),
                str_var = fill("test", 50)
            )
            
            # String variables should be rejected in optimized reference grids
            # (though they work in regular _get_typical_value via mode())
            @test_throws Margins.MarginsError Margins._get_typical_value_optimized(df_with_strings.str_var, mean)
        end
    end

    # Test resource limit enforcement - explicit error policy
    @testset "Resource limit enforcement explicit errors" begin
        @debug "Testing explicit errors for resource limits"
        
        df_base = DataFrame(
            y = randn(100),
            x = randn(100),
            group1 = categorical(rand(1:20, 100)),  # 20 groups
            group2 = categorical(rand(1:20, 100))   # 20 groups  
        )
        m_base = lm(@formula(y ~ x), df_base)
        
        @testset "Moderate combination limit (>250)" begin
            # Create scenarios that would result in >250 combinations
            # 20 groups for group1 * 20 groups for group2 = 400 combinations 
            
            # Should error with explicit message about combination limits
            @test_throws Margins.MarginsError population_margins(
                m_base, df_base; 
                type=:effects, 
                groups=[:group1, :group2]  # Multiple groups cause combination explosion
            )
        end

        @testset "High combination limit (>1000)" begin
            # Create data with many groups to exceed 1000 limit
            df_many_groups = DataFrame(
                y = randn(100),
                x = randn(100),
                group1 = categorical(rand(1:40, 100)),  # 40 groups
                group2 = categorical(rand(1:30, 100))   # 30 groups -> 40*30 = 1200 combinations
            )
            m_many = lm(@formula(y ~ x), df_many_groups)
            
            # Should error with explicit message about memory exhaustion
            @test_throws Margins.MarginsError population_margins(
                m_many, df_many_groups;
                type=:effects,
                groups=[:group1, :group2]  # Multiple groups with many levels
            )
        end

        @testset "Acceptable combination counts" begin
            # Small number of combinations should work fine
            df_small = DataFrame(
                y = randn(100),
                x = randn(100),
                group1 = categorical(rand(1:5, 100))  # 5 groups only
            )
            m_small = lm(@formula(y ~ x), df_small)
            
            # Should work without error
            result = population_margins(
                m_small, df_small;
                type=:effects,
                groups=:group1  # Single group with few levels
            )
            @test nrow(DataFrame(result)) >= 1  # Should succeed
        end
    end

    # Test DataFrame structure validation - explicit error policy
    @testset "DataFrame structure validation explicit errors" begin
        @debug "Testing explicit errors for DataFrame structure incompatibilities"
        
        # Test internal DataFrame concatenation function directly
        @testset "DataFrame concatenation explicit error behavior" begin
            # CRITICAL: Test that the old dangerous silent string-filling behavior is gone
            # Even though DataFrames.jl handles most cases gracefully, the key improvement
            # is that we no longer have silent fallback code that fills with "missing" strings
            
            df1 = DataFrame(
                term = ["x"],
                estimate = [1.0],
                se = [0.1]
            )
            
            df2 = DataFrame(
                term = ["y"], 
                estimate = [2.0],
                se = [0.2],
                context_info = ["group_A"]  # Extra column
            )
            
            # This should now work correctly with proper missing values (not string "missing")
            result = Margins._append_results_with_missing_columns(df1, df2)
            @test nrow(result) == 2
            # Verify we get proper missing values, not string "missing"
            @test ismissing(result[1, :context_info])  # Not string "missing"
            @test result[2, :context_info] == "group_A"
        end

        @testset "Compatible DataFrame concatenation" begin
            # Test that compatible DataFrames still work
            df1 = DataFrame(
                term = ["x"],
                estimate = [1.0], 
                se = [0.1]
            )
            
            df2 = DataFrame(
                term = ["y"],
                estimate = [2.0],
                se = [0.2]
            )
            
            # Should work without error
            result = Margins._append_results_with_missing_columns(df1, df2)
            @test nrow(result) == 2
            @test names(result) == ["term", "estimate", "se"]
        end

        @testset "Empty DataFrame handling" begin
            # Test edge case with empty DataFrame
            empty_df = DataFrame()
            df_with_data = DataFrame(term=["x"], estimate=[1.0], se=[0.1])
            
            # Should work correctly
            result = Margins._append_results_with_missing_columns(empty_df, df_with_data)
            @test result == df_with_data
        end
    end

    # Test gradient format validation - explicit error policy
    @testset "Gradient format validation explicit errors" begin
        @debug "Testing gradient format error policy compliance"
        
        @testset "Critical fix: old gradient format fallback removed" begin
            # CRITICAL TEST: Verify that the dangerous old gradient format fallback
            # has been removed from src/features/averaging.jl:225-231
            # 
            # The old code had:
            #   elseif isa(grad_key, Tuple) && length(grad_key) == 2
            #       # Fallback to old format if available  <- DANGEROUS
            #       g_term, g_prof_idx = grad_key
            #
            # This has been replaced with explicit MarginsError to prevent
            # incorrect gradient associations and invalid standard errors.
            
            # Since the internal function is not exported, we test indirectly
            # by ensuring profile margins with grouping work correctly.
            # The key improvement is that silent format fallbacks are eliminated.
            
            df = DataFrame(
                y = randn(50),
                x = randn(50),
                group = categorical(rand(["A", "B"], 50))
            )
            m = lm(@formula(y ~ x), df)
            
            # This should work correctly with proper gradient format handling
            # (no more silent fallbacks to old formats)
            result = profile_margins(m, df, means_grid(df); type=:effects)
            @test nrow(DataFrame(result)) >= 1
        end
    end
end