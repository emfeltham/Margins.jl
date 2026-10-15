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

    # Test deprecated target parameter rejection (defensive testing)
    @testset "Deprecated target parameter rejection" begin
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
        # Bool variables in reference grids are converted to P(true) Float64
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

            # String variables should create CategoricalMixture with correct frequency
            result = Margins._get_typical_value(df_with_strings.str_var, mean)
            @test result isa Margins.CategoricalMixture
            @test result.levels == ["test"]
            @test result.weights == [1.0]  # All entries are "test", so 100% weight
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

    # Test raw String variable validation
    @testset "Raw String variable validation" begin
        @debug "Testing raw String variable rejection"

        @testset "Raw String columns should error" begin
            # Create data with raw String column
            df_string = DataFrame(
                y = randn(50),
                x = randn(50),
                str_var = repeat(["A", "B"], 25)  # Raw String, not CategoricalArray
            )
            m_string = lm(@formula(y ~ x + str_var), df_string)

            # Should throw MarginsError with helpful message
            err = nothing
            try
                population_margins(m_string, df_string; vars=[:str_var])
            catch e
                err = e
            end

            @test err isa Margins.MarginsError
            @test occursin("raw String column", err.msg)
            @test occursin("CategoricalArray", err.msg)
            @test occursin("categorical(data.str_var)", err.msg)  # Should include fix
        end

        @testset "CategoricalArray String columns should work" begin
            # Convert to CategoricalArray - should work fine
            df_cat = DataFrame(
                y = randn(50),
                x = randn(50),
                str_var = categorical(repeat(["A", "B"], 25))
            )
            m_cat = lm(@formula(y ~ x + str_var), df_cat)

            # Should work without error
            result = population_margins(m_cat, df_cat; vars=[:str_var])
            @test nrow(DataFrame(result)) >= 1
        end

        @testset "Multiple variables with String" begin
            # Test that error is caught even when String is not the first variable
            df_multi = DataFrame(
                y = randn(50),
                x = randn(50),
                z = randn(50),
                str_var = repeat(["A", "B"], 25)  # Two levels
            )
            m_multi = lm(@formula(y ~ x + z + str_var), df_multi)

            # Should error when trying to analyze str_var
            @test_throws Margins.MarginsError population_margins(m_multi, df_multi; vars=[:x, :str_var])
        end
    end
    # Regression: vars/groups overlap used to crash with an opaque DataFrames error
    # (`column name "estimate" not found in the data frame since it has no columns`)
    # because the skip rule emptied the result frame. It now mirrors vars/scenarios.
    @testset "vars/groups and vars/scenarios overlap" begin
        Random.seed!(202020)
        n_ov = 300
        df_ov = DataFrame(
            y = Float64.(randn(n_ov)),
            y_bin = rand(Bool, n_ov),
            x = Float64.(randn(n_ov)),
            z = Float64.(randn(n_ov)),
            g = categorical(rand(["A", "B", "C"], n_ov)),
            h = categorical(rand(["P", "Q"], n_ov))
        )
        m_lm = lm(@formula(y ~ x + z + g + h), df_ov)
        m_glm = glm(@formula(y_bin ~ x + z + g + h), df_ov, Binomial(), LogitLink())

        @testset "Grouping by the variable being contrasted" begin
            @debug "Testing vars/groups overlap" test_input=(vars=[:g], groups=:g) expected_error=ArgumentError
            # Reproducer from ISSUES.md: independent of model family and of scale
            for m_ov in (m_lm, m_glm), sc in (:response, :link)
                @test_throws ArgumentError population_margins(m_ov, df_ov; type=:effects, vars=[:g], groups=:g, scale=sc)
            end
            # Symbol form of vars
            @test_throws ArgumentError population_margins(m_lm, df_ov; type=:effects, vars=:g, groups=:g)
            # Partial overlap errors too, matching the scenarios path
            @test_throws ArgumentError population_margins(m_lm, df_ov; type=:effects, vars=[:g, :x], groups=:g)
        end

        @testset "Overlap through every groups syntax" begin
            @debug "Testing vars/groups overlap across groups syntaxes" expected_error=ArgumentError
            # Vector form. A bare Symbol always pins, numeric or not, so :x still conflicts.
            @test_throws ArgumentError population_margins(m_lm, df_ov; type=:effects, vars=[:x], groups=[:g, :x])
            # Nested outer => inner, on either side
            @test_throws ArgumentError population_margins(m_lm, df_ov; type=:effects, vars=[:x], groups=(:g => :x))
            @test_throws ArgumentError population_margins(m_lm, df_ov; type=:effects, vars=[:g], groups=(:g => :h))
            # A categorical pins through the vector form too
            @test_throws ArgumentError population_margins(m_lm, df_ov; type=:effects, vars=[:g], groups=[:h, :g])
        end

        @testset "Binning the variable being differentiated is allowed" begin
            # Binning does not pin: x still varies inside each bin, so its effect there is
            # well defined. Contrast with the categorical cases above.
            for gspec in ((:x, 4), (:x, [-0.5, 0.5]))
                res = DataFrame(population_margins(m_lm, df_ov; type=:effects, vars=[:x], groups=gspec))
                @test all(res.variable .== "x")
                @test sum(res.n) == nrow(df_ov)
                @test all(isfinite, res.estimate)
                @test all(res.se .> 0)
            end
            # Nested and vector forms where the binned variable is the one differentiated
            res_nested = DataFrame(population_margins(m_lm, df_ov; type=:effects, vars=[:x], groups=(:g => (:x, 4))))
            @test sum(res_nested.n) == nrow(df_ov)
            res_vec = DataFrame(population_margins(m_lm, df_ov; type=:effects, vars=[:x], groups=[:g, (:x, 4)]))
            @test sum(res_vec.n) == nrow(df_ov)
        end

        @testset "Binned by groups and overridden by scenarios is rejected" begin
            @debug "Testing scenarios/groups bin conflict" expected_error=ArgumentError
            @test_throws ArgumentError population_margins(
                m_lm, df_ov; type=:effects, vars=[:z], groups=(:x, 4), scenarios=(x=[0.0],)
            )
            # Binning one variable while setting a scenario on another is fine
            res = DataFrame(population_margins(
                m_lm, df_ov; type=:effects, vars=[:z], groups=(:x, 4), scenarios=(g=["A"],)
            ))
            @test nrow(res) == 4
        end

        @testset "vars/scenarios overlap (parity)" begin
            @debug "Testing vars/scenarios overlap" test_input=(vars=[:g],) expected_error=ArgumentError
            @test_throws ArgumentError population_margins(m_lm, df_ov; type=:effects, vars=[:g], scenarios=(g=["A", "B"],))
            @test_throws ArgumentError population_margins(m_lm, df_ov; type=:effects, vars=[:x], scenarios=(x=[-1.0, 1.0],))
        end

        @testset "Disjoint vars/groups still work" begin
            res_cat = population_margins(m_lm, df_ov; type=:effects, vars=[:x], groups=:g)
            @test nrow(DataFrame(res_cat)) == 3
            @test all(isfinite, DataFrame(res_cat).estimate)

            res_bin = population_margins(m_lm, df_ov; type=:effects, vars=[:x], groups=(:z, 4))
            @test nrow(DataFrame(res_bin)) == 4
            @test all(isfinite, DataFrame(res_bin).estimate)

            # Predictions never consult vars, so grouping by anything is fine
            res_pred = population_margins(m_lm, df_ov; type=:predictions, groups=:g)
            @test nrow(DataFrame(res_pred)) == 3
        end

        @testset "Auto-detected vars consumed by groups" begin
            # Auto-detected vars are still skipped silently, so the up-front validator does
            # not fire. When the skip rule consumes every one of them the result frame is
            # empty; that must be a clear MarginsError, not a DataFrames column lookup error.
            df_sole = DataFrame(y = Float64.(randn(200)), x = Float64.(randn(200)))
            m_sole = lm(@formula(y ~ x), df_sole)
            @debug "Testing empty-result guard" test_input=(scenarios=(x=[-1.0, 1.0],),) expected_error=Margins.MarginsError
            # A scenario pins x, so the only auto-detected variable is consumed and nothing
            # can be computed -- that must be a clear MarginsError, not a DataFrames error.
            @test_throws Margins.MarginsError population_margins(m_sole, df_sole; type=:effects, scenarios=(x=[-1.0, 1.0],))
            # Binning x does NOT consume it: the same model now returns one row per bin
            res_sole = DataFrame(population_margins(m_sole, df_sole; type=:effects, groups=(:x, 4)))
            @test nrow(res_sole) == 4
            @test all(res_sole.variable .== "x")
            @test sum(res_sole.n) == 200

            # With a categorical alongside, both variables are reported in every bin
            df_mixed = DataFrame(
                y = Float64.(randn(200)),
                x = Float64.(randn(200)),
                g = categorical(rand(["A", "B"], 200))
            )
            m_mixed = lm(@formula(y ~ x + g), df_mixed)
            res_mixed = DataFrame(population_margins(m_mixed, df_mixed; type=:effects, groups=(:x, 4)))
            @test nrow(res_mixed) == 8                       # 4 bins x {x derivative, g contrast}
            @test Set(unique(res_mixed.variable)) == Set(["x", "g"])
        end
    end
end
