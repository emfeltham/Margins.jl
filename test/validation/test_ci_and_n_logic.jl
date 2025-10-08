# test_ci_and_n_logic.jl - Verify CI columns and :n logic unchanged and correct

using Random, DataFrames, CategoricalArrays, GLM, Statistics, Distributions

@testset "CI Columns and Sample Size Logic" begin
    # Set up test data
    Random.seed!(1234)
    n = 100
    df = DataFrame(
        x = randn(n),
        z = randn(n),
        group1 = categorical(rand(["A", "B"], n)),
        treatment = categorical(rand(["control", "treated"], n))
    )
    df.y = randn(n) .+ 0.5 * df.x .+ 0.3 * df.z
    m = lm(@formula(y ~ x + z + treatment), df)
    
    @testset "Standard Table CI Columns" begin
        # Test standard table with alpha specified
        res = population_margins(m, df; type=:effects, vars=[:x], ci_alpha=0.05)
        df_result = DataFrame(res; format=:standard)
        
        # Should have CI columns when alpha is specified
        @test :ci_lower in propertynames(df_result)
        @test :ci_upper in propertynames(df_result)
        @test length(df_result.ci_lower) == length(df_result.estimate)
        @test length(df_result.ci_upper) == length(df_result.estimate)
        
        # CI bounds should be reasonable (lower < estimate < upper)
        @test all(df_result.ci_lower .< df_result.estimate)
        @test all(df_result.estimate .< df_result.ci_upper)
        
        # CI width should be approximately 2 * z_crit * SE for 95% CI
        z_crit = quantile(Normal(), 0.975)  # Exact critical value
        expected_half_width = z_crit .* df_result.se
        actual_half_width = (df_result.ci_upper .- df_result.ci_lower) ./ 2
        @test all(abs.(actual_half_width .- expected_half_width) .< 1e-6)
    end
    
    @testset "Confidence Table Format" begin
        # Test dedicated confidence table format
        res = population_margins(m, df; type=:effects, vars=[:x], ci_alpha=0.10)  # 90% CI
        df_result = DataFrame(res; format=:confidence)
        
        @test :lower in propertynames(df_result)  # confidence table uses :lower/:upper
        @test :upper in propertynames(df_result)
        @test :ci_lower ∉ propertynames(df_result)  # not :ci_lower/:ci_upper
        @test :ci_upper ∉ propertynames(df_result)
        
        # Should use 90% CI (z ≈ 1.645)
        # Note: confidence format doesn't include :se column, so we need to get it from standard format
        df_std = DataFrame(res; format=:standard)
        expected_half_width = 1.6449 .* df_std.se  # quantile(Normal(), 0.95)
        actual_half_width = (df_result.upper .- df_result.lower) ./ 2
        @test all(abs.(actual_half_width .- expected_half_width) .< 1e-3)
    end
    
    @testset "Default Alpha Behavior" begin
        # population_margins always includes CI columns (default alpha=0.05)
        res = population_margins(m, df; type=:effects, vars=[:x])  # uses default alpha=0.05
        df_result = DataFrame(res; format=:standard)
        
        @test :ci_lower in propertynames(df_result)  # CI columns always present
        @test :ci_upper in propertynames(df_result)
        
        # Should be 95% CI (default alpha=0.05)
        z_crit = quantile(Normal(), 0.975)  # Exact critical value
        expected_half_width = z_crit .* df_result.se
        actual_half_width = (df_result.ci_upper .- df_result.ci_lower) ./ 2
        @test all(abs.(actual_half_width .- expected_half_width) .< 1e-6)
        
        # Explicit alpha should also work
        res_explicit = population_margins(m, df; type=:effects, vars=[:x], ci_alpha=0.05)
        df_explicit = DataFrame(res_explicit; format=:standard)
        @test :ci_lower in propertynames(df_explicit)
        @test :ci_upper in propertynames(df_explicit)
        
        # CI values should match between default and explicit
        @test df_result.ci_lower ≈ df_explicit.ci_lower
        @test df_result.ci_upper ≈ df_explicit.ci_upper
    end
    
    @testset "Sample Size (:n) Logic - Simple Case" begin
        # Simple case: no grouping
        res = population_margins(m, df; type=:effects, vars=[:x])
        df_result = DataFrame(res)
        
        @test :n in propertynames(df_result)
        @test all(df_result.n .== n)  # Should be full sample size
        @test length(df_result.n) == length(df_result.estimate)
    end
    
    @testset "Sample Size (:n) Logic - With Grouping" begin
        # Grouped case: should have subgroup sizes
        res = population_margins(m, df; type=:effects, vars=[:x], groups=:group1)
        df_result = DataFrame(res)
        
        @test :n in propertynames(df_result)
        @test length(df_result.n) == length(df_result.estimate)
        
        # Each subgroup should have n ≤ total sample size
        @test all(df_result.n .<= n)
        @test all(df_result.n .> 0)  # All subgroups should be non-empty
        
        # Total across subgroups should equal full sample
        # (accounting for the fact that we may have multiple rows per subgroup for different effects)
        unique_groups = unique(df_result.group1)
        for group in unique_groups
            group_rows = df_result[df_result.group1 .== group, :]
            group_n = first(group_rows.n)  # All rows for same group should have same n
            @test all(group_rows.n .== group_n)
            
            # Verify this matches actual data
            actual_group_size = sum(df.group1 .== group)
            @test group_n == actual_group_size
        end
    end
    
    @testset "Sample Size (:n) Logic - Predictions" begin
        # Test predictions also have correct :n
        res = population_margins(m, df; type=:predictions)
        df_result = DataFrame(res)
        
        @test :n in propertynames(df_result)
        @test all(df_result.n .== n)
        
        # With grouping
        res_grouped = population_margins(m, df; type=:predictions, groups=:treatment)
        df_grouped = DataFrame(res_grouped)
        
        @test :n in propertynames(df_grouped)
        @test all(df_grouped.n .<= n)
        @test all(df_grouped.n .> 0)
    end
    
    @testset "Stata Table Format - Uses :N (uppercase)" begin
        # Stata format should use uppercase N
        res = population_margins(m, df; type=:effects, vars=[:x])
        df_stata = DataFrame(res; format=:stata)
        
        @test :N in propertynames(df_stata)  # Uppercase N for Stata
        @test :n ∉ propertynames(df_stata)   # Not lowercase n
        @test all(df_stata.N .== n)
    end
    
    @testset "Profile Margins CI and N Logic" begin
        # Profile margins should also have correct CI and n logic
        res = profile_margins(m, df, means_grid(df); type=:effects, vars=[:x], ci_alpha=0.05)
        df_result = DataFrame(res)
        
        @test :n in propertynames(df_result)
        @test all(df_result.n .== n)  # Should use full sample size
        
        # Profile format should have CI columns
        df_profile = DataFrame(res; format=:profile)
        @test :lower in propertynames(df_profile)
        @test :upper in propertynames(df_profile)
        @test all(df_profile.lower .< df_profile.estimate)
        @test all(df_profile.estimate .< df_profile.upper)
    end
    
    @testset "CI Calculation Function Correctness" begin
        # Test the underlying CI calculation function directly
        estimates = [1.0, 2.0, 3.0]
        ses = [0.1, 0.2, 0.15]
        alpha = 0.05
        
        lower, upper = Margins._calculate_confidence_intervals(estimates, ses, alpha)
        
        z = quantile(Normal(), 1 - alpha/2)  # Should be ≈ 1.96
        expected_lower = estimates .- z .* ses
        expected_upper = estimates .+ z .* ses
        
        @test lower ≈ expected_lower
        @test upper ≈ expected_upper
    end
    
    @testset "Column Ordering - CI and N Placement" begin
        # Verify CI and N columns appear in correct positions after context columns
        res = population_margins(m, df; type=:effects, vars=[:x], groups=:group1, ci_alpha=0.05)
        df_result = DataFrame(res)
        
        col_names = names(df_result)
        
        # Context columns should be first
        @test col_names[1] == "group1"
        @test col_names[2] == "type"
        
        # Statistical columns should follow
        @test "estimate" in col_names
        @test "se" in col_names
        @test "ci_lower" in col_names
        @test "ci_upper" in col_names
        @test "n" in col_names
        
        # All should come after context columns
        context_end = 2  # group1, type
        estimate_pos = findfirst(==("estimate"), col_names)
        n_pos = findfirst(==("n"), col_names)
        ci_lower_pos = findfirst(==("ci_lower"), col_names)
        
        @test estimate_pos > context_end
        @test n_pos > context_end
        @test ci_lower_pos > context_end
    end
end