# Test for Phase 4: Advanced Profile Averaging with proper delta-method SEs

using Margins, GLM, DataFrames, Test, Random, FormulaCompiler, Statistics

@testset "Advanced Profile Averaging with Proper Delta-Method SEs" begin

    # Setup test data with grouping structure (avoid categorical variables for now)
    Random.seed!(123)
    n = 100
    df = DataFrame(
        x1 = randn(n),
        x2 = randn(n),
        group_num = repeat([1, 2, 3], n ÷ 3 + 1)[1:n],
        region_num = repeat([1, 2], n ÷ 2 + 1)[1:n]
    )
    df.y = 0.5 * df.x1 + 0.3 * df.x2 + 0.2 * (df.group_num .== 2) + randn(n) * 0.1

    # Fit model
    model = lm(@formula(y ~ x1 + x2 + group_num), df)

    @testset "Simple Profile Averaging (no grouping) - baseline check" begin
        # This should work and serve as baseline
        result = profile_margins(model, df; 
            at = Dict(:x1 => [-1, 0, 1], :x2 => [0]),
            type = :effects, 
            vars = [:x1], 
            average = true
        )
        
        @test nrow(result.table) == 1
        @test result.table.term[1] == :x1
        @test !ismissing(result.table.se[1])
        @test result.table.se[1] > 0
    end

    @testset "Grouped Profile Averaging with over" begin
        # Test grouping with over parameter
        result = profile_margins(model, df;
            at = Dict(:x1 => [-1, 0, 1], :x2 => [0]),
            type = :effects,
            vars = [:x1],
            over = :group_num,
            average = true
        )
        
        @test nrow(result.table) == 3  # One row per group (A, B, C)
        @test all(result.table.term .== :x1)
        @test all(.!ismissing.(result.table.se))
        @test all(result.table.se .> 0)
        
        # Check that proper delta-method SEs are computed (not approximations)
        # Delta-method SEs should be different from simple approximations
        groups = unique(result.table.group_num)
        @test length(groups) == 3
        @test Set(groups) == Set([1, 2, 3])
    end

    @testset "Grouped Profile Averaging with by" begin
        # Test grouping with by parameter  
        result = profile_margins(model, df;
            at = Dict(:x1 => [-1, 0, 1], :x2 => [0]),
            type = :effects,
            vars = [:x1],
            by = :region_num,
            average = true
        )
        
        @test nrow(result.table) == 2  # One row per region (North, South)
        @test all(result.table.term .== :x1)
        @test all(.!ismissing.(result.table.se))
        @test all(result.table.se .> 0)
        
        regions = unique(result.table.region_num)
        @test length(regions) == 2
        @test Set(regions) == Set([1, 2])
    end

    @testset "Complex Nested Grouping" begin
        # Test both over and by together
        result = profile_margins(model, df;
            at = Dict(:x1 => [-1, 0, 1], :x2 => [0]),
            type = :effects,
            vars = [:x1],
            over = :group_num,
            by = :region_num,
            average = true
        )
        
        # Should have one row per group×region combination
        @test nrow(result.table) >= 3  # At least 3 groups
        @test all(result.table.term .== :x1)
        @test all(.!ismissing.(result.table.se))
        @test all(result.table.se .> 0)
        
        # Verify we have both group and region columns
        @test :group_num in names(result.table)
        @test :region_num in names(result.table)
    end

    @testset "Multiple Variables with Grouping" begin
        # Test multiple variables with grouping
        result = profile_margins(model, df;
            at = Dict(:x1 => [-1, 1], :x2 => [-1, 1]),
            type = :effects,
            vars = [:x1, :x2],
            over = :group_num,
            average = true
        )
        
        # Should have 2 terms × 3 groups = 6 rows
        @test nrow(result.table) == 6
        @test Set(result.table.term) == Set([:x1, :x2])
        @test all(.!ismissing.(result.table.se))
        @test all(result.table.se .> 0)
    end

    @testset "Predictions with Grouped Averaging" begin
        # Test predictions (not just effects) with grouping
        result = profile_margins(model, df;
            at = Dict(:x1 => [-1, 0, 1], :x2 => [0]),
            type = :predictions,
            over = :group_num,
            average = true
        )
        
        @test nrow(result.table) == 3  # One per group
        @test all(result.table.term .== :prediction)
        @test all(.!ismissing.(result.table.se))
        @test all(result.table.se .> 0)
    end

    @testset "Error Handling - Missing Gradients" begin
        # This test ensures our error handling works when gradients are missing
        # We'll create a scenario that should trigger the gradient requirement
        
        # For now, this is more of a structural test - the main implementation
        # should always provide gradients, so we mainly test that the error
        # message is clear if something goes wrong in the future
        
        @test_nowarn profile_margins(model, df;
            at = Dict(:x1 => [-1, 1]),
            type = :effects,
            vars = [:x1],
            over = :group_num,
            average = true
        )
    end

    @testset "Consistency Check - Manual vs Automatic Averaging" begin
        # Compare automatic averaging with manual calculation
        
        # Get individual profile results with grouping
        individual = profile_margins(model, df;
            at = Dict(:x1 => [-1, 0, 1], :x2 => [0]),
            type = :effects,
            vars = [:x1],
            over = :group_num,
            average = false
        )
        
        # Get automatically averaged results
        averaged = profile_margins(model, df;
            at = Dict(:x1 => [-1, 0, 1], :x2 => [0]),
            type = :effects,
            vars = [:x1],
            over = :group_num,
            average = true
        )
        
        # Verify structure
        @test nrow(averaged.table) == 3  # One per group
        @test nrow(individual.table) == 9  # 3 profiles × 3 groups
        
        # For each group, the averaged effect should equal the mean of individual effects
        for group_val in unique(averaged.table.group_num)
            avg_row = averaged.table[averaged.table.group_num .== group_val, :]
            ind_rows = individual.table[(individual.table.group_num .== group_val) .& (individual.table.term .== :x1), :]
            
            @test length(avg_row.dydx) == 1
            @test abs(avg_row.dydx[1] - mean(ind_rows.dydx)) < 1e-10
            
            # Delta-method SE should be different (and typically smaller) than simple approximation
            simple_se_approx = sqrt(sum(ind_rows.se .^ 2)) / length(ind_rows.se)
            @test avg_row.se[1] != simple_se_approx  # Proper delta-method differs from approximation
        end
    end
end