# Test for HP Issue #1: Categorical Effects with Table-Based Profiles

using Margins, GLM, DataFrames, Test, Random, CategoricalArrays

@testset "Categorical Effects with Table-Based Profiles" begin

    # Setup test data with categorical variables
    Random.seed!(456)
    n = 100
    df = DataFrame(
        x1 = randn(n),
        x2 = randn(n), 
        treatment = categorical(repeat(["Control", "Treatment"], n ÷ 2)),
        region = categorical(repeat(["North", "South", "East"], n ÷ 3 + 1)[1:n])
    )
    # Add some effect of treatment and region
    df.y = 0.5 * df.x1 + 0.3 * df.x2 + 
           0.4 * (df.treatment .== "Treatment") +
           0.2 * (df.region .== "North") +
           0.1 * (df.region .== "East") + randn(n) * 0.1

    # Fit model with categorical variables
    model = lm(@formula(y ~ x1 + x2 + treatment + region), df)

    @testset "Basic Table-Based Categorical Effects" begin
        # Create table-based reference grid
        reference_grid = DataFrame(
            x1 = [0.0, 1.0, 0.0, 1.0],
            x2 = [0.0, 0.0, 1.0, 1.0],
            treatment = categorical(["Control", "Control", "Treatment", "Treatment"]),
            region = categorical(["North", "North", "North", "North"])
        )
        
        # Test that this no longer warns and works
        result = @test_nowarn profile_margins(model, df, reference_grid;
                                              type = :effects,
                                              vars = [:treatment])
        
        # Verify structure
        @test nrow(DataFrame(result)) == 4  # One contrast per profile
        @test all(DataFrame(result).term .== :treatment)
        @test all(DataFrame(result).level_from .== "Control") 
        @test all(DataFrame(result).level_to .== "Treatment")
        @test all(.!ismissing.(DataFrame(result).se))
        @test all(DataFrame(result).se .> 0)
        
        # Check profile columns are preserved
        @test :at_x1 in names(DataFrame(result))
        @test :at_x2 in names(DataFrame(result)) 
        @test Set(DataFrame(result).at_x1) == Set([0.0, 1.0])
        @test Set(DataFrame(result).at_x2) == Set([0.0, 1.0])
    end

    @testset "Multiple Categorical Variables" begin
        # Test with multiple categorical variables
        reference_grid = DataFrame(
            x1 = [0.0, 0.0],
            x2 = [0.0, 1.0], 
            treatment = categorical(["Control", "Treatment"]),
            region = categorical(["North", "South"])
        )
        
        result = profile_margins(model, df, reference_grid;
                                type = :effects,
                                vars = [:treatment, :region])
        
        # Should have contrasts for both treatment and region
        treatment_rows = result[DataFrame(result).term .== :treatment, :]
        region_rows = result[DataFrame(result).term .== :region, :]
        
        @test nrow(DataFrame(treatment_rows)) == 2  # One per profile
        @test nrow(DataFrame(region_rows)) >= 2     # Multiple region contrasts possible
        
        # All should have valid SEs
        @test all(.!ismissing.(DataFrame(result).se))
        @test all(DataFrame(result).se .> 0)
    end

    @testset "Different Contrast Types" begin
        reference_grid = DataFrame(
            x1 = [0.0, 1.0],
            x2 = [0.5, 0.5],
            treatment = categorical(["Control", "Control"]),
            region = categorical(["North", "South"])
        )
        
        # Test baseline contrasts
        result_baseline = profile_margins(model, df, reference_grid;
                                         type = :effects,
                                         vars = [:region],
                                         contrasts = :baseline)
        
        # Test pairwise contrasts (default)
        result_pairwise = profile_margins(model, df, reference_grid;
                                         type = :effects,
                                         vars = [:region],
                                         contrasts = :pairwise)
        
        # Baseline should have fewer contrasts than pairwise for multi-level categorical
        region_levels = length(levels(df.region))
        if region_levels > 2
            baseline_contrasts = nrow(result_baseline[DataFrame(result_baseline).term .== :region, :])
            pairwise_contrasts = nrow(result_pairwise[DataFrame(result_pairwise).term .== :region, :])
            @test baseline_contrasts < pairwise_contrasts
        end
    end

    @testset "Link Scale Effects" begin
        # Test categorical effects on link scale (η) vs response scale (μ) 
        reference_grid = DataFrame(
            x1 = [0.0],
            x2 = [0.0],
            treatment = categorical(["Control"]),
            region = categorical(["North"])
        )
        
        result_eta = profile_margins(model, df, reference_grid;
                                    type = :effects,
                                    vars = [:treatment],
                                    target = :eta)  # Link scale
        
        result_mu = profile_margins(model, df, reference_grid;
                                   type = :effects, 
                                   vars = [:treatment],
                                   target = :mu)   # Response scale
        
        # For linear model, η and μ effects should be identical
        @test abs(DataFrame(result_eta).estimate[1] - DataFrame(result_mu).estimate[1]) < 1e-10
        @test abs(DataFrame(result_eta).se[1] - DataFrame(result_mu).se[1]) < 1e-10
    end

    @testset "Averaging Support" begin
        # Test that categorical effects work with averaging
        reference_grid = DataFrame(
            x1 = [-1.0, 0.0, 1.0],
            x2 = [0.0, 0.0, 0.0],
            treatment = categorical(["Control", "Control", "Control"]),
            region = categorical(["North", "North", "North"])
        )
        
        # Individual profiles
        result_individual = profile_margins(model, df, reference_grid;
                                           type = :effects,
                                           vars = [:treatment],
                                           average = false)
        
        # Averaged profiles
        result_averaged = profile_margins(model, df, reference_grid;
                                         type = :effects,
                                         vars = [:treatment], 
                                         average = true)
        
        @test nrow(DataFrame(result_individual)) == 3  # One per profile
        @test nrow(DataFrame(result_averaged)) == 1    # Averaged to single row
        
        # Averaged effect should equal mean of individual effects  
        @test abs(DataFrame(result_averaged).estimate[1] - mean(DataFrame(result_individual).estimate)) < 1e-10
        
        # Averaged SE should be proper delta-method, not simple average
        simple_se_avg = mean(DataFrame(result_individual).se)
        @test DataFrame(result_averaged).se[1] != simple_se_avg  # Should differ from simple average
    end

    @testset "Mixed Continuous and Categorical" begin
        # Test table with both continuous and categorical effects
        reference_grid = DataFrame(
            x1 = [0.0, 1.0],
            x2 = [0.0, 0.0],
            treatment = categorical(["Control", "Treatment"]),
            region = categorical(["North", "South"])
        )
        
        result = profile_margins(model, df, reference_grid;
                                type = :effects,
                                vars = [:x1, :treatment, :region])
        
        # Should have continuous and categorical effects
        continuous_terms = result[DataFrame(result).term .== :x1, :]
        categorical_terms = result[DataFrame(result).term .!= :x1, :]
        
        @test nrow(DataFrame(continuous_terms)) == 2  # One per profile
        @test nrow(categorical_terms) > 0  # At least some categorical contrasts
        
        # Continuous terms shouldn't have level_from/level_to
        @test all(ismissing.(continuous_terms.level_from))
        @test all(ismissing.(continuous_terms.level_to))
        
        # Categorical terms should have level_from/level_to
        @test all(.!ismissing.(categorical_terms.level_from))
        @test all(.!ismissing.(categorical_terms.level_to))
    end

    @testset "Consistency with Dict-Based Profiles" begin
        # Compare table-based vs dict-based results for same profiles
        
        # Table-based
        reference_grid = DataFrame(
            x1 = [0.0, 1.0],
            x2 = [0.5, 0.5],
            treatment = categorical(["Control", "Control"]),
            region = categorical(["North", "South"])
        )
        
        result_table = profile_margins(model, df, reference_grid;
                                      type = :effects,
                                      vars = [:treatment])
        
        # Dict-based equivalent
        result_dict = profile_margins(model, df;
                                     at = Dict(:x1 => [0.0, 1.0], 
                                              :x2 => [0.5],
                                              :treatment => ["Control"],
                                              :region => ["North", "South"]),
                                     type = :effects,
                                     vars = [:treatment])
        
        # Results should be very similar (allowing for small numerical differences)
        @test nrow(DataFrame(result_table)) == nrow(result_dict)
        
        # Sort both by profile columns for comparison
        table_sorted = sort(result_table, [:at_x1, :at_region])
        dict_sorted = sort(result_dict, [:at_x1, :at_region])
        
        for i in 1:nrow(table_sorted)
            @test abs(table_sorted.estimate[i] - dict_sorted.estimate[i]) < 1e-8
            @test abs(table_sorted.se[i] - dict_sorted.se[i]) < 1e-8
        end
    end
end