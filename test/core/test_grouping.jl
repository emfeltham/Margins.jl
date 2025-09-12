using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins

@testset "Grouping functionality" begin
    Random.seed!(456)
    n = 300
    df = DataFrame(
        y = rand(Bool, n),
        x = Float64.(randn(n)),  # Ensure Float64
        z = Float64.(randn(n)),  # Ensure Float64  
        group1 = categorical(rand(["A","B"], n)),
        group2 = categorical(rand(["X","Y","Z"], n)),
        treatment = categorical(rand(["Control","Treated"], n)),  # Add categorical variable for effects testing
        binary_var = categorical(rand(["Yes","No"], n))  # Add binary categorical variable
    )

    m = glm(@formula(y ~ x + z + group1 + treatment + binary_var), df, Binomial(), LogitLink())

    # Test unified groups parameter - simple categorical (basic test first)
    @testset "Simple categorical groups" begin
        # First test without groups to make sure basic functionality works
        res_basic = population_margins(m, df; type=:effects, vars=[:x])
        @test nrow(DataFrame(res_basic)) == 1
        @test all(isfinite, DataFrame(res_basic).estimate)
        
        # Now test with groups
        res_groups = population_margins(m, df; type=:effects, vars=[:x], groups=:group1)
        @test nrow(DataFrame(res_groups)) >= 1  # May have multiple groups
        @test all(isfinite, DataFrame(res_groups).estimate)
    end

    # Test unified groups parameter - multiple categorical 
    @testset "Multiple categorical groups" begin
        res_multi = population_margins(m, df; type=:effects, vars=[:x], groups=[:group1, :group2])
        expected_rows = length(levels(df.group1)) * length(levels(df.group2))
        @test nrow(DataFrame(res_multi)) <= expected_rows  # May be fewer due to empty combinations
        @test :group1 in propertynames(DataFrame(res_multi))
        @test :group2 in propertynames(DataFrame(res_multi))
    end

    # Test unified groups parameter - continuous groups (quartiles)
    @testset "Continuous groups (quartiles)" begin
        res_continuous = population_margins(m, df; type=:effects, vars=[:z], groups=(:x, 4))
        @test nrow(DataFrame(res_continuous)) == 4  # Four quartiles
        @test :x in propertynames(DataFrame(res_continuous))
        @test all(isfinite, DataFrame(res_continuous).estimate)
        # Check that we have distinct x values (quartile midpoints)
        @test length(unique(DataFrame(res_continuous).x)) == 4
    end

    # Test unified groups parameter - nested groups
    @testset "Nested groups" begin
        res_nested = population_margins(m, df; type=:effects, vars=[:x], groups=(:group1 => :group2))
        expected_rows = length(levels(df.group1)) * length(levels(df.group2))
        @test nrow(DataFrame(res_nested)) <= expected_rows  # May be fewer due to empty combinations  
        @test :group1 in propertynames(DataFrame(res_nested))
        @test :group2 in propertynames(DataFrame(res_nested))
    end

    # Test unified groups parameter - predictions
    @testset "Groups with predictions" begin
        res_pred = population_margins(m, df; type=:predictions, groups=:group1)
        @test nrow(DataFrame(res_pred)) == length(levels(df.group1))
        @test :group1 in propertynames(DataFrame(res_pred))
        @test all(isfinite, DataFrame(res_pred).estimate)
        @test all(DataFrame(res_pred).type .== "AAP")
    end

    # MISSING TEST COVERAGE: Categorical variables with grouping
    @testset "Categorical effects with simple grouping" begin
        # This tests the fixed _compute_categorical_baseline_ame_context function
        res_cat_groups = population_margins(m, df; type=:effects, vars=[:treatment], groups=:group1)
        df_results = DataFrame(res_cat_groups)
        @test nrow(df_results) >= 1  # Should have at least one group
        @test all(isfinite, df_results.estimate)
        @test "treatment" in df_results.variable
        @test :group1 in propertynames(df_results)
    end

    @testset "Categorical effects with multiple groups" begin
        # Test categorical effects with multiple grouping variables
        res_cat_multi = population_margins(m, df; type=:effects, vars=[:binary_var], groups=[:group1, :group2])
        df_results = DataFrame(res_cat_multi)
        @test nrow(df_results) >= 1  # Should have at least one combination
        @test all(isfinite, df_results.estimate)
        @test "binary_var" in df_results.variable
        @test :group1 in propertynames(df_results)
        @test :group2 in propertynames(df_results)
    end

    @testset "Mixed continuous and categorical effects with grouping" begin
        # Test both continuous and categorical variables with grouping
        res_mixed = population_margins(m, df; type=:effects, vars=[:x, :treatment], groups=:group2)
        df_results = DataFrame(res_mixed)
        @test nrow(df_results) >= 2  # At least one continuous + one categorical effect per group
        @test all(isfinite, df_results.estimate)
        @test "x" in df_results.variable
        @test "treatment" in df_results.variable
        @test :group2 in propertynames(df_results)
    end

    @testset "Categorical effects with nested grouping" begin
        # Test categorical effects with nested groups
        res_nested_cat = population_margins(m, df; type=:effects, vars=[:treatment], groups=(:group1 => :group2))
        df_results = DataFrame(res_nested_cat)
        @test nrow(df_results) >= 1  # Should have nested combinations
        @test all(isfinite, df_results.estimate)
        @test "treatment" in df_results.variable
        @test :group1 in propertynames(df_results)
        @test :group2 in propertynames(df_results)
    end
end