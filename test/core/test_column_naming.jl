# test_column_naming.jl - Test column naming conventions and programmatic identification

using Random, DataFrames, CategoricalArrays, GLM, Statistics

@testset "Column Naming and Programmatic Identification" begin
    # Set up test data
    Random.seed!(1234)
    n = 100
    df = DataFrame(
        x = randn(n),
        z = randn(n),
        group1 = categorical(rand(["A", "B"], n)),
        group2 = categorical(rand(["X", "Y"], n)),
        treatment = categorical(rand(["control", "treated"], n))
    )
    df.y = randn(n) .+ 0.5 * df.x .+ 0.3 * df.z
    m = lm(@formula(y ~ x + z + treatment), df)
    
    @testset "Population Effects with Groups Only" begin
        res = population_margins(m, df; type=:effects, vars=[:x], groups=[:group1, :group2])
        df_result = DataFrame(res)
        
        # Test metadata contains correct information
        @test haskey(res.metadata, :groups_vars)
        @test haskey(res.metadata, :scenarios_vars)
        @test res.metadata[:groups_vars] == [:group1, :group2]
        @test res.metadata[:scenarios_vars] == Symbol[]
        
        # Test column naming: groups should be unprefixed
        @test :group1 in propertynames(df_result)
        @test :group2 in propertynames(df_result)
        @test :at_group1 ∉ propertynames(df_result)
        @test :at_group2 ∉ propertynames(df_result)
        
        # Test column ordering: context columns come first
        col_names = names(df_result)
        group1_pos = findfirst(==("group1"), col_names)
        group2_pos = findfirst(==("group2"), col_names)
        type_pos = findfirst(==("type"), col_names)
        
        @test group1_pos < type_pos  # Groups before type
        @test group2_pos < type_pos  # Groups before type
        
        # Test programmatic identification
        groups_vars = res.metadata[:groups_vars]
        @test all(var -> var in propertynames(df_result), groups_vars)
    end
    
    @testset "Population Predictions with Groups Only" begin
        res = population_margins(m, df; type=:predictions, groups=:group1)
        df_result = DataFrame(res)
        
        # Test metadata
        @test res.metadata[:groups_vars] == [:group1]
        @test res.metadata[:scenarios_vars] == Symbol[]
        
        # Test column naming: groups should be unprefixed
        @test :group1 in propertynames(df_result)
        @test :at_group1 ∉ propertynames(df_result)
        
        # Test column ordering: context columns first
        col_names = names(df_result)
        group1_pos = findfirst(==("group1"), col_names)
        type_pos = findfirst(==("type"), col_names)
        
        @test group1_pos < type_pos
    end
    
    @testset "Profile Effects - Bare Column Names" begin
        res = profile_margins(m, df, means_grid(df); type=:effects, vars=[:x])
        df_result = DataFrame(res)
        
        # Test metadata: profile analysis should have empty groups/scenarios
        @test get(res.metadata, :groups_vars, Symbol[]) == Symbol[]
        @test get(res.metadata, :scenarios_vars, Symbol[]) == Symbol[]
        @test get(res.metadata, :analysis_type, :unknown) == :profile
        
        # Test column naming: profile columns should be bare names (no at_ prefix)
        profile_columns = propertynames(df_result)
        at_columns = filter(name -> startswith(string(name), "at_"), profile_columns)
        @test length(at_columns) == 0  # No at_ prefixes in profile analysis
        
        # Should have some profile columns from means_grid
        # (exact columns depend on means_grid implementation)
        non_stats_columns = filter(name -> !(string(name) in ["type", "variable", "contrast", "estimate", "se", "t_stat", "p_value", "n"]), profile_columns)
        @test length(non_stats_columns) > 0  # Should have at least some profile columns
    end
    
    @testset "Profile Predictions - Bare Column Names" begin
        res = profile_margins(m, df, cartesian_grid(x=[-1, 0, 1], z=[0]); type=:predictions)
        df_result = DataFrame(res)
        
        # Test metadata: profile analysis
        @test get(res.metadata, :analysis_type, :unknown) == :profile
        
        # Test column naming: should have bare x and z columns
        @test :x in propertynames(df_result)
        @test :z in propertynames(df_result)
        @test :at_x ∉ propertynames(df_result)
        @test :at_z ∉ propertynames(df_result)
        
        # Test column ordering: profile columns first
        col_names = names(df_result)
        x_pos = findfirst(==("x"), col_names)
        z_pos = findfirst(==("z"), col_names)
        type_pos = findfirst(==("type"), col_names)
        
        @test x_pos < type_pos
        @test z_pos < type_pos
    end
    
    @testset "Programmatic Identification Helper Function" begin
        # Test the context_columns helper function directly
        
        # Case 1: Groups only
        res_groups = population_margins(m, df; type=:effects, vars=[:x], groups=[:group1, :group2])
        groups, scenarios = Margins.context_columns(res_groups)
        @test groups == [:group1, :group2]
        @test scenarios == Symbol[]
        
        # Case 2: Profile analysis
        res_profile = profile_margins(m, df, means_grid(df); type=:effects, vars=[:x])
        groups_prof, scenarios_prof = Margins.context_columns(res_profile)
        @test groups_prof == Symbol[]
        @test scenarios_prof == Symbol[]
        
        # Case 3: No profile values
        # Create a minimal result for testing edge case
        mock_metadata = Dict{Symbol, Any}(:groups_vars => [:group1], :scenarios_vars => [:scenario1])
        groups_empty, scenarios_empty = Margins.context_columns(
            Margins.EffectsResult(Float64[], Float64[], String[], String[], nothing, nothing, Matrix{Float64}(undef, 0, 0), mock_metadata)
        )
        @test groups_empty == Symbol[]
        @test scenarios_empty == Symbol[]
    end
    
    @testset "Column Ordering Consistency" begin
        # Test that context columns always come first across different result types
        
        res_effects = population_margins(m, df; type=:effects, vars=[:x], groups=[:group1])
        res_predictions = population_margins(m, df; type=:predictions, groups=[:group1])
        
        df_effects = DataFrame(res_effects)
        df_predictions = DataFrame(res_predictions)
        
        # Both should have group1 as first column
        @test names(df_effects)[1] == "group1"
        @test names(df_predictions)[1] == "group1"
        
        # Both should have type as second column
        @test names(df_effects)[2] == "type"
        @test names(df_predictions)[2] == "type"
    end
end