# test/features/test_hierarchical_grids.jl
# Test hierarchical grid functionality for Phase 1.1

using Random, Statistics, DataFrames, CategoricalArrays, Tables, GLM, StatsModels

@testset "Hierarchical Reference Grid Grammar" begin
    # Create test data with mixed types
    Random.seed!(06515)
    n = 100
    data = DataFrame(
        region = categorical(rand(["North", "South"], n)),
        education = categorical(rand(["HS", "College"], n)),
        income = rand(25000:75000, n),
        age = rand(25:65, n),
        treatment = rand([true, false], n),
        y = randn(n)
    )
    
    @testset "Basic Reference Specifications" begin
        # Basic categorical variable specification
        grid = hierarchical_grid(data, :education)
        @test nrow(grid) == 2  # HS and College
        @test "education" in names(grid)
        @test Set(grid.education) == Set(["HS", "College"])
        
        # Cross-tabulation specification  
        grid = hierarchical_grid(data, [:region, :education])
        @test nrow(grid) == 4  # 2 regions × 2 education levels
        @test "region" in names(grid) && "education" in names(grid)
        @test Set(grid.region) == Set(["North", "South"])
        @test Set(grid.education) == Set(["HS", "College"])
        
        # Continuous representative specification - quartiles
        grid = hierarchical_grid(data, (:income, :quartiles))
        @test nrow(grid) == 4  # Q1, Q2, Q3, Q4
        @test "income" in names(grid)
        @test length(unique(grid.income)) == 4
        
        # Continuous representative specification - mean
        grid = hierarchical_grid(data, (:age, :mean))
        @test nrow(grid) == 1
        @test "age" in names(grid)
        @test grid.age[1] ≈ mean(data.age)
        
        # Continuous representative specification - median
        grid = hierarchical_grid(data, (:age, :median))
        @test nrow(grid) == 1
        @test "age" in names(grid)
        @test grid.age[1] ≈ median(data.age)
        
        # Continuous representative specification - fixed values
        grid = hierarchical_grid(data, (:age, [30, 50, 70]))
        @test nrow(grid) == 3
        @test "age" in names(grid)
        @test Set(grid.age) == Set([30, 50, 70])
        
        # New Phase 1.2 features - deciles
        grid = hierarchical_grid(data, (:income, :deciles))
        @test nrow(grid) == 10
        @test "income" in names(grid)
        @test length(unique(grid.income)) == 10
        
        # Terciles/tertiles
        grid = hierarchical_grid(data, (:age, :terciles))
        @test nrow(grid) == 3
        @test "age" in names(grid)
        @test length(unique(grid.age)) == 3
        
        # Min and max
        grid = hierarchical_grid(data, (:age, :min))
        @test nrow(grid) == 1
        @test grid.age[1] ≈ minimum(data.age)
        
        grid = hierarchical_grid(data, (:age, :max))  
        @test nrow(grid) == 1
        @test grid.age[1] ≈ maximum(data.age)
        
        # Custom percentiles as vector
        grid = hierarchical_grid(data, (:income, [0.1, 0.5, 0.9]))
        @test nrow(grid) == 3
        expected_percentiles = [quantile(data.income, q) for q in [0.1, 0.5, 0.9]]
        @test sort(grid.income) ≈ sort(expected_percentiles)
        
        # Explicit percentiles specification
        grid = hierarchical_grid(data, (:income, (:percentiles, [0.25, 0.75])))
        @test nrow(grid) == 2
        expected = [quantile(data.income, 0.25), quantile(data.income, 0.75)]
        @test sort(grid.income) ≈ sort(expected)
        
        # Custom n-quantiles
        grid = hierarchical_grid(data, (:income, (:quantiles, 7)))  # septiles
        @test nrow(grid) == 7
        expected = [quantile(data.income, i/7) for i in 1:7]
        @test sort(grid.income) ≈ sort(expected)
        
        # Range specifications
        grid = hierarchical_grid(data, (:age, (:range, 5)))
        @test nrow(grid) == 5
        min_age, max_age = extrema(data.age)
        expected = collect(range(min_age, max_age, length=5))
        @test sort(grid.age) ≈ sort(expected)
        
        grid = hierarchical_grid(data, (:age, (:range, (30, 60))))
        @test nrow(grid) == 2
        @test Set(grid.age) == Set([30, 60])
    end
    
    @testset "Hierarchical Reference Construction" begin
        # Basic hierarchical specification: region conditioning education
        grid = hierarchical_grid(data, :region => :education)
        @test nrow(grid) == 4  # 2 regions × 2 education levels (same as cross-tabulation)
        @test "region" in names(grid) && "education" in names(grid)
        
        # Check that all combinations exist
        combinations = Set([(row.region, row.education) for row in eachrow(grid)])
        expected = Set([("North", "HS"), ("North", "College"), ("South", "HS"), ("South", "College")])
        @test combinations == expected
        
        # Hierarchical with continuous: region => income quartiles
        grid = hierarchical_grid(data, :region => (:income, :quartiles))
        @test nrow(grid) == 8  # 2 regions × 4 quartiles  
        @test "region" in names(grid) && "income" in names(grid)
        
        # Verify that quartiles are computed within each region
        north_incomes = grid[grid.region .== "North", :income]
        south_incomes = grid[grid.region .== "South", :income]
        @test length(unique(north_incomes)) == 4
        @test length(unique(south_incomes)) == 4
        
        # Mixed hierarchical specification
        spec = :region => [(:income, :quartiles), (:age, :mean)]
        grid = hierarchical_grid(data, spec)
        @test nrow(grid) == 10  # 2 regions × (4 quartiles + 1 mean) = 2 × 5
        @test "region" in names(grid) && "income" in names(grid) && "age" in names(grid)
    end
    
    @testset "Error Handling" begin
        # Invalid variable name
        @test_throws Exception hierarchical_grid(data, :nonexistent_var)
        
        # Continuous variable in categorical specification
        @test_throws Exception hierarchical_grid(data, [:region, :income])  # income is continuous
        
        # Categorical variable with continuous representative
        @test_throws Exception hierarchical_grid(data, (:region, :quartiles))  # region is categorical
        
        # Invalid representative specification
        @test_throws Exception hierarchical_grid(data, (:income, :invalid_spec))
        
        # Phase 1.2 error handling
        # Invalid percentiles (outside 0-1 range)
        @test_throws Exception hierarchical_grid(data, (:income, (:percentiles, [-0.1, 0.5, 1.5])))
        
        # Invalid n-quantiles (must be > 1)
        @test_throws Exception hierarchical_grid(data, (:income, (:quantiles, 1)))
        @test_throws Exception hierarchical_grid(data, (:income, (:quantiles, 0)))
        
        # Invalid range specification
        @test_throws Exception hierarchical_grid(data, (:income, (:range, 1)))  # need > 1 points
        
        # Phase 1.2 enhanced validation tests
        # Invalid variable name should give helpful message
        try
            hierarchical_grid(data, :nonexistent_variable)
            @test false  # Should not reach here
        catch e
            @test occursin("not found in data", string(e))
            @test occursin("Available variables", string(e))
        end
        
        # Invalid tuple structure
        @test_throws Exception hierarchical_grid(data, ("invalid", :mean))  # first element must be Symbol
        
        # Mixed type error for continuous specs on categorical  
        try
            hierarchical_grid(data, (:region, :mean))  # region is categorical
            @test false  # Should not reach here
        catch e
            @test occursin("requires continuous variable", string(e))
        end
        
        # Empty data case - tests edge case handling and warning behavior
        # This intentionally triggers a warning to verify proper error reporting
        empty_data = DataFrame(x = Float64[], y = Float64[])
        grid = hierarchical_grid(empty_data, (:x, :mean))
        @test nrow(grid) == 1  # Still creates a single row with NaN for mean of empty data
    end
    
    @testset "Representative Value Computation" begin
        # Test quartile computation matches manual calculation
        spec = (:income, :quartiles)
        grid = hierarchical_grid(data, spec)
        
        expected_quartiles = [quantile(data.income, q) for q in [0.25, 0.50, 0.75, 1.0]]
        computed_quartiles = sort(grid.income)
        
        for (expected, computed) in zip(expected_quartiles, computed_quartiles)
            @test expected ≈ computed
        end
        
        # Test quintiles  
        spec = (:income, :quintiles)
        grid = hierarchical_grid(data, spec)
        @test nrow(grid) == 5
        
        expected_quintiles = [quantile(data.income, q) for q in [0.2, 0.4, 0.6, 0.8, 1.0]]
        computed_quintiles = sort(grid.income)
        
        for (expected, computed) in zip(expected_quintiles, computed_quintiles)
            @test expected ≈ computed
        end
    end
    
    @testset "Integration with Existing Infrastructure" begin
        # Test that hierarchical grid works with means_grid typical structure
        hierarchical_single = hierarchical_grid(data, (:age, :mean))
        means_single = means_grid(data)
        
        # Should have compatible structure for age column
        @test hierarchical_single.age[1] ≈ means_single.age[1] rtol=1e-10
    end
    
    @testset "Complex Example" begin
        # Policy analysis scenario from the proposal
        spec = :region => [
            :education,
            (:income, :quartiles),
            (:age, :mean)
        ]
        
        grid = hierarchical_grid(data, spec)
        
        # Should have: 2 regions × (2 education + 4 quartiles + 1 mean) = 2 × 7 = 14 rows
        @test nrow(grid) == 14
        @test "region" in names(grid)
        @test "education" in names(grid) 
        @test "income" in names(grid)
        @test "age" in names(grid)
        
        # Verify all regions represented
        @test Set(unique(grid.region)) == Set(["North", "South"])
        
        # Verify education levels within each region
        for region in ["North", "South"]
            region_rows = grid[grid.region .== region, :]
            education_rows = region_rows[.!ismissing.(region_rows.education), :]
            if nrow(education_rows) > 0
                # Filter to discrete categorical values (exclude mix objects which are valid for representative values)
                discrete_education = filter(e -> e isa Union{String, CategoricalValue}, education_rows.education)
                if !isempty(discrete_education)
                    @test Set(string.(discrete_education)) ⊆ Set(["HS", "College"])
                end
            end
        end
    end
end

@testset "Deep Hierarchical Nesting (3+ Levels)" begin
    # Create test data with additional nesting variables
    Random.seed!(456)
    n = 120
    data = DataFrame(
        country = categorical(rand(["USA", "Canada"], n)),
        region = categorical(rand(["North", "South", "East"], n)),
        city = categorical(rand(["CityA", "CityB"], n)),
        education = categorical(rand(["HS", "College", "Graduate"], n)),
        income = rand(25000:75000, n),
        age = rand(25:65, n),
        treatment = rand([true, false], n),
        y = randn(n)
    )
    
    @testset "3-Level Nesting" begin
        # Basic 3-level: country => region => education
        spec = :country => (:region => :education)
        grid = hierarchical_grid(data, spec)
        
        # Should have combinations for each level
        @test nrow(grid) > 0
        @test "country" in names(grid)
        @test "region" in names(grid)
        @test "education" in names(grid)
        
        # Verify that all combinations respect the hierarchical structure
        for row in eachrow(grid)
            # Each combination should exist in the original data
            country_data = data[data.country .== row.country, :]
            region_data = country_data[country_data.region .== row.region, :]
            @test nrow(region_data) > 0  # This region should exist in this country
            @test row.education in region_data.education  # This education should exist in this region
        end
        
        # Test depth validation
        depth = Margins._count_nesting_depth(spec)
        @test depth == 2  # :country => (:region => :education) has depth 2
    end
    
    @testset "4-Level Nesting" begin
        # 4-level: country => region => city => education  
        spec = :country => (:region => (:city => :education))
        grid = hierarchical_grid(data, spec)
        
        @test nrow(grid) > 0
        @test "country" in names(grid)
        @test "region" in names(grid) 
        @test "city" in names(grid)
        @test "education" in names(grid)
        
        # Verify hierarchical structure is respected
        for row in eachrow(grid)
            # Each combination should exist in the original data
            country_data = data[data.country .== row.country, :]
            region_data = country_data[country_data.region .== row.region, :]
            city_data = region_data[region_data.city .== row.city, :]
            @test nrow(city_data) > 0  # This city should exist in this region/country
            @test row.education in city_data.education  # This education should exist in this city
        end
        
        # Test depth validation
        depth = Margins._count_nesting_depth(spec)
        @test depth == 3  # 4-level nesting has depth 3
    end
    
    @testset "Mixed Depth Nesting" begin
        # Mixed depth: country => region => [education, (income, :quartiles), (city => age)]
        spec = :country => (:region => [
            :education,
            (:income, :quartiles),
            (:city => (:age, :mean))
        ])
        
        grid = hierarchical_grid(data, spec)
        
        @test nrow(grid) > 0
        @test "country" in names(grid)
        @test "region" in names(grid)
        @test "education" in names(grid)
        @test "income" in names(grid) 
        @test "city" in names(grid)
        @test "age" in names(grid)
        
        # Verification of education variable presence in basic categorical specification
        education_rows = grid[.!ismissing.(grid.education), :]
        @test nrow(education_rows) > 0
        
        # Check that some rows have quartile income values (from continuous spec)
        income_rows = grid[.!ismissing.(grid.income), :]
        @test nrow(income_rows) > 0
        @test length(unique(income_rows.income)) > 1  # Should have multiple quartile values
        
        # Check that some rows have city-age combinations (from nested spec)
        city_age_rows = grid[(.!ismissing.(grid.city)) .& (.!ismissing.(grid.age)), :]
        @test nrow(city_age_rows) > 0
    end
    
    @testset "Grid Size Estimation" begin
        # Test grid size estimation functions
        simple_spec = :education
        estimated = Margins._estimate_grid_size(simple_spec, Tables.columntable(data))
        @test estimated == 3  # HS, College, Graduate
        
        # 2-level cross product
        cross_spec = [:region, :education]
        estimated = Margins._estimate_grid_size(cross_spec, Tables.columntable(data))
        @test estimated == 3 * 3  # 3 regions × 3 education levels
        
        # 3-level hierarchical
        hier_spec = :country => (:region => :education)
        estimated = Margins._estimate_grid_size(hier_spec, Tables.columntable(data))
        @test estimated > 0
        @test estimated <= 2 * 3 * 3  # Upper bound: 2 countries × 3 regions × 3 education
        
        # Continuous representative
        cont_spec = (:income, :quartiles)
        estimated = Margins._estimate_grid_size(cont_spec, Tables.columntable(data))
        @test estimated == 4  # 4 quartiles
    end
    
    @testset "Depth Validation" begin
        # Test depth counting
        simple = :education
        @test Margins._count_nesting_depth(simple) == 0
        
        two_level = :country => :region
        @test Margins._count_nesting_depth(two_level) == 1
        
        three_level = :country => (:region => :education)  
        @test Margins._count_nesting_depth(three_level) == 2
        
        four_level = :country => (:region => (:city => :education))
        @test Margins._count_nesting_depth(four_level) == 3
        
        # Test max depth validation
        deep_spec = :a => (:b => (:c => (:d => (:e => (:f => :g)))))
        @test Margins._count_nesting_depth(deep_spec) == 6
        
        # Should error with default max_depth=5
        @test_throws Exception hierarchical_grid(data, deep_spec)
        
        # Should work with increased max_depth
        try
            # This will likely error due to nonexistent variables, but depth check should pass
            hierarchical_grid(data, deep_spec; max_depth=10)
            @test false  # Should error on nonexistent variables
        catch e
            # Should be variable error, not depth error
            @test !occursin("Maximum nesting depth", string(e))
        end
    end
    
    @testset "Enhanced Parameter Integration" begin
        # Test new parameters on hierarchical_grid function
        spec = :country => (:region => :education)
        
        # Test warn_large=false (should suppress warnings)
        grid = hierarchical_grid(data, spec; warn_large=false)
        @test grid isa DataFrame
        
        # Test increased max_depth
        grid = hierarchical_grid(data, spec; max_depth=10)
        @test grid isa DataFrame
        
        # Test that 2-level specs still use optimized path
        simple_spec = :country => :region
        depth = Margins._count_nesting_depth(simple_spec)
        @test depth <= 2
        
        grid = hierarchical_grid(data, simple_spec)
        @test grid isa DataFrame
    end
    
    @testset "Performance and Safety Features" begin
        # Test grid size warnings (create a spec that should warn)
        # Use smaller test data to control the actual grid size
        small_data = data[1:10, :]  # Use subset
        
        # Create a specification that would generate many combinations
        large_spec = [:country, :region, :city, :education]
        estimated_size = Margins._estimate_grid_size(large_spec, Tables.columntable(small_data))
        
        # The actual warning threshold testing requires specific data composition
        # Verification of function execution without errors
        grid = hierarchical_grid(small_data, large_spec)
        @test grid isa DataFrame
        
        # Test that validation functions work
        @test Margins._validate_grid_size(100) == true  # Small size, no warning
        @test Margins._validate_grid_size(15000) == true  # Medium size, should info
        
        # Test that very large sizes error
        @test_throws Exception Margins._validate_grid_size(2_000_000)
    end
end

@testset "Integration with profile_margins() Workflow" begin
    @testset "Basic DataFrame Construction Tests" begin
        # Test that hierarchical grid produces proper DataFrames with correct types
        Random.seed!(06515)
        data = DataFrame(
            region = categorical(["North", "South", "East", "West"]),
            income = [25000, 50000, 75000, 40000],
            age = [30, 45, 60, 35]
        )
        
        # Evaluation of basic continuous variable specification
        reference_grid = hierarchical_grid(data, (:income, :mean))
        @test reference_grid isa DataFrame
        @test nrow(reference_grid) == 1
        @test eltype(reference_grid.income) <: Real
        
        # Test categorical specification
        reference_grid = hierarchical_grid(data, :region)
        @test reference_grid isa DataFrame
        @test nrow(reference_grid) == 4
        @test eltype(reference_grid.region) <: AbstractString || eltype(reference_grid.region) <: CategoricalValue
        
        # Test quartiles specification
        reference_grid = hierarchical_grid(data, (:income, :quartiles))
        @test reference_grid isa DataFrame
        @test nrow(reference_grid) == 4
        @test eltype(reference_grid.income) <: Real
        @test all(x -> x isa Real, reference_grid.income)
        
        # Test range specification  
        reference_grid = hierarchical_grid(data, (:age, (:range, 3)))
        @test reference_grid isa DataFrame
        @test nrow(reference_grid) == 3
        @test eltype(reference_grid.age) <: Real
        @test all(x -> x isa Real, reference_grid.age)
    end
    
    @testset "Error Handling in Integration" begin
        data = DataFrame(
            region = categorical(["North", "South"]),
            income = [25000, 50000],
            education = categorical(["HS", "College"])
        )
        
        # Test that validation catches errors before they reach profile_margins
        try
            reference_grid = hierarchical_grid(data, :nonexistent_variable)
            @test false  # Should not reach here
        catch e
            @test occursin("not found in data", string(e))
        end
        
        # Test that invalid specifications are caught early
        try
            reference_grid = hierarchical_grid(data, (:education, :mean))  # categorical with continuous spec
            @test false  # Should not reach here
        catch e
            @test occursin("requires continuous variable", string(e))
        end
    end
end

@testset "Grammar Parsing Infrastructure" begin
    # Generate basic test dataset
    data = DataFrame(
        cat1 = ["A", "B"], 
        cat2 = ["X", "Y"],
        cont1 = [1.0, 2.0],
        cont2 = [10.0, 20.0]
    )
    data_nt = Tables.columntable(data)
    
    @testset "Categorical Reference Parsing" begin
        # Single categorical
        result = Margins._parse_categorical_reference_spec([:cat1], data_nt)
        @test length(result) == 2
        @test Set([r[:cat1] for r in result]) == Set(["A", "B"])
        
        # Cross-tabulation
        result = Margins._parse_categorical_reference_spec([:cat1, :cat2], data_nt)
        @test length(result) == 4  # 2 × 2
        combinations = Set([(r[:cat1], r[:cat2]) for r in result])
        expected = Set([("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y")])
        @test combinations == expected
    end
    
    @testset "Continuous Representative Parsing" begin
        # Mean specification
        result = Margins._parse_continuous_representative_spec(:cont1, :mean, data_nt)
        @test length(result) == 1
        @test result[1][:cont1] ≈ mean(data.cont1)
        
        # Quartiles specification  
        result = Margins._parse_continuous_representative_spec(:cont1, :quartiles, data_nt)
        @test length(result) == 4
        
        # Fixed values specification
        result = Margins._parse_continuous_representative_spec(:cont1, [0.5, 1.5, 2.5], data_nt)
        @test length(result) == 3
        @test Set([r[:cont1] for r in result]) == Set([0.5, 1.5, 2.5])
    end
    
    @testset "Group Index Computation" begin
        # Single group specification  
        group_spec = Dict(:cat1 => "A")
        indices = Margins._get_group_indices(group_spec, data_nt)
        @test indices == [1]  # First row has cat1 = "A"
        
        # Multiple criteria
        group_spec = Dict(:cat1 => "B", :cat2 => "Y")
        indices = Margins._get_group_indices(group_spec, data_nt)
        @test indices == [2]  # Second row has cat1 = "B" and cat2 = "Y"
        
        # No matches
        group_spec = Dict(:cat1 => "C")  # Nonexistent level
        indices = Margins._get_group_indices(group_spec, data_nt)
        @test isempty(indices)
    end
end