using Test
using Random
using Statistics
using DataFrames, GLM, StatsModels
using Margins

@testset "Table-based Profile Margins" begin
    Random.seed!(12345)
    n = 200
    df = DataFrame(
        x1 = randn(n),
        x2 = randn(n),
        treated = rand(Bool, n),
        y = rand(Bool, n)
    )
    
    m = glm(@formula(y ~ x1 + x2 + treated), df, Binomial(), LogitLink())
    
    @testset "Basic Table-based Effects" begin
        # Create reference grid
        reference_grid = DataFrame(
            x1 = [0.0, 1.0, 0.0, 1.0],
            x2 = [0.5, 0.5, -0.5, -0.5], 
            treated = [true, false, true, false]
        )
        
        result = profile_margins(m, df, reference_grid; type=:effects, vars=[:x1])
        
        @test nrow(DataFrame(result)) == 4  # 4 scenarios
        @test all(isfinite, DataFrame(result).estimate)
        @test all(DataFrame(result).se .> 0)
        # Check term descriptions contain x1 variable
        @test all(occursin.("x1", DataFrame(result).variable))
        
        # Check profile columns are present
        @test "x1" in names(DataFrame(result))
        @test "x2" in names(DataFrame(result))  
        @test "treated" in names(DataFrame(result))
        
        # Check profile values match reference grid
        @test Set(DataFrame(result).x1) == Set([0.0, 1.0])
        @test Set(DataFrame(result).x2) == Set([0.5, -0.5])
        @test Set(DataFrame(result).treated) == Set([true, false])
    end
    
    @testset "Table-based Predictions" begin
        reference_grid = DataFrame(
            x1 = [-1.0, 0.0, 1.0],
            x2 = [0.0, 0.0, 0.0],
            treated = [false, false, false]
        )
        
        result_response = profile_margins(m, df, reference_grid; type=:predictions, scale=:response)
        result_link = profile_margins(m, df, reference_grid; type=:predictions, scale=:link)
        
        @test nrow(DataFrame(result_response)) == 3
        @test nrow(DataFrame(result_link)) == 3
        @test all(0.0 .<= DataFrame(result_response).estimate .<= 1.0)  # Response scale bounded
        @test all(isfinite, DataFrame(result_link).estimate)  # Link scale unbounded but finite
        
        # Should be different scales
        @test maximum(abs.(DataFrame(result_response).estimate .- DataFrame(result_link).estimate)) > 0.1
    end
    
    @testset "Equivalence with Grid-based Approach" begin
        # Cartesian grid specification
        result_dict = profile_margins(m, df, cartesian_grid(x1=[0.0, 1.0], x2=[0.5, -0.5], treated=[true, false]); type=:effects, vars=[:x1])
        
        # Equivalent table specification (Cartesian product)
        reference_grid = DataFrame(
            x1 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            x2 = [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5],
            treated = [true, false, true, false, true, false, true, false]
        )
        result_table = profile_margins(m, df, reference_grid; type=:effects, vars=[:x1])
        
        @test nrow(DataFrame(result_dict)) == nrow(DataFrame(result_table)) == 8
        
        # Sort both DataFrames for comparison
        df_dict = DataFrame(result_dict)
        df_table = DataFrame(result_table)
        sort!(df_dict, [:x1, :x2, :treated])
        sort!(df_table, [:x1, :x2, :treated])
        
        # Should produce identical results
        @test maximum(abs.(df_dict.estimate .- df_table.estimate)) < 1e-12
        @test maximum(abs.(df_dict.se .- df_table.se)) < 1e-12
    end
    
    @testset "Averaging Functionality" begin
        reference_grid = DataFrame(
            x1 = [0.0, 0.5, 1.0],
            x2 = [0.0, 0.0, 0.0],
            treated = [false, false, false]
        )
        
        result_no_avg = profile_margins(m, df, reference_grid; type=:effects, vars=[:x1])
        # Population margins for averaging comparison
        result_avg = population_margins(m, df; type=:effects, vars=[:x1])
        
        @test nrow(DataFrame(result_no_avg)) == 3
        @test nrow(DataFrame(result_avg)) == 1  # Averaged to single row
        @test all(isfinite, DataFrame(result_avg).estimate)
        @test all(DataFrame(result_avg).se .> 0)
    end
    
    @testset "Multiple Variables" begin
        reference_grid = DataFrame(
            x1 = [0.0, 1.0],
            x2 = [0.0, 0.0], 
            treated = [false, false]
        )
        
        result = profile_margins(m, df, reference_grid; type=:effects, vars=[:x1, :x2])
        
        @test nrow(DataFrame(result)) == 4  # 2 scenarios × 2 variables
        # Check both variables are present in term descriptions
        terms = DataFrame(result).variable
        @test any(occursin.("x1", terms))
        @test any(occursin.("x2", terms))
        @test all(isfinite, DataFrame(result).estimate)
        @test all(DataFrame(result).se .> 0)
    end
    
    @testset "Error Handling" begin
        # Empty reference grid
        empty_grid = DataFrame()
        @test_throws ArgumentError profile_margins(m, df, empty_grid; type=:effects)
        
        # Zero-row reference grid
        zero_grid = DataFrame(x1=Float64[], x2=Float64[], treated=Bool[])
        @test_throws ArgumentError profile_margins(m, df, zero_grid; type=:effects)
        
        # Invalid type
        valid_grid = DataFrame(x1=[0.0], x2=[0.0], treated=[false])
        @test_throws ArgumentError profile_margins(m, df, valid_grid; type=:invalid)
    end
    
    @testset "Different Link Types" begin
        reference_grid = DataFrame(
            x1 = [0.0, 1.0],
            x2 = [0.0, 0.0],
            treated = [false, false]
        )
        
        # Test with different link functions
        m_probit = glm(@formula(y ~ x1 + x2 + treated), df, Binomial(), ProbitLink())
        
        result_link = profile_margins(m_probit, df, reference_grid; type=:effects, vars=[:x1], scale=:link)
        result_response = profile_margins(m_probit, df, reference_grid; type=:effects, vars=[:x1], scale=:response)
        
        @test nrow(DataFrame(result_link)) == 2
        @test nrow(DataFrame(result_response)) == 2
        @test all(isfinite, DataFrame(result_link).estimate)
        @test all(isfinite, DataFrame(result_response).estimate)
        
        # Effects should be different for ProbitLink
        @test maximum(abs.(DataFrame(result_link).estimate .- DataFrame(result_response).estimate)) > 0.01
    end
    
    @testset "Continuous Variables Only" begin
        # Table-based approach currently only supports continuous variables
        # Bool variables would trigger FormulaCompiler derivative issues
        reference_grid = DataFrame(
            x1 = [0.0, 1.0],
            x2 = [0.0, 0.0],
            treated = [false, true]
        )
        
        # Only request continuous variables (:continuous auto-detection should work)
        result = profile_margins(m, df, reference_grid; type=:effects, vars=[:x1, :x2])  # Explicit continuous vars
        
        # Should only include x1 and x2 (continuous variables)
        terms = DataFrame(result).variable
        @test any(occursin.("x1", terms))
        @test any(occursin.("x2", terms))
        @test nrow(DataFrame(result)) == 4  # 2 scenarios × 2 continuous vars
        @test all(isfinite, DataFrame(result).estimate)
    end
end