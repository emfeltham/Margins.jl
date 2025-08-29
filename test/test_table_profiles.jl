using Test
using Random
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
        
        @test nrow(result.table) == 4  # 4 scenarios
        @test all(isfinite, result.table.dydx)
        @test all(result.table.se .> 0)
        @test result.table.term == fill(:x1, 4)
        
        # Check profile columns are present
        @test "at_x1" in names(result.table)
        @test "at_x2" in names(result.table)  
        @test "at_treated" in names(result.table)
        
        # Check profile values match reference grid
        @test Set(result.table.at_x1) == Set([0.0, 1.0])
        @test Set(result.table.at_x2) == Set([0.5, -0.5])
        @test Set(result.table.at_treated) == Set([true, false])
    end
    
    @testset "Table-based Predictions" begin
        reference_grid = DataFrame(
            x1 = [-1.0, 0.0, 1.0],
            x2 = [0.0, 0.0, 0.0],
            treated = [false, false, false]
        )
        
        result_response = profile_margins(m, df, reference_grid; type=:predictions, scale=:response)
        result_link = profile_margins(m, df, reference_grid; type=:predictions, scale=:link)
        
        @test nrow(result_response.table) == 3
        @test nrow(result_link.table) == 3
        @test all(0.0 .<= result_response.table.dydx .<= 1.0)  # Response scale bounded
        @test all(isfinite, result_link.table.dydx)  # Link scale unbounded but finite
        
        # Should be different scales
        @test maximum(abs.(result_response.table.dydx .- result_link.table.dydx)) > 0.1
    end
    
    @testset "Equivalence with Dict-based Approach" begin
        # Dict specification
        dict_spec = Dict(
            :x1 => [0.0, 1.0],
            :x2 => [0.5, -0.5],
            :treated => [true, false]
        )
        result_dict = profile_margins(m, df; at=dict_spec, type=:effects, vars=[:x1])
        
        # Equivalent table specification (Cartesian product)
        reference_grid = DataFrame(
            x1 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            x2 = [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5],
            treated = [true, false, true, false, true, false, true, false]
        )
        result_table = profile_margins(m, df, reference_grid; type=:effects, vars=[:x1])
        
        @test nrow(result_dict.table) == nrow(result_table.table) == 8
        
        # Sort both for comparison
        sort!(result_dict.table, [:at_x1, :at_x2, :at_treated])
        sort!(result_table.table, [:at_x1, :at_x2, :at_treated])
        
        # Should produce identical results
        @test maximum(abs.(result_dict.table.dydx .- result_table.table.dydx)) < 1e-12
        @test maximum(abs.(result_dict.table.se .- result_table.table.se)) < 1e-12
    end
    
    @testset "Averaging Functionality" begin
        reference_grid = DataFrame(
            x1 = [0.0, 0.5, 1.0],
            x2 = [0.0, 0.0, 0.0],
            treated = [false, false, false]
        )
        
        result_no_avg = profile_margins(m, df, reference_grid; type=:effects, vars=[:x1], average=false)
        result_avg = profile_margins(m, df, reference_grid; type=:effects, vars=[:x1], average=true)
        
        @test nrow(result_no_avg.table) == 3
        @test nrow(result_avg.table) == 1  # Averaged to single row
        @test all(isfinite, result_avg.table.dydx)
        @test all(result_avg.table.se .> 0)
    end
    
    @testset "Multiple Variables" begin
        reference_grid = DataFrame(
            x1 = [0.0, 1.0],
            x2 = [0.0, 0.0], 
            treated = [false, false]
        )
        
        result = profile_margins(m, df, reference_grid; type=:effects, vars=[:x1, :x2])
        
        @test nrow(result.table) == 4  # 2 scenarios × 2 variables
        @test Set(result.table.term) == Set([:x1, :x2])
        @test all(isfinite, result.table.dydx)
        @test all(result.table.se .> 0)
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
        
        result_eta = profile_margins(m_probit, df, reference_grid; type=:effects, vars=[:x1], target=:eta)
        result_mu = profile_margins(m_probit, df, reference_grid; type=:effects, vars=[:x1], target=:mu)
        
        @test nrow(result_eta.table) == 2
        @test nrow(result_mu.table) == 2
        @test all(isfinite, result_eta.table.dydx)
        @test all(isfinite, result_mu.table.dydx)
        
        # Effects should be different for ProbitLink
        @test maximum(abs.(result_eta.table.dydx .- result_mu.table.dydx)) > 0.01
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
        result = profile_margins(m, df, reference_grid; type=:effects, vars=:continuous)
        
        # Should only include x1 and x2 (continuous variables)
        @test Set(result.table.term) == Set([:x1, :x2])
        @test nrow(result.table) == 4  # 2 scenarios × 2 continuous vars
        @test all(isfinite, result.table.dydx)
    end
end