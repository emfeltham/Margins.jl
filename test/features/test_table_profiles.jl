using Test
using Random
using Statistics
using DataFrames, GLM, StatsModels
using Margins

# Profile estimates and SEs on a DataFrame grid are checked exactly against closed forms for
# every link, scale, measure and backend by validation/test_feature_matrix_oracle.jl
# (cartesian_grid and means_grid). Here: a hand-built table equals the cartesian_grid, and
# empty grids are errors (type=:invalid is in core/test_errors.jl).
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

    @testset "Error Handling" begin
        # Empty reference grid
        empty_grid = DataFrame()
        @test_throws ArgumentError profile_margins(m, df, empty_grid; type=:effects)

        # Zero-row reference grid
        zero_grid = DataFrame(x1=Float64[], x2=Float64[], treated=Bool[])
        @test_throws ArgumentError profile_margins(m, df, zero_grid; type=:effects)
    end
end
