# Test for HP Issue #2: vcov Failure Error Behavior (No Fallbacks)

using Margins, GLM, DataFrames, Test, Random, Logging, StatsBase

@testset "vcov Failure Error Behavior" begin

    # Setup test data
    Random.seed!(789)
    n = 50
    df = DataFrame(
        x1 = randn(n),
        x2 = randn(n)
    )
    df.y = 0.5 * df.x1 + 0.3 * df.x2 + randn(n) * 0.1

    # Fit normal model (should work without warnings)
    model_good = lm(@formula(y ~ x1 + x2), df)

    @testset "Normal Model - No Errors" begin
        # This should work without any errors
        result = population_margins(model_good, df; type=:effects, vars=[:x1])
        
        # Should produce valid results
        @test nrow(DataFrame(result)) == 1
        @test DataFrame(result).se[1] > 0
    end

    @testset "Broken vcov - Error Thrown" begin
        # Create a mock model that will fail vcov() call
        struct MockBrokenModel
            coef::Vector{Float64}
        end
        
        # This will fail when StatsBase.vcov is called
        Base.show(io::IO, m::MockBrokenModel) = print(io, "MockBrokenModel")
        StatsModels.coef(m::MockBrokenModel) = m.coef
        StatsBase.vcov(m::MockBrokenModel) = error("Mock vcov failure")
        
        mock_model = MockBrokenModel([1.0, 0.5, 0.3])  # Intercept + 2 coefficients
        
        # Test direct utility function - should error, not return fallback
        try
            Σ = Margins._vcov_model(mock_model, 3)
            @test false  # Should not reach here
        catch e
            @test isa(e, ErrorException)
            error_msg = string(e)
            @test occursin("Failed to extract covariance matrix", error_msg)
            @test occursin("Mock vcov failure", error_msg)
            @test occursin("Solutions:", error_msg)
            @test occursin("vcov=your_covariance_matrix", error_msg)
            @test occursin("Wrong standard errors are worse than no standard errors", error_msg)
        end
    end

    @testset "Error Appears in Full Pipeline" begin
        # Test that _resolve_vcov also throws errors (no fallbacks)
        struct AnotherMockModel
            beta::Vector{Float64}
        end
        StatsModels.coef(m::AnotherMockModel) = m.beta
        StatsBase.vcov(::AnotherMockModel) = error("Simulated vcov failure")
        
        # Test that _resolve_vcov also errors
        try
            Σ = Margins._resolve_vcov(:model, AnotherMockModel([0.1, 0.2]), 2)
            @test false  # Should not reach here
        catch e
            @test isa(e, ErrorException)
            @test occursin("Failed to extract covariance matrix", string(e))
            @test occursin("Simulated vcov failure", string(e))
        end
    end

    @testset "Custom vcov Parameter Works Correctly" begin
        # When user provides explicit vcov, no errors should occur
        custom_vcov = [0.1 0.0 0.0; 0.0 0.2 0.0; 0.0 0.0 0.3]
        
        # Should work fine with explicit vcov
        result = population_margins(model_good, df; 
                                  type=:effects, 
                                  vars=[:x1], 
                                  vcov=custom_vcov)
        
        # Should use the custom vcov (different SEs than model vcov)
        @test DataFrame(result).se[1] > 0
        @test nrow(DataFrame(result)) == 1
    end

    @testset "Error Message Quality" begin
        # Test that the error provides actionable guidance
        struct TestBrokenModel
            coefficients::Vector{Float64}
        end
        StatsModels.coef(m::TestBrokenModel) = m.coefficients
        StatsBase.vcov(::TestBrokenModel) = throw(DomainError("Test error"))
        
        try
            Margins._vcov_model(TestBrokenModel([1.0, 2.0]), 2)
            @test false  # Should not reach here
        catch e
            error_msg = string(e)
            
            # Check that error details are included
            @test occursin("DomainError", error_msg)
            @test occursin("Test error", error_msg)
            
            # Check that solutions are provided
            @test occursin("Solutions:", error_msg)
            @test occursin("Provide explicit covariance matrix", error_msg)
            @test occursin("Ensure your model was fitted properly", error_msg)
            @test occursin("CovarianceMatrices.jl", error_msg)
        end
    end
end