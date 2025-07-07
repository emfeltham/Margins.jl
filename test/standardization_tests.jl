@testset "Integration with Margins.jl" begin
    # Test that the ZScoredTerm support enables Margins.jl to work
    
    contrasts = Dict(:x => ZScore(), :y => ZScore())
    
    f = @formula(y ~ x + z)
    m = lm(f, df; contrasts=contrasts)
    
    # This should work without the original MethodError
    @test_nowarn margins(m, :x, df)
    
    # Test with specific variable
    ame_result = margins(m, :x, df)
    @test ame_result isa MarginsResult
    
    # The AME should be approximately the coefficient for standardized variable
    coefs = coef(m)
    coef_names = coefnames(m)
    x_coef_idx = findfirst(contains("x"), coef_names)
    
    if !isnothing(x_coef_idx)
        # For standardized variables, AME should be close to the coefficient
        @test abs(ame_result.effects[:x] - coefs[x_coef_idx]) < 0.1
    end
end
