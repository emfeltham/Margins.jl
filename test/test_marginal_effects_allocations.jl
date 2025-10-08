# Test to verify marginal_effects_mu! allocation behavior

using Test, Margins, GLM, DataFrames, CategoricalArrays, Tables
using FormulaCompiler
using Margins: marginal_effects_mu!

@testset "FormulaCompiler Marginal Effects Allocation Tests" begin
    
    # Create test data matching our typical usage
    n = 1000  # Reasonable size to test scaling
    df = DataFrame(
        y = randn(n),
        x_continuous = randn(n),
        x_boolean = rand([true, false], n),
        x_categorical = categorical(rand(["A", "B", "C"], n))
    )
    
    model = lm(@formula(y ~ x_continuous * x_boolean * x_categorical), df)
    data_nt = Tables.columntable(df)
    
    # Build FormulaCompiler components like Margins does
    compiled = compile_formula(model, data_nt)
    continuous_vars = continuous_variables(compiled, data_nt)
    de = build_derivative_evaluator(compiled, data_nt; vars=continuous_vars)
    
    β = coef(model)
    g_buf = Vector{Float64}(undef, length(continuous_vars))
    
    @testset "Single Call Allocation Test" begin
        # Warm up
        marginal_effects_mu!(g_buf, de, β, 1; link=GLM.IdentityLink(), backend=:ad)
        
        # Test single call allocation
        allocs_single = @allocated marginal_effects_mu!(g_buf, de, β, 1; link=GLM.IdentityLink(), backend=:ad)
        
        @test allocs_single == 0  # Should be zero allocation according to FormulaCompiler tests
        @info "Single marginal_effects_mu! allocation: $allocs_single bytes"
    end
    
    @testset "Loop Allocation Test" begin
        function test_marginal_effects_loop(g_buf, de, β, rows)
            for row in rows
                marginal_effects_mu!(g_buf, de, β, row; link=GLM.IdentityLink(), backend=:ad)
            end
        end
        
        rows = 1:100  # Test 100 rows
        
        # Warm up
        test_marginal_effects_loop(g_buf, de, β, 1:10)
        
        # Test loop allocation
        allocs_loop = @allocated test_marginal_effects_loop(g_buf, de, β, rows)
        allocs_per_call = allocs_loop / length(rows)
        
        @info "Loop marginal_effects_mu! allocation analysis:"
        @info "Total allocation: $allocs_loop bytes"
        @info "Per-call allocation: $allocs_per_call bytes"
        
        # Allocation should be minimal under efficient FormulaCompiler implementation
        @test allocs_per_call < 1000  # Allowable overhead with reasonable upper bound
    end
    
    @testset "Backend Comparison" begin
        # Test AD backend
        allocs_ad = @allocated marginal_effects_mu!(g_buf, de, β, 1; link=GLM.IdentityLink(), backend=:ad)
        
        # Test FD backend  
        allocs_fd = @allocated marginal_effects_mu!(g_buf, de, β, 1; link=GLM.IdentityLink(), backend=:fd)
        
        @info "Computational backend allocation comparison:"
        @info "Automatic differentiation backend: $allocs_ad bytes"
        @info "Finite differences backend: $allocs_fd bytes"
        
        # FD backend expected to achieve zero allocation
        @test allocs_fd == 0
    end
    
    @testset "Derivative Evaluator Size Impact" begin
        # Test with small dataset
        small_df = df[1:10, :]
        small_data = Tables.columntable(small_df)
        small_compiled = compile_formula(model, small_data)
        small_vars = continuous_variables(small_compiled, small_data)
        small_de = build_derivative_evaluator(small_compiled, small_data; vars=small_vars)
        small_g_buf = Vector{Float64}(undef, length(small_vars))
        
        # Warm up
        marginal_effects_mu!(small_g_buf, small_de, β, 1; link=GLM.IdentityLink(), backend=:ad)
        
        # Test small dataset allocation
        allocs_small = @allocated marginal_effects_mu!(small_g_buf, small_de, β, 1; link=GLM.IdentityLink(), backend=:ad)
        
        # Test large dataset allocation (original de)
        allocs_large = @allocated marginal_effects_mu!(g_buf, de, β, 1; link=GLM.IdentityLink(), backend=:ad)
        
        @info "Dataset size scaling analysis:"
        @info "Small dataset allocation (n=10): $allocs_small bytes"
        @info "Large dataset allocation (n=1000): $allocs_large bytes"
        
        # If derivative evaluator size causes allocations, we'll see it here
        if allocs_large > allocs_small
            @info "Large dataset exhibits increased allocation overhead"
        else
            @info "Dataset size demonstrates allocation invariance"
        end
    end
end