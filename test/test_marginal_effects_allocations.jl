# Test to verify FormulaCompiler.marginal_effects_mu! allocation behavior

using Test, Margins, GLM, DataFrames, CategoricalArrays, Tables
using FormulaCompiler

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
    compiled = FormulaCompiler.compile_formula(model, data_nt)
    continuous_vars = FormulaCompiler.continuous_variables(compiled, data_nt)
    de = FormulaCompiler.build_derivative_evaluator(compiled, data_nt; vars=continuous_vars)
    
    β = coef(model)
    g_buf = Vector{Float64}(undef, length(continuous_vars))
    
    @testset "Single Call Allocation Test" begin
        # Warm up
        FormulaCompiler.marginal_effects_mu!(g_buf, de, β, 1; link=GLM.IdentityLink(), backend=:ad)
        
        # Test single call allocation
        allocs_single = @allocated FormulaCompiler.marginal_effects_mu!(g_buf, de, β, 1; link=GLM.IdentityLink(), backend=:ad)
        
        @test allocs_single == 0  # Should be zero allocation according to FormulaCompiler tests
        println("Single marginal_effects_mu! call: $allocs_single bytes")
    end
    
    @testset "Loop Allocation Test" begin
        function test_marginal_effects_loop(g_buf, de, β, rows)
            for row in rows
                FormulaCompiler.marginal_effects_mu!(g_buf, de, β, row; link=GLM.IdentityLink(), backend=:ad)
            end
        end
        
        rows = 1:100  # Test 100 rows
        
        # Warm up
        test_marginal_effects_loop(g_buf, de, β, 1:10)
        
        # Test loop allocation
        allocs_loop = @allocated test_marginal_effects_loop(g_buf, de, β, rows)
        allocs_per_call = allocs_loop / length(rows)
        
        println("Loop marginal_effects_mu! calls:")
        println("  Total: $allocs_loop bytes")
        println("  Per call: $allocs_per_call bytes")
        
        # This should be zero or very small if FormulaCompiler is efficient
        @test allocs_per_call < 1000  # Allow some overhead but shouldn't be massive
    end
    
    @testset "Backend Comparison" begin
        # Test AD backend
        allocs_ad = @allocated FormulaCompiler.marginal_effects_mu!(g_buf, de, β, 1; link=GLM.IdentityLink(), backend=:ad)
        
        # Test FD backend  
        allocs_fd = @allocated FormulaCompiler.marginal_effects_mu!(g_buf, de, β, 1; link=GLM.IdentityLink(), backend=:fd)
        
        println("Backend allocation comparison:")
        println("  AD backend: $allocs_ad bytes")
        println("  FD backend: $allocs_fd bytes")
        
        # FD should definitely be zero allocation
        @test allocs_fd == 0
    end
    
    @testset "Derivative Evaluator Size Impact" begin
        # Test with small dataset
        small_df = df[1:10, :]
        small_data = Tables.columntable(small_df)
        small_compiled = FormulaCompiler.compile_formula(model, small_data)
        small_vars = FormulaCompiler.continuous_variables(small_compiled, small_data)
        small_de = FormulaCompiler.build_derivative_evaluator(small_compiled, small_data; vars=small_vars)
        small_g_buf = Vector{Float64}(undef, length(small_vars))
        
        # Warm up
        FormulaCompiler.marginal_effects_mu!(small_g_buf, small_de, β, 1; link=GLM.IdentityLink(), backend=:ad)
        
        # Test small dataset allocation
        allocs_small = @allocated FormulaCompiler.marginal_effects_mu!(small_g_buf, small_de, β, 1; link=GLM.IdentityLink(), backend=:ad)
        
        # Test large dataset allocation (original de)
        allocs_large = @allocated FormulaCompiler.marginal_effects_mu!(g_buf, de, β, 1; link=GLM.IdentityLink(), backend=:ad)
        
        println("Dataset size impact:")
        println("  Small dataset (n=10): $allocs_small bytes") 
        println("  Large dataset (n=1000): $allocs_large bytes")
        
        # If derivative evaluator size causes allocations, we'll see it here
        if allocs_large > allocs_small
            println("  ⚠️  Large dataset causes more allocations!")
        else
            println("  ✅  Dataset size doesn't affect allocations")
        end
    end
end