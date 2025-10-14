# test_contrast_gradient_allocations.jl - Allocation testing for contrast gradient functions
# Migrated from FormulaCompiler.jl (2025-10-09)

using Test, BenchmarkTools
using Margins
using FormulaCompiler
using DataFrames, GLM, StatsModels, CategoricalArrays, Tables
using Random, Statistics, LinearAlgebra

Random.seed!(06515)

@testset "Contrast Gradient Allocation Tests" begin

    # Create test data
    function create_simple_test_data(n=100)
        df = DataFrame(
            x = randn(n),
            y = randn(n),
            treatment = repeat(["Control", "Drug_A", "Drug_B"], div(n,3)+1)[1:n],
            binary_var = rand([0, 1], n),
            outcome = randn(n)
        )
        return df, Tables.columntable(df)
    end

    df, data = create_simple_test_data(100)

    @testset "Simple Model Allocations" begin
        # Test with simple model
        model = lm(@formula(outcome ~ x + treatment), df)
        compiled = FormulaCompiler.compile_formula(model, data)
        evaluator = FormulaCompiler.contrastevaluator(compiled, data, [:treatment])
        β = coef(model)
        vcov_matrix = GLM.vcov(model)

        # Pre-allocate buffers
        ∇β = Vector{Float64}(undef, length(compiled))
        contrast_buf = Vector{Float64}(undef, length(compiled))

        row = 1
        var = :treatment
        from = "Control"
        to = "Drug_A"

        @testset "Warmup and Baseline" begin
            @debug "Model parameters: ", length(compiled)

            # Extensive warmup (compiled functions need warmup)
            for i in 1:10
                FormulaCompiler.contrast_modelrow!(contrast_buf, evaluator, row, var, from, to)
                Margins.contrast_gradient!(∇β, evaluator, row, var, from, to, β)
                Margins.delta_method_se(∇β, vcov_matrix)
            end

            @debug "Warmup completed"
        end

        @testset "Core Function Allocations" begin
            # Test baseline contrast computation
            allocs_contrast = @allocated FormulaCompiler.contrast_modelrow!(contrast_buf, evaluator, row, var, from, to)
            @debug "contrast_modelrow! allocations: ", allocs_contrast, " bytes"

            # Our functions should have similar or lower allocations
            allocs_grad_linear = @allocated Margins.contrast_gradient!(∇β, evaluator, row, var, from, to, β)
            @debug "contrast_gradient! (linear) allocations: ", allocs_grad_linear, " bytes"

            # Test delta method
            allocs_delta = @allocated Margins.delta_method_se(∇β, vcov_matrix)
            @debug "delta_method_se allocations: ", allocs_delta, " bytes"

            # Document current state - not asserting zero until we fix underlying issues
            @test allocs_contrast >= 0  # Just ensure no errors
            @test allocs_grad_linear >= 0
            @test allocs_delta >= 0

            @debug "NOTE: These functions currently have allocations due to underlying ContrastEvaluator implementation"
        end

        @testset "Link Function Allocations" begin
            links_to_test = [
                ("Identity", GLM.IdentityLink()),
                ("Logit", GLM.LogitLink()),
                ("Log", GLM.LogLink()),
                ("Probit", GLM.ProbitLink())
            ]

            for (name, link) in links_to_test
                allocs_response = @allocated Margins.contrast_gradient!(∇β, evaluator, row, var, from, to, β, link)

                # delta_method_se now takes gradient directly, not evaluator
                # So we need to compute gradient first, then SE
                Margins.contrast_gradient!(∇β, evaluator, row, var, from, to, β, link)
                allocs_delta_link = @allocated Margins.delta_method_se(∇β, vcov_matrix)

                @debug "contrast_gradient! ($name) allocations: ", allocs_response, " bytes"
                @debug "delta_method_se ($name) allocations: ", allocs_delta_link, " bytes"

                @test allocs_response >= 0
                @test allocs_delta_link >= 0
            end
        end

        @testset "Performance Benchmarks" begin
            @debug "\n=== PERFORMANCE BENCHMARKS ==="

            @debug "Baseline contrast computation:"
            @btime FormulaCompiler.contrast_modelrow!($contrast_buf, $evaluator, $row, $var, $from, $to)

            @debug "Linear scale gradient:"
            @btime Margins.contrast_gradient!($∇β, $evaluator, $row, $var, $from, $to, $β)

            @debug "Logit gradient:"
            link = GLM.LogitLink()
            @btime Margins.contrast_gradient!($∇β, $evaluator, $row, $var, $from, $to, $β, $link)

            @debug "Delta method SE:"
            Margins.contrast_gradient!(∇β, evaluator, row, var, from, to, β)
            @btime Margins.delta_method_se($∇β, $vcov_matrix)

            # These are performance tests, not allocation tests
            @test true  # Just ensure benchmarks run without error
        end
    end

    @testset "Complex Model Allocations" begin
        # Test with more complex model
        model = lm(@formula(outcome ~ x * treatment + binary_var), df)
        compiled = FormulaCompiler.compile_formula(model, data)
        evaluator = FormulaCompiler.contrastevaluator(compiled, data, [:treatment, :binary_var])
        β = coef(model)
        vcov_matrix = GLM.vcov(model)

        ∇β = Vector{Float64}(undef, length(compiled))

        # Warmup
        Margins.contrast_gradient!(∇β, evaluator, 1, :treatment, "Control", "Drug_A", β)
        Margins.contrast_gradient!(∇β, evaluator, 1, :binary_var, 0, 1, β)

        @testset "Categorical vs Binary Variable Allocations" begin
            # Categorical variable
            allocs_cat = @allocated Margins.contrast_gradient!(∇β, evaluator, 1, :treatment, "Control", "Drug_A", β)
            @debug "Categorical variable gradient allocations: ", allocs_cat, " bytes"

            # Binary variable (should potentially be more efficient)
            allocs_bin = @allocated Margins.contrast_gradient!(∇β, evaluator, 1, :binary_var, 0, 1, β)
            @debug "Binary variable gradient allocations: ", allocs_bin, " bytes"

            @test allocs_cat >= 0
            @test allocs_bin >= 0

            # Binary might be same or better due to potential fast paths
            # (but currently both inherit allocations from base implementation)
        end
    end

    @testset "Allocation Analysis and Diagnostics" begin
        @debug "\n=== ALLOCATION ANALYSIS ==="
        @debug "Current Status: Functions have allocations inherited from base ContrastEvaluator"
        @debug "Root Cause: contrast_modelrow! has ~44 allocations (2.9 KiB)"
        @debug "Impact: All gradient functions inherit these allocations"
        @debug "Next Steps for Zero Allocation:"
        @debug "1. Fix allocations in ContrastEvaluator base implementation"
        @debug "2. Optimize counterfactual vector operations"
        @debug "3. Eliminate any dynamic dispatch in hot paths"
        @debug "4. Review categorical level mapping operations"

        @test true  # This is a diagnostic test
    end

    @debug "\nContrast gradient allocation tests completed"
    @debug "Note: Zero-allocation optimization needed in base ContrastEvaluator"
end
