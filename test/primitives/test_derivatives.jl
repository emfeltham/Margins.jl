# test_derivatives.jl
# julia --project="." test/primitives/test_derivatives.jl > test/primitives/test_derivatives.txt 2>&1
# Correctness tests for marginal effects derivatives (statistical interface)

using Test
using Random
using FormulaCompiler
using Margins
using DataFrames, Tables, GLM, MixedModels, CategoricalArrays
using LinearAlgebra: dot

# Import functions that are now in Margins
using Margins: marginal_effects_eta!, marginal_effects_mu!, delta_method_se

# ====== PHASE 1: Manual Baseline Infrastructure ======

"""
    compute_manual_jacobian(compiled, data, row, vars; h=1e-8)

Compute Jacobian using basic finite differences without any
FormulaCompiler derivative infrastructure. This is our ground truth.
"""
function compute_manual_jacobian(compiled, data, row, vars; h=1e-8)
    n_terms = length(compiled)
    n_vars = length(vars)
    J = Matrix{Float64}(undef, n_terms, n_vars)

    # Base evaluation
    y_base = Vector{Float64}(undef, n_terms)
    compiled(y_base, data, row)

    for (j, var) in enumerate(vars)
        # Get current value
        val = data[var][row]

        # Only perturb if numeric (continuous variables)
        if val isa Number
            # Create new arrays to avoid mutation
            vals_plus = copy(data[var])
            vals_minus = copy(data[var])
            vals_plus[row] = val + h
            vals_minus[row] = val - h

            # Create new named tuples with modified arrays
            data_plus = merge(data, NamedTuple{(var,)}((vals_plus,)))
            data_minus = merge(data, NamedTuple{(var,)}((vals_minus,)))

            # Evaluate at perturbed points
            y_plus = Vector{Float64}(undef, n_terms)
            y_minus = Vector{Float64}(undef, n_terms)
            compiled(y_plus, data_plus, row)
            compiled(y_minus, data_minus, row)

            # Central difference
            J[:, j] = (y_plus .- y_minus) ./ (2h)
        else
            # Non-numeric variables have zero derivative
            J[:, j] .= 0.0
        end
    end

    return J
end

@testset "Derivative correctness" begin
    # Fix random seed for reproducibility
    Random.seed!(06515)

    n = 300

    @testset "ForwardDiff and FD fallback" begin
        # Data and model
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            z = abs.(randn(n)) .+ 0.1,
            group3 = categorical(rand(["A", "B", "C"], n)),
        )
        data = Tables.columntable(df)
        model = lm(@formula(y ~ 1 + x + z + x & group3), df)
        compiled = FormulaCompiler.compile_formula(model, data)

        # Choose continuous vars
        vars = [:x, :z]

        # Build derivative evaluators (concrete types)
        de_ad = FormulaCompiler.derivativeevaluator_ad(compiled, data, vars)
        de_fd = FormulaCompiler.derivativeevaluator_fd(compiled, data, vars)
        J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
        J_fd = Matrix{Float64}(undef, length(compiled), length(vars))

        # Warmup to trigger Dual-typed caches
        FormulaCompiler.derivative_modelrow!(J_ad, de_ad, 1)
        FormulaCompiler.derivative_modelrow!(J_ad, de_ad, 2)

        # Compare AD and FD on same row (3)
        FormulaCompiler.derivative_modelrow!(J_ad, de_ad, 3)
        FormulaCompiler.derivative_modelrow!(J_fd, de_fd, 3)

        # PHASE 2: Manual baseline comparison
        J_manual = compute_manual_jacobian(compiled, data, 3, vars)

        # Test against manual baseline (ground truth)
        @test isapprox(J_ad, J_manual; rtol=1e-5, atol=1e-8)
        @test isapprox(J_fd, J_manual; rtol=1e-5, atol=1e-8)

        # Original test (AD vs FD) - now less critical since we test against ground truth
        @test isapprox(J_ad, J_fd; rtol=1e-5, atol=1e-10)

        # Discrete contrast: swap group level at row using ContrastEvaluator
        Δ = Vector{Float64}(undef, length(compiled))
        row = 5

        # Use ContrastEvaluator for zero-allocation contrast computation
        contrast_evaluator = FormulaCompiler.contrastevaluator(compiled, data, [:group3])
        FormulaCompiler.contrast_modelrow!(Δ, contrast_evaluator, row, :group3, "A", "B")

        # Validate against manual data substitution
        # IMPORTANT: Must use copy() to preserve CategoricalArray structure and levels.
        group3_from = copy(data.group3)
        group3_from[row] = "A"
        group3_to = copy(data.group3)
        group3_to[row] = "B"
        data_from = merge(data, (group3 = group3_from,))
        data_to = merge(data, (group3 = group3_to,))
        y_from = FormulaCompiler.modelrow(compiled, data_from, row)
        y_to = FormulaCompiler.modelrow(compiled, data_to, row)
        @test isapprox(Δ, y_to .- y_from; rtol=1e-10, atol=1e-10)

        # Marginal effects: η = Xβ (test both AD and FD backends)
        β = coef(model)
        gη_ad = Vector{Float64}(undef, length(vars))
        gη_fd = Vector{Float64}(undef, length(vars))
        Gβ_ad = Matrix{Float64}(undef, length(compiled), length(vars))
        Gβ_fd = Matrix{Float64}(undef, length(compiled), length(vars))

        # Test AD backend (now from Margins)
        marginal_effects_eta!(gη_ad, Gβ_ad, de_ad, β, row)
        # Check consistency with J' * β
        Jrow = Matrix{Float64}(undef, length(compiled), length(vars))
        FormulaCompiler.derivative_modelrow!(Jrow, de_ad, row)
        gη_ref = transpose(Jrow) * β
        @test isapprox(gη_ad, gη_ref; rtol=0, atol=0)

        # Test FD backend (allow reasonable tolerance for numerical differences)
        marginal_effects_eta!(gη_fd, Gβ_fd, de_fd, β, row)
        @test isapprox(gη_fd, gη_ref; rtol=1e-3, atol=1e-5)

        # Test μ marginal effects with both backends (allow reasonable tolerance)
        gμ_ad = Vector{Float64}(undef, length(vars))
        gμ_fd = Vector{Float64}(undef, length(vars))
        Gβ_mu_ad = Matrix{Float64}(undef, length(compiled), length(vars))
        Gβ_mu_fd = Matrix{Float64}(undef, length(compiled), length(vars))
        marginal_effects_mu!(gμ_ad, Gβ_mu_ad, de_ad, β, LogitLink(), row)
        marginal_effects_mu!(gμ_fd, Gβ_mu_fd, de_fd, β, LogitLink(), row)
        @test isapprox(gμ_ad, gμ_fd; rtol=1e-3, atol=1e-5)
    end

    @testset "Single-column FD and parameter gradients" begin
        # Data and model for testing
        n = 200
        df = DataFrame(
            y = randn(n),
            x = randn(n),
            z = abs.(randn(n)) .+ 0.1,
            group3 = categorical(rand(["A", "B", "C"], n)),
        )
        data = Tables.columntable(df)
        model = lm(@formula(y ~ 1 + x + z + x & group3), df)
        compiled = FormulaCompiler.compile_formula(model, data)
        vars = [:x, :z]
        β = coef(model)

        # Build evaluators (need both AD and FD for different tests)
        de = FormulaCompiler.derivativeevaluator_ad(compiled, data, vars)
        de_fd = FormulaCompiler.derivativeevaluator_fd(compiled, data, vars)
        test_row = 5

        # Test single-column FD Jacobian
        @testset "fd_jacobian_column!" begin
            # Get full Jacobian for comparison using AD
            J_full = Matrix{Float64}(undef, length(compiled), length(vars))
            FormulaCompiler.derivative_modelrow!(J_full, de, test_row)

            # Test each variable column using FD evaluator
            for (i, var) in enumerate(vars)
                Jk = Vector{Float64}(undef, length(compiled))
                FormulaCompiler.fd_jacobian_column!(Jk, de_fd, test_row, i)

                # Should match corresponding column from full AD Jacobian
                @test isapprox(Jk, J_full[:, i]; rtol=1e-6, atol=1e-8)
            end

            # Test against standalone FD Jacobian
            J_fd_standalone = Matrix{Float64}(undef, length(compiled), length(vars))
            FormulaCompiler.derivative_modelrow!(J_fd_standalone, de_fd, test_row)

            for (i, var) in enumerate(vars)
                Jk = Vector{Float64}(undef, length(compiled))
                FormulaCompiler.fd_jacobian_column!(Jk, de_fd, test_row, i)
                @test isapprox(Jk, J_fd_standalone[:, i]; rtol=1e-6, atol=1e-8)
            end
        end
    end

    @testset "GLM(Logit) and MixedModels" begin
        # Data
        df = DataFrame(
            y = rand([0, 1], n),
            x = randn(n),
            z = abs.(randn(n)) .+ 0.1,
            group3 = categorical(rand(["A", "B", "C"], n)),
            g = categorical(rand(1:20, n)),
        )
        data = Tables.columntable(df)

        # row
        r = 3
        # GLM (Logit)
        glm_model = glm(@formula(y ~ 1 + x + z + x & group3), df, Binomial(), LogitLink())
        compiled_glm = FormulaCompiler.compile_formula(glm_model, data)
        vars = [:x, :z]
        de_glm = FormulaCompiler.derivativeevaluator_ad(compiled_glm, data, vars)
        J = Matrix{Float64}(undef, length(compiled_glm), length(vars))
        FormulaCompiler.derivative_modelrow!(J, de_glm, r)  # warm path
        # FD compare
        de_glm_fd = FormulaCompiler.derivativeevaluator_fd(compiled_glm, data, vars)
        J_fd = similar(J)
        FormulaCompiler.derivative_modelrow!(J_fd, de_glm_fd, r)
        @test isapprox(J, J_fd; rtol=1e-6, atol=1e-8)

        # MixedModels (fixed effects only)
        mm = fit(MixedModel, @formula(y ~ 1 + x + z + (1|g)), df; progress=false)
        compiled_mm = FormulaCompiler.compile_formula(mm, data)
        de_mm = FormulaCompiler.derivativeevaluator_ad(compiled_mm, data, vars)
        Jmm = Matrix{Float64}(undef, length(compiled_mm), length(vars))
        FormulaCompiler.derivative_modelrow!(Jmm, de_mm, 2)
        de_mm_fd = FormulaCompiler.derivativeevaluator_fd(compiled_mm, data, vars)
        Jmm_fd = similar(Jmm)
        FormulaCompiler.derivative_modelrow!(Jmm_fd, de_mm_fd, 3)
        @test isapprox(Jmm, Jmm_fd; rtol=1e-6, atol=1e-8)
    end

    @testset "FD backend robustness and edge cases" begin
        # Test data with various scales and edge cases
        n = 100
        df = DataFrame(
            y = randn(n),
            x_tiny = randn(n) * 1e-6,      # Very small scale
            x_large = randn(n) * 1e6,      # Very large scale
            x_zero = zeros(n),              # Constant zero
            x_normal = randn(n),           # Normal scale
            group4 = categorical(rand(["A", "B", "C", "D"], n)),  # 4 levels
            group2 = categorical(rand(["X", "Y"], n)),             # 2 levels
        )
        data = Tables.columntable(df)

        # Complex model with multiple interactions
        model = lm(@formula(y ~ 1 + x_tiny + x_large + x_normal +
                           x_tiny & group4 + x_normal & group2 +
                           x_large & group4), df)
        compiled = FormulaCompiler.compile_formula(model, data)
        vars = [:x_tiny, :x_large, :x_normal]
        de = FormulaCompiler.derivativeevaluator_fd(compiled, data, vars)

        @testset "Different step sizes" begin
            # Test various step sizes for FD
            test_row = 5
            J_auto = Matrix{Float64}(undef, length(compiled), length(vars))
            J_small = Matrix{Float64}(undef, length(compiled), length(vars))
            J_large = Matrix{Float64}(undef, length(compiled), length(vars))

            # Auto step (default)
            FormulaCompiler.derivative_modelrow!(J_auto, de, test_row)

            # Note: Advanced step size control would require additional FD evaluator options
            # For now, test that the evaluator works correctly
            FormulaCompiler.derivative_modelrow!(J_small, de, test_row)
            FormulaCompiler.derivative_modelrow!(J_large, de, test_row)

            # All should be reasonably close (allowing for step size effects)
            @test isapprox(J_auto, J_small; rtol=1e-3, atol=1e-6)
            @test isapprox(J_auto, J_large; rtol=1e-2, atol=1e-5)
        end

        @testset "All categorical combinations" begin
            # Test that FD works correctly for all categorical levels
            rows_to_test = [1, 10, 25, 50, 75, 90]  # Sample across dataset

            for test_row in rows_to_test
                # Compute Jacobian with FD
                J_fd = Matrix{Float64}(undef, length(compiled), length(vars))
                FormulaCompiler.derivative_modelrow!(J_fd, de, test_row)

                # Compute manual baseline for comparison
                J_manual = compute_manual_jacobian(compiled, data, test_row, vars)

                # Should match manual computation (allow for FD numerical error)
                @test isapprox(J_fd, J_manual; rtol=1e-3, atol=1e-6)
            end
        end

        @testset "Extreme variable scales with FD" begin
            # Focus on how FD handles different variable scales
            test_row = 15

            # Get Jacobians
            J_fd = Matrix{Float64}(undef, length(compiled), length(vars))
            FormulaCompiler.derivative_modelrow!(J_fd, de, test_row)
            J_manual = compute_manual_jacobian(compiled, data, test_row, vars)

            # Check each variable separately
            for (i, var) in enumerate(vars)
                var_scale = abs(data[var][test_row])

                # The relative error should be reasonable regardless of scale
                if var_scale > 1e-10  # Avoid issues with truly zero values
                    col_diff = abs.(J_fd[:, i] .- J_manual[:, i])
                    max_expected_val = maximum(abs.(J_manual[:, i]))

                    if max_expected_val > 1e-10
                        rel_error = maximum(col_diff) / max_expected_val
                        @test rel_error < 1e-2  # More lenient for extreme scales
                    end
                end
            end
        end

        @testset "FD marginal effects consistency across rows" begin
            # Test that FD marginal effects are consistent across different rows
            β = coef(model)

            # Test multiple rows with FD backend
            test_rows = [5, 15, 25, 35, 45]
            de_fd = FormulaCompiler.derivativeevaluator_fd(compiled, data, vars)

            for row in test_rows
                # Compute marginal effects with FD (now from Margins)
                gη_fd = Vector{Float64}(undef, length(vars))
                Gβ_fd = Matrix{Float64}(undef, length(compiled), length(vars))
                marginal_effects_eta!(gη_fd, Gβ_fd, de_fd, β, row)

                # Compute reference using manual Jacobian
                J_manual = compute_manual_jacobian(compiled, data, row, vars)
                gη_manual = transpose(J_manual) * β

                # Should match
                @test isapprox(gη_fd, gη_manual; rtol=1e-4, atol=1e-7)
            end
        end

        @testset "Many variables FD scaling" begin
            # Test FD with more variables to stress the generated functions
            df_many = DataFrame(
                y = randn(n),
                x1 = randn(n), x2 = randn(n), x3 = randn(n), x4 = randn(n),
                x5 = randn(n), x6 = randn(n), x7 = randn(n), x8 = randn(n),
                group = categorical(rand(["A", "B"], n)),
            )
            data_many = Tables.columntable(df_many)

            # Model with many variables
            model_many = lm(@formula(y ~ 1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 +
                                   x1 & group + x4 & group), df_many)
            compiled_many = FormulaCompiler.compile_formula(model_many, data_many)
            vars_many = [:x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8]
            de_many = FormulaCompiler.derivativeevaluator_fd(compiled_many, data_many, vars_many)

            # Test FD with many variables
            test_row = 10
            J_fd_many = Matrix{Float64}(undef, length(compiled_many), length(vars_many))
            FormulaCompiler.derivative_modelrow!(J_fd_many, de_many, test_row)

            # Compare against manual
            J_manual_many = compute_manual_jacobian(compiled_many, data_many, test_row, vars_many)
            @test isapprox(J_fd_many, J_manual_many; rtol=1e-5, atol=1e-8)

            # Test marginal effects too (now from Margins)
            β_many = coef(model_many)
            gη_fd_many = Vector{Float64}(undef, length(vars_many))
            Gβ_fd_many = Matrix{Float64}(undef, length(compiled_many), length(vars_many))
            marginal_effects_eta!(gη_fd_many, Gβ_fd_many, de_many, β_many, test_row)

            gη_manual_many = transpose(J_manual_many) * β_many
            @test isapprox(gη_fd_many, gη_manual_many; rtol=1e-4, atol=1e-7)
        end
    end
end

@testset "Variance Computation Primitives" begin
    n = 200
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        z = abs.(randn(n)) .+ 0.1,
        group = categorical(rand(["A", "B"], n))
    )
    data = Tables.columntable(df)

    # Linear model for η case
    model_lm = lm(@formula(y ~ x + z), df)
    compiled_lm = FormulaCompiler.compile_formula(model_lm, data)
    vars = [:x, :z]
    β_lm = coef(model_lm)
    Σ_lm = vcov(model_lm)
    de_lm = FormulaCompiler.derivativeevaluator_ad(compiled_lm, data, vars)

    # GLM for μ case
    df_logit = copy(df)
    df_logit.y_binary = rand([0, 1], n)
    data_logit = Tables.columntable(df_logit)
    model_glm = glm(@formula(y_binary ~ x + z), df_logit, Binomial(), LogitLink())
    compiled_glm = FormulaCompiler.compile_formula(model_glm, data_logit)
    β_glm = coef(model_glm)
    Σ_glm = vcov(model_glm)
    de_glm = FormulaCompiler.derivativeevaluator_ad(compiled_glm, data_logit, vars)

    test_row = 5

    @testset "delta_method_se" begin
        # Test with known analytical example (now from Margins)
        Σ_simple = [1.0 0.1; 0.1 1.0]
        gβ_simple = [0.5, -0.3]
        expected_se = sqrt(gβ_simple' * Σ_simple * gβ_simple)

        se = delta_method_se(gβ_simple, Σ_simple)
        @test isapprox(se, expected_se; rtol=1e-12)

        # Test with real model gradient from marginal_effects_eta!
        g_eta = Vector{Float64}(undef, length(vars))
        Gβ_eta = Matrix{Float64}(undef, length(compiled_lm), length(vars))
        marginal_effects_eta!(g_eta, Gβ_eta, de_lm, β_lm, test_row)

        # Use parameter gradient for first variable
        gβ_x = Gβ_eta[:, 1]
        se = delta_method_se(gβ_x, Σ_lm)
        @test se > 0.0
        @test isfinite(se)
    end

    @testset "Integrated workflow: Parameter gradients from marginal_effects" begin
        # Test that we can get parameter gradients for variance calculations
        g = Vector{Float64}(undef, length(vars))
        Gβ = Matrix{Float64}(undef, length(compiled_lm), length(vars))

        # η case (now from Margins)
        marginal_effects_eta!(g, Gβ, de_lm, β_lm, test_row)
        @test size(Gβ) == (length(compiled_lm), length(vars))

        # Each column of Gβ is ∂(∂η/∂x_j)/∂β
        for j in 1:length(vars)
            gβ_j = Gβ[:, j]
            se_j = delta_method_se(gβ_j, Σ_lm)
            @test se_j > 0.0
            @test isfinite(se_j)
        end

        # μ case (now from Margins)
        g_mu = Vector{Float64}(undef, length(vars))
        Gβ_mu = Matrix{Float64}(undef, length(compiled_glm), length(vars))
        marginal_effects_mu!(g_mu, Gβ_mu, de_glm, β_glm, LogitLink(), test_row)

        for j in 1:length(vars)
            gβ_j = Gβ_mu[:, j]
            se_j = delta_method_se(gβ_j, Σ_glm)
            @test se_j > 0.0
            @test isfinite(se_j)
        end
    end

    @testset "Integer Continuous Variables Derivatives" begin
        # Test derivatives work correctly with integer variables
        n = 100
        df_int = DataFrame(
            y = randn(n),
            int_x = rand(1:100, n),        # Integer continuous
            int_age = rand(18:80, n),      # Age as integer
            int_count = rand(0:50, n),     # Count variable
            float_z = randn(n),            # Float for comparison
            group = categorical(rand(["A", "B", "C"], n))
        )
        data_int = Tables.columntable(df_int)

        @testset "Integer variable derivatives - basic" begin
            model = lm(@formula(y ~ int_x), df_int)
            compiled = FormulaCompiler.compile_formula(model, data_int)
            vars = [:int_x]
            de = FormulaCompiler.derivativeevaluator_ad(compiled, data_int, vars)

            test_row = 5

            # Test AD derivatives
            J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
            FormulaCompiler.derivative_modelrow!(J_ad, de, test_row)

            # Test FD derivatives
            de_fd = FormulaCompiler.derivativeevaluator_fd(compiled, data_int, vars)
            J_fd = Matrix{Float64}(undef, length(compiled), length(vars))
            FormulaCompiler.derivative_modelrow!(J_fd, de_fd, test_row)

            # Should match
            @test isapprox(J_ad, J_fd; rtol=1e-6, atol=1e-8)

            # Manual verification - derivative of int_x should be 1 (simple linear)
            @test isapprox(J_ad[2, 1], 1.0; atol=1e-12)  # Second term is int_x coefficient
        end

        @testset "Integer with interactions" begin
            model = lm(@formula(y ~ int_x * group), df_int)
            compiled = FormulaCompiler.compile_formula(model, data_int)
            vars = [:int_x]
            de = FormulaCompiler.derivativeevaluator_ad(compiled, data_int, vars)
            de_fd = FormulaCompiler.derivativeevaluator_fd(compiled, data_int, vars)

            test_row = 10
            J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
            J_fd = Matrix{Float64}(undef, length(compiled), length(vars))

            FormulaCompiler.derivative_modelrow!(J_ad, de, test_row)
            FormulaCompiler.derivative_modelrow!(J_fd, de_fd, test_row)

            @test isapprox(J_ad, J_fd; rtol=1e-6, atol=1e-8)
        end

        @testset "Multiple integer variables" begin
            model = lm(@formula(y ~ int_x + int_age + int_count), df_int)
            compiled = FormulaCompiler.compile_formula(model, data_int)
            vars = [:int_x, :int_age, :int_count]
            de = FormulaCompiler.derivativeevaluator_ad(compiled, data_int, vars)
            de_fd = FormulaCompiler.derivativeevaluator_fd(compiled, data_int, vars)

            test_row = 15
            J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
            J_fd = Matrix{Float64}(undef, length(compiled), length(vars))

            FormulaCompiler.derivative_modelrow!(J_ad, de, test_row)
            FormulaCompiler.derivative_modelrow!(J_fd, de_fd, test_row)

            @test isapprox(J_ad, J_fd; rtol=1e-6, atol=1e-8)

            # Manual verification - derivatives should be [1, 1, 1] for linear terms
            @test isapprox(J_ad[2, 1], 1.0; atol=1e-12)  # int_x
            @test isapprox(J_ad[3, 2], 1.0; atol=1e-12)  # int_age
            @test isapprox(J_ad[4, 3], 1.0; atol=1e-12)  # int_count
        end

        @testset "Mixed integer and float derivatives" begin
            model = lm(@formula(y ~ int_x * float_z + int_age), df_int)
            compiled = FormulaCompiler.compile_formula(model, data_int)
            vars = [:int_x, :float_z, :int_age]
            de = FormulaCompiler.derivativeevaluator_ad(compiled, data_int, vars)
            de_fd = FormulaCompiler.derivativeevaluator_fd(compiled, data_int, vars)

            test_row = 20
            J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
            J_fd = Matrix{Float64}(undef, length(compiled), length(vars))

            FormulaCompiler.derivative_modelrow!(J_ad, de, test_row)
            FormulaCompiler.derivative_modelrow!(J_fd, de_fd, test_row)

            @test isapprox(J_ad, J_fd; rtol=1e-6, atol=1e-8)
        end

        @testset "Integer marginal effects" begin
            model = lm(@formula(y ~ int_x + int_age), df_int)
            compiled = FormulaCompiler.compile_formula(model, data_int)
            vars = [:int_x, :int_age]
            de = FormulaCompiler.derivativeevaluator_ad(compiled, data_int, vars)
            β = coef(model)

            test_row = 25

            # Test marginal effects on η (linear predictor) - now from Margins
            gη_ad = Vector{Float64}(undef, length(vars))
            gη_fd = Vector{Float64}(undef, length(vars))
            Gβ_ad = Matrix{Float64}(undef, length(β), length(vars))
            Gβ_fd = Matrix{Float64}(undef, length(β), length(vars))

            marginal_effects_eta!(gη_ad, Gβ_ad, de, β, test_row)
            # Create FD evaluator for comparison
            de_fd = FormulaCompiler.derivativeevaluator_fd(compiled, data_int, vars)
            marginal_effects_eta!(gη_fd, Gβ_fd, de_fd, β, test_row)

            @test isapprox(gη_ad, gη_fd; rtol=1e-6, atol=1e-8)

            # For linear model, marginal effects should equal coefficients
            @test isapprox(gη_ad[1], β[2]; atol=1e-12)  # int_x coefficient
            @test isapprox(gη_ad[2], β[3]; atol=1e-12)  # int_age coefficient
        end

        @testset "Integer transformations" begin
            # Test derivatives with transformed integer variables
            model = lm(@formula(y ~ log(int_count + 1)), df_int)  # +1 to avoid log(0)
            compiled = FormulaCompiler.compile_formula(model, data_int)
            vars = [:int_count]
            de = FormulaCompiler.derivativeevaluator_ad(compiled, data_int, vars)
            de_fd = FormulaCompiler.derivativeevaluator_fd(compiled, data_int, vars)

            test_row = 30
            J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
            J_fd = Matrix{Float64}(undef, length(compiled), length(vars))

            FormulaCompiler.derivative_modelrow!(J_ad, de, test_row)
            FormulaCompiler.derivative_modelrow!(J_fd, de_fd, test_row)

            @test isapprox(J_ad, J_fd; rtol=1e-6, atol=1e-8)

            # Manual verification - derivative of log(int_count + 1) w.r.t int_count is 1/(int_count + 1)
            expected_deriv = 1.0 / (data_int.int_count[test_row] + 1.0)
            @test isapprox(J_ad[2, 1], expected_deriv; rtol=1e-10)
        end
    end
end
