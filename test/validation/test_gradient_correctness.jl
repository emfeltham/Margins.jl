# test_gradient_correctness.jl - Mathematical Correctness Tests for Parameter Gradients
# Migrated from FormulaCompiler.jl (2025-10-09)

using Test
using Margins
using FormulaCompiler
using DataFrames, Tables, GLM, CategoricalArrays
using LinearAlgebra

@testset "Parameter Gradient Mathematical Correctness" begin

    # Test setup with known analytical solutions
    n = 100
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        z = randn(n),
        group = categorical(rand(["A", "B"], n))
    )
    data = Tables.columntable(df)

    # Simple linear model for analytical verification
    model = lm(@formula(y ~ x + z), df)
    compiled = FormulaCompiler.compile_formula(model, data)
    vars = [:x, :z]
    β = coef(model)

    # Build both evaluators
    de_ad = FormulaCompiler.derivativeevaluator(:ad, compiled, data, vars)
    de_fd = FormulaCompiler.derivativeevaluator(:fd, compiled, data, vars)

    # Buffers
    g_ad = Vector{Float64}(undef, length(vars))
    g_fd = Vector{Float64}(undef, length(vars))
    Gβ_ad = Matrix{Float64}(undef, length(compiled), length(vars))
    Gβ_fd = Matrix{Float64}(undef, length(compiled), length(vars))

    # === Test 1: η-scale Parameter Gradients ===

    # For linear model y ~ x + z, η = β₀ + β₁x + β₂z
    # ∂η/∂x = β₁, ∂η/∂z = β₂
    # ∂(∂η/∂x)/∂β = [0, 1, 0]ᵀ (for x)
    # ∂(∂η/∂z)/∂β = [0, 0, 1]ᵀ (for z)

    test_row = 42
    Margins.marginal_effects_eta!(g_ad, Gβ_ad, de_ad, β, test_row)
    Margins.marginal_effects_eta!(g_fd, Gβ_fd, de_fd, β, test_row)

    # Verify marginal effects consistency (relaxed tolerance for FD vs AD)
    @test g_ad ≈ g_fd rtol=1e-6 atol=1e-8

    # Verify parameter gradients consistency (relaxed tolerance for FD vs AD)
    @test Gβ_ad ≈ Gβ_fd rtol=1e-6 atol=1e-8

    # Verify analytical correctness for linear model
    # For y ~ x + z, the marginal effects should be constant = [β₁, β₂]
    expected_marginals = β[2:end]  # Skip intercept
    @test g_ad ≈ expected_marginals rtol=1e-6
    @test g_fd ≈ expected_marginals rtol=1e-6

    # For linear model, parameter gradients should be design matrix columns
    # ∂(∂η/∂x)/∂β should be the model matrix column for x
    # ∂(∂η/∂z)/∂β should be the model matrix column for z

    # Debug dimensions
    @debug "  compiled length: $(length(compiled))"
    @debug "  Gβ_ad size: $(size(Gβ_ad))"
    @debug "  vars length: $(length(vars))"

    # Get model matrix for verification
    model_matrix = modelmatrix(model)
    @debug "  model_matrix size: $(size(model_matrix))"

    # The Gβ matrix should be (n_params × n_vars), not (n_rows × n_vars)
    # For y ~ x + z, we have 3 parameters: intercept, x, z
    n_params = size(model_matrix, 2)
    expected_Gβ = zeros(n_params, length(vars))

    # x is in column 2 of model matrix (after intercept)
    # z is in column 3 of model matrix
    x_col_idx = 2  # Column for x in model matrix
    z_col_idx = 3  # Column for z in model matrix

    # For linear model, parameter gradients are the standard basis vectors
    # ∂(∂η/∂x)/∂β = [0, 1, 0]ᵀ (x coefficient is second parameter)
    # ∂(∂η/∂z)/∂β = [0, 0, 1]ᵀ (z coefficient is third parameter)
    expected_Gβ[x_col_idx, 1] = 1.0  # ∂(∂η/∂x)/∂βₓ = 1
    expected_Gβ[z_col_idx, 2] = 1.0  # ∂(∂η/∂z)/∂βz = 1

    @test Gβ_ad ≈ expected_Gβ rtol=1e-6
    @test Gβ_fd ≈ expected_Gβ rtol=1e-6

    # === Test 2: μ-scale Parameter Gradients with Identity Link ===

    # For Identity link, g(η) = η, so g'(η) = 1, g''(η) = 0
    # Therefore μ-scale should match η-scale exactly

    Margins.marginal_effects_mu!(g_ad, Gβ_ad, de_ad, β, IdentityLink(), test_row)
    Margins.marginal_effects_mu!(g_fd, Gβ_fd, de_fd, β, IdentityLink(), test_row)

    # Should be identical to η-scale for Identity link
    g_eta_ad = Vector{Float64}(undef, length(vars))
    g_eta_fd = Vector{Float64}(undef, length(vars))
    Gβ_eta_ad = Matrix{Float64}(undef, length(compiled), length(vars))
    Gβ_eta_fd = Matrix{Float64}(undef, length(compiled), length(vars))

    Margins.marginal_effects_eta!(g_eta_ad, Gβ_eta_ad, de_ad, β, test_row)
    Margins.marginal_effects_eta!(g_eta_fd, Gβ_eta_fd, de_fd, β, test_row)

    @test g_ad ≈ g_eta_ad rtol=1e-10
    @test g_fd ≈ g_eta_fd rtol=1e-10
    @test Gβ_ad ≈ Gβ_eta_ad rtol=1e-10
    @test Gβ_fd ≈ Gβ_eta_fd rtol=1e-10

    # === Test 3: μ-scale Parameter Gradients with Nonlinear Links ===

    # Test multiple nonlinear links
    for (link_name, link) in [("Log", LogLink()), ("Logit", LogitLink()), ("Probit", ProbitLink())]

        @debug "  Testing $(link_name) link..."

        # Test consistency between AD and FD
        Margins.marginal_effects_mu!(g_ad, Gβ_ad, de_ad, β, link, test_row)
        Margins.marginal_effects_mu!(g_fd, Gβ_fd, de_fd, β, link, test_row)

        @test g_ad ≈ g_fd rtol=1e-6 atol=1e-8
        @test Gβ_ad ≈ Gβ_fd rtol=1e-6 atol=1e-8

        # Verify chain rule implementation by numerical differentiation
        # Check that our chain rule matches finite differences of the full function

        # Small perturbation for numerical verification
        ε = 1e-8
        g_plus = Vector{Float64}(undef, length(vars))
        g_minus = Vector{Float64}(undef, length(vars))
        Gβ_plus = Matrix{Float64}(undef, length(compiled), length(vars))
        Gβ_minus = Matrix{Float64}(undef, length(compiled), length(vars))
        Gβ_numerical = Matrix{Float64}(undef, length(compiled), length(vars))

        for i in 1:length(β)
            β_plus = copy(β)
            β_minus = copy(β)
            β_plus[i] += ε
            β_minus[i] -= ε

            Margins.marginal_effects_mu!(g_plus, Gβ_plus, de_fd, β_plus, link, test_row)
            Margins.marginal_effects_mu!(g_minus, Gβ_minus, de_fd, β_minus, link, test_row)

            # Numerical derivative: ∂g/∂β[i] ≈ (g(β+ε) - g(β-ε))/(2ε)
            for j in 1:length(vars)
                Gβ_numerical[i, j] = (g_plus[j] - g_minus[j]) / (2ε)
            end
        end

        # Our analytical chain rule should match numerical differentiation
        @test Gβ_fd ≈ Gβ_numerical rtol=1e-4 atol=1e-6
    end

    # === Test 4: Batch Consistency ===

    @debug "Testing batch consistency..."

    # Test multiple rows to ensure consistency across the dataset
    test_rows = [1, 25, 50, 75, 99]

    for test_row in test_rows
        # η-scale
        Margins.marginal_effects_eta!(g_ad, Gβ_ad, de_ad, β, test_row)
        Margins.marginal_effects_eta!(g_fd, Gβ_fd, de_fd, β, test_row)

        @test g_ad ≈ g_fd rtol=1e-6 atol=1e-8
        @test Gβ_ad ≈ Gβ_fd rtol=1e-6 atol=1e-8

        # μ-scale (Logit)
        Margins.marginal_effects_mu!(g_ad, Gβ_ad, de_ad, β, LogitLink(), test_row)
        Margins.marginal_effects_mu!(g_fd, Gβ_fd, de_fd, β, LogitLink(), test_row)

        @test g_ad ≈ g_fd rtol=1e-6 atol=1e-8
        @test Gβ_ad ≈ Gβ_fd rtol=1e-6 atol=1e-8
    end

    # === Test 5: Delta Method Integration ===

    @debug "Testing delta method integration..."

    # Test that parameter gradients work correctly with delta_method_se

    # Create a test covariance matrix
    p = length(β)
    Σ = Matrix{Float64}(I, p, p) * 0.01  # Simple diagonal covariance

    # Get parameter gradient for first variable at test row
    Margins.marginal_effects_eta!(g_fd, Gβ_fd, de_fd, β, test_row)

    # Extract parameter gradient for first variable
    gβ_x = Gβ_fd[:, 1]  # Parameter gradient for variable x

    # Compute standard error using delta method
    se_x = Margins.delta_method_se(gβ_x, Σ)

    # Should be positive and finite
    @test se_x > 0
    @test isfinite(se_x)

    # Should be deterministic (same result each time)
    se_x2 = Margins.delta_method_se(gβ_x, Σ)
    @test se_x ≈ se_x2

    # === Test 6: Dimensional Consistency ===

    @debug "Testing dimensional consistency..."

    # Verify dimensions are correct
    @test size(Gβ_ad) == (length(compiled), length(vars))
    @test size(Gβ_fd) == (length(compiled), length(vars))
    @test length(g_ad) == length(vars)
    @test length(g_fd) == length(vars)

    # Verify that each column of Gβ corresponds to the correct variable
    # by checking the gradient structure matches the model matrix

    # For our simple model y ~ x + z, columns should match model matrix structure
    @test size(Gβ_fd, 1) == size(modelmatrix(model), 2)  # Same number of parameters
    @test size(Gβ_fd, 2) == length(vars)  # One column per variable

    @debug "All mathematical correctness tests passed!"
    @debug "  - η-scale gradients: analytically verified for linear model"
    @debug "  - μ-scale gradients: chain rule verified vs numerical differentiation"
    @debug "  - Identity link: confirmed to match η-scale exactly"
    @debug "  - Nonlinear links: AD vs FD cross-validated"
    @debug "  - Batch consistency: verified across multiple rows"
    @debug "  - Delta method integration: confirmed working"
    @debug "  - Dimensional consistency: all matrix sizes correct"
end
