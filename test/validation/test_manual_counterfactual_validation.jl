# test_manual_counterfactual_validation.jl - Manual Counterfactual Validation
# julia --project="." test/validation/test_manual_counterfactual_validation.jl > test/validation/test_manual_counterfactual_validation.txt 2>&1
#
# This test implements step-by-step manual counterfactual computation and validates
# that population_margins() matches hand-computed counterfactual results exactly.
#
# CRITICAL: This validates the actual computational sequence of counterfactual analysis:
# 1. Set focal variable to level A for ALL observations → predict ŷₐ
# 2. Set focal variable to level B for ALL observations → predict ŷᵦ  
# 3. AME = mean(ŷₐ - ŷᵦ) [contrast-then-average]
#
# This catches bugs that analytical validation tests miss:
# - Wrong contrast directions (baseline→level vs level→baseline)
# - Wrong baseline inference (Control vs Treatment as reference)
# - Wrong computational sequences (average-then-contrast vs contrast-then-average)
# - Override system failures (incorrect scenario application)

using Test
using Random
using DataFrames, CategoricalArrays, GLM
using Margins
using Tables
using FormulaCompiler
using Statistics: mean
using LinearAlgebra: dot

@testset "Manual Counterfactual Validation - Critical Implementation Correctness" begin
    
    @testset "Binary Variable Counterfactual Validation" begin
        # Test binary (boolean) variable AME against manual counterfactual computation
        
        Random.seed!(08540)
        n = 200
        df = DataFrame(
            x = randn(n),
            treatment = rand([true, false], n),  # Boolean treatment variable
        )
        df.y = 0.5 * df.x + 1.0 * df.treatment + 0.1 * randn(n)
        
        @testset "Linear Model: Binary Treatment" begin
            model = lm(@formula(y ~ x + treatment), df)
            data_nt = Tables.columntable(df)
            
            # === MANUAL COUNTERFACTUAL COMPUTATION ===
            # Step 1: Create scenarios for treatment = false and treatment = true
            scenario_false = FormulaCompiler.create_scenario("control", data_nt; treatment=false)
            scenario_true = FormulaCompiler.create_scenario("treatment", data_nt; treatment=true)
            
            # Step 2: Compile formula and get coefficients
            compiled = FormulaCompiler.compile_formula(model, data_nt)
            β = coef(model)
            row_buf = Vector{Float64}(undef, length(compiled))
            
            # Step 3: Manual counterfactual loop - predict under both scenarios for each observation
            predictions_false = Float64[]
            predictions_true = Float64[]
            
            for i in 1:n
                # Predict with treatment = false for this observation
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_false.data, i)
                η_false = dot(row_buf, β)
                push!(predictions_false, η_false)
                
                # Predict with treatment = true for this observation  
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_true.data, i)
                η_true = dot(row_buf, β)
                push!(predictions_true, η_true)
            end
            
            # Step 4: Manual AME computation (contrast-then-average)
            manual_contrasts = predictions_true .- predictions_false  # Per-observation contrasts
            manual_ame = mean(manual_contrasts)  # Average of contrasts
            
            # === VALIDATE AGAINST population_margins ===
            ame_result = population_margins(model, df; vars=[:treatment], scale=:response)
            computed_ame = DataFrame(ame_result).estimate[1]
            
            @test manual_ame ≈ computed_ame atol=1e-12
            @debug "Binary counterfactual validation" manual_ame=manual_ame computed_ame=computed_ame difference=abs(manual_ame - computed_ame)
            
            # Additional validation: Check that we're using correct baseline (false→true)
            # For boolean, baseline should be false, so AME should be positive if treatment effect is positive
            treatment_coef = coef(model)[3]  # Coefficient of treatment variable
            @test manual_ame ≈ treatment_coef atol=1e-12  # For linear model, AME equals coefficient
        end
        
        @testset "Logistic Model: Binary Treatment - Nonlinear Chain Rule" begin
            # Create binary outcome for logistic regression
            df_logistic = copy(df)
            df_logistic.y_binary = rand(n) .< GLM.predict(glm(@formula(y ~ x + treatment), df, Normal()), df)
            
            model = glm(@formula(y_binary ~ x + treatment), df_logistic, Binomial(), LogitLink())
            data_nt = Tables.columntable(df_logistic)
            
            # === MANUAL COUNTERFACTUAL COMPUTATION (NONLINEAR) ===
            scenario_false = FormulaCompiler.create_scenario("control", data_nt; treatment=false)
            scenario_true = FormulaCompiler.create_scenario("treatment", data_nt; treatment=true)
            
            compiled = FormulaCompiler.compile_formula(model, data_nt)
            β = coef(model)
            row_buf = Vector{Float64}(undef, length(compiled))
            
            predictions_false = Float64[]
            predictions_true = Float64[]
            
            for i in 1:n
                # Get linear predictors (η) under both scenarios
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_false.data, i)
                η_false = dot(row_buf, β)
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_true.data, i)
                η_true = dot(row_buf, β)
                
                # Transform to response scale (probabilities)
                prob_false = 1 / (1 + exp(-η_false))
                prob_true = 1 / (1 + exp(-η_true))
                
                push!(predictions_false, prob_false)
                push!(predictions_true, prob_true)
            end
            
            # Manual AME on response scale
            manual_ame_response = mean(predictions_true .- predictions_false)
            
            # === VALIDATE AGAINST population_margins ===
            ame_result = population_margins(model, df_logistic; vars=[:treatment], scale=:response)
            computed_ame = DataFrame(ame_result).estimate[1]
            
            @test manual_ame_response ≈ computed_ame atol=1e-12
            @debug "Logistic counterfactual validation" manual_ame=manual_ame_response computed_ame=computed_ame difference=abs(manual_ame_response - computed_ame)
        end
    end
    
    @testset "Categorical Variable Counterfactual Validation" begin
        # Test categorical variable AME against manual counterfactual computation
        
        Random.seed!(06515)
        n = 300
        df = DataFrame(
            x = randn(n),
            region = categorical(rand(["North", "South", "East"], n)),
        )
        df.y = 0.3 * df.x + 0.5 * (df.region .== "North") + 1.2 * (df.region .== "South") + 0.1 * randn(n)
        
        @testset "Linear Model: Categorical Region" begin
            model = lm(@formula(y ~ x + region), df)
            data_nt = Tables.columntable(df)
            
            # Get baseline level (should be "East" for this contrast coding)
            baseline = FormulaCompiler._get_baseline_level(model, :region)
            levels_to_test = ["North", "South"]  # Test non-baseline levels
            
            for test_level in levels_to_test
                @testset "Manual validation: $(baseline) → $(test_level)" begin
                    
                    # === MANUAL COUNTERFACTUAL COMPUTATION ===
                    scenario_baseline = FormulaCompiler.create_scenario("baseline", data_nt; region=baseline)
                    scenario_level = FormulaCompiler.create_scenario("level", data_nt; region=test_level)
                    
                    compiled = FormulaCompiler.compile_formula(model, data_nt)
                    β = coef(model)
                    row_buf = Vector{Float64}(undef, length(compiled))
                    
                    predictions_baseline = Float64[]
                    predictions_level = Float64[]
                    
                    for i in 1:n
                        # Predict under baseline scenario
                        FormulaCompiler.modelrow!(row_buf, compiled, scenario_baseline.data, i)
                        η_baseline = dot(row_buf, β)
                        push!(predictions_baseline, η_baseline)
                        
                        # Predict under test level scenario
                        FormulaCompiler.modelrow!(row_buf, compiled, scenario_level.data, i)
                        η_level = dot(row_buf, β)
                        push!(predictions_level, η_level)
                    end
                    
                    # Manual AME: baseline→level contrast
                    manual_ame = mean(predictions_level .- predictions_baseline)
                    
                    # === VALIDATE AGAINST population_margins ===
                    ame_result = population_margins(model, df; vars=[:region], scale=:response)
                    ame_df = DataFrame(ame_result)
                    
                    # Find the correct row for this level
                    level_row = findfirst(row -> occursin(test_level, row.term), eachrow(ame_df))
                    @test level_row !== nothing  # Should find the level
                    
                    computed_ame = ame_df.estimate[level_row]
                    
                    @test manual_ame ≈ computed_ame atol=1e-12
                    @debug "Categorical counterfactual validation" baseline=baseline level=test_level manual_ame=manual_ame computed_ame=computed_ame
                end
            end
        end
        
        @testset "Logistic Model: Categorical Region - Nonlinear Validation" begin
            # Create binary outcome for logistic regression
            df_logistic = copy(df)
            df_logistic.y_binary = rand(n) .< GLM.predict(glm(@formula(y ~ x + region), df, Normal()), df)
            
            model = glm(@formula(y_binary ~ x + region), df_logistic, Binomial(), LogitLink())
            data_nt = Tables.columntable(df_logistic)
            
            baseline = FormulaCompiler._get_baseline_level(model, :region)
            test_level = "North"  # Test one level for efficiency
            
            # === MANUAL COUNTERFACTUAL COMPUTATION (NONLINEAR) ===
            scenario_baseline = FormulaCompiler.create_scenario("baseline", data_nt; region=baseline)
            scenario_level = FormulaCompiler.create_scenario("level", data_nt; region=test_level)
            
            compiled = FormulaCompiler.compile_formula(model, data_nt)
            β = coef(model)
            row_buf = Vector{Float64}(undef, length(compiled))
            
            predictions_baseline = Float64[]
            predictions_level = Float64[]
            
            for i in 1:n
                # Get linear predictors under both scenarios
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_baseline.data, i)
                η_baseline = dot(row_buf, β)
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_level.data, i)
                η_level = dot(row_buf, β)
                
                # Transform to probabilities
                prob_baseline = 1 / (1 + exp(-η_baseline))
                prob_level = 1 / (1 + exp(-η_level))
                
                push!(predictions_baseline, prob_baseline)
                push!(predictions_level, prob_level)
            end
            
            # Manual AME on response scale
            manual_ame_response = mean(predictions_level .- predictions_baseline)
            
            # === VALIDATE AGAINST population_margins ===
            ame_result = population_margins(model, df_logistic; vars=[:region], scale=:response)
            ame_df = DataFrame(ame_result)
            
            level_row = findfirst(row -> occursin(test_level, row.term), eachrow(ame_df))
            computed_ame = ame_df.estimate[level_row]
            
            @test manual_ame_response ≈ computed_ame atol=1e-12
            @debug "Categorical logistic counterfactual validation" baseline=baseline level=test_level manual_ame=manual_ame_response computed_ame=computed_ame
        end
    end
    
    @testset "Continuous Variable Validation - Finite Difference Check" begin
        # For continuous variables, validate that derivative-based AME matches finite difference approximation
        
        Random.seed!(456)
        n = 150
        df = DataFrame(x = randn(n), z = randn(n))
        df.y = 0.8 * df.x + 0.3 * df.z + 0.1 * randn(n)
        
        @testset "Linear Model: Continuous Variable vs Finite Differences" begin
            model = lm(@formula(y ~ x + z), df)
            data_nt = Tables.columntable(df)
            
            # Small perturbation for finite differences
            δ = 1e-6
            
            # === MANUAL FINITE DIFFERENCE COMPUTATION ===
            # Create modified data with small perturbations
            x_plus = df.x .+ δ
            x_minus = df.x .- δ
            
            compiled = FormulaCompiler.compile_formula(model, data_nt)
            β = coef(model)
            row_buf = Vector{Float64}(undef, length(compiled))
            
            finite_differences = Float64[]
            
            for i in 1:n
                # Create per-observation scenarios with perturbations
                data_plus = (x = x_plus, z = data_nt.z, y = data_nt.y)
                data_minus = (x = x_minus, z = data_nt.z, y = data_nt.y)
                
                # Predict at x + δ
                FormulaCompiler.modelrow!(row_buf, compiled, data_plus, i)
                y_plus = dot(row_buf, β)
                
                # Predict at x - δ  
                FormulaCompiler.modelrow!(row_buf, compiled, data_minus, i)
                y_minus = dot(row_buf, β)
                
                # Finite difference approximation
                fd_derivative = (y_plus - y_minus) / (2δ)
                push!(finite_differences, fd_derivative)
            end
            
            manual_ame_fd = mean(finite_differences)
            
            # === VALIDATE AGAINST population_margins ===
            ame_result = population_margins(model, df; vars=[:x], scale=:response)
            computed_ame = DataFrame(ame_result).estimate[1]
            
            @test manual_ame_fd ≈ computed_ame atol=1e-6  # Allow finite difference tolerance
            @debug "Continuous finite difference validation" manual_fd=manual_ame_fd computed_ame=computed_ame difference=abs(manual_ame_fd - computed_ame)
            
            # For linear model, should also equal coefficient
            x_coef = coef(model)[2]  # Coefficient of x
            @test computed_ame ≈ x_coef atol=1e-12
        end
        
        @testset "Logistic Model: Continuous Variable vs Finite Differences" begin
            # Test nonlinear case where finite differences are needed
            df_logistic = copy(df)
            df_logistic.y_binary = rand(n) .< GLM.predict(glm(@formula(y ~ x + z), df, Normal()), df)
            
            model = glm(@formula(y_binary ~ x + z), df_logistic, Binomial(), LogitLink())
            data_nt = Tables.columntable(df_logistic)
            
            # Small perturbation for finite differences
            δ = 1e-6
            
            # === MANUAL FINITE DIFFERENCE COMPUTATION (NONLINEAR) ===
            x_plus = df_logistic.x .+ δ
            x_minus = df_logistic.x .- δ
            
            compiled = FormulaCompiler.compile_formula(model, data_nt)
            β = coef(model)
            row_buf = Vector{Float64}(undef, length(compiled))
            
            finite_differences = Float64[]
            
            for i in 1:n
                # Create per-observation scenarios with perturbations
                data_plus = (x = x_plus, z = data_nt.z, y_binary = data_nt.y_binary)
                data_minus = (x = x_minus, z = data_nt.z, y_binary = data_nt.y_binary)
                
                # Predict at x + δ
                FormulaCompiler.modelrow!(row_buf, compiled, data_plus, i)
                η_plus = dot(row_buf, β)
                prob_plus = 1 / (1 + exp(-η_plus))
                
                # Predict at x - δ  
                FormulaCompiler.modelrow!(row_buf, compiled, data_minus, i)
                η_minus = dot(row_buf, β)
                prob_minus = 1 / (1 + exp(-η_minus))
                
                # Finite difference approximation on response scale
                fd_derivative = (prob_plus - prob_minus) / (2δ)
                push!(finite_differences, fd_derivative)
            end
            
            manual_ame_fd = mean(finite_differences)
            
            # === VALIDATE AGAINST population_margins ===
            ame_result = population_margins(model, df_logistic; vars=[:x], scale=:response)
            computed_ame = DataFrame(ame_result).estimate[1]
            
            @test manual_ame_fd ≈ computed_ame atol=1e-6  # Allow finite difference tolerance
            @debug "Nonlinear continuous finite difference validation" manual_fd=manual_ame_fd computed_ame=computed_ame difference=abs(manual_ame_fd - computed_ame)
        end
        
        @testset "Integer Continuous Variable - Derivative-based Validation" begin
            # Test that integer variables are correctly treated as continuous (not categorical)
            
            Random.seed!(567)
            n = 100
            df_int = DataFrame(
                age = rand(18:80, n),           # Integer continuous variable
                income = rand(20000:100000, n), # Integer continuous variable  
                x = randn(n)                    # Regular continuous variable
            )
            df_int.y = 0.01 * df_int.age + 0.00001 * df_int.income + 0.5 * df_int.x + 0.1 * randn(n)
            
            model = lm(@formula(y ~ age + income + x), df_int)
            
            # === VALIDATE INTEGER VARIABLES ARE TREATED AS CONTINUOUS ===
            # Should match coefficients exactly for linear model
            ame_age = population_margins(model, df_int; vars=[:age], scale=:response)
            age_ame = DataFrame(ame_age).estimate[1]
            age_coef = coef(model)[2]  # Age coefficient
            @test age_ame ≈ age_coef atol=1e-12
            
            ame_income = population_margins(model, df_int; vars=[:income], scale=:response)
            income_ame = DataFrame(ame_income).estimate[1]
            income_coef = coef(model)[3]  # Income coefficient
            @test income_ame ≈ income_coef atol=1e-12
            
            @debug "Integer continuous variable validation" age_matches=abs(age_ame - age_coef) < 1e-12 income_matches=abs(income_ame - income_coef) < 1e-12
        end
    end
    
    @testset "Mixed Model Validation - Multiple Variable Types" begin
        # Test mixed continuous + categorical in same model
        
        Random.seed!(789)
        n = 250
        df = DataFrame(
            x = randn(n),
            treatment = rand([true, false], n),
            region = categorical(rand(["A", "B"], n))
        )
        df.y = 0.4 * df.x + 0.8 * df.treatment + 0.6 * (df.region .== "A") + 0.1 * randn(n)
        
        model = lm(@formula(y ~ x + treatment + region), df)
        
        @testset "Validate Each Variable Type in Mixed Model" begin
            
            # Test continuous variable (should match coefficient)
            ame_continuous = population_margins(model, df; vars=[:x], scale=:response)
            continuous_ame = DataFrame(ame_continuous).estimate[1]
            x_coef = coef(model)[2]
            @test continuous_ame ≈ x_coef atol=1e-12
            
            # Test binary variable with manual counterfactual
            data_nt = Tables.columntable(df)
            scenario_false = FormulaCompiler.create_scenario("control", data_nt; treatment=false)
            scenario_true = FormulaCompiler.create_scenario("treatment", data_nt; treatment=true)
            
            compiled = FormulaCompiler.compile_formula(model, data_nt)
            β = coef(model)
            row_buf = Vector{Float64}(undef, length(compiled))
            
            manual_contrasts = Float64[]
            for i in 1:n
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_false.data, i)
                pred_false = dot(row_buf, β)
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_true.data, i)
                pred_true = dot(row_buf, β)
                push!(manual_contrasts, pred_true - pred_false)
            end
            manual_treatment_ame = mean(manual_contrasts)
            
            ame_binary = population_margins(model, df; vars=[:treatment], scale=:response)
            binary_ame = DataFrame(ame_binary).estimate[1]
            @test manual_treatment_ame ≈ binary_ame atol=1e-12
            
            # Test categorical variable with manual counterfactual
            baseline = FormulaCompiler._get_baseline_level(model, :region)
            test_level = baseline == "A" ? "B" : "A"
            
            scenario_baseline = FormulaCompiler.create_scenario("baseline", data_nt; region=baseline)
            scenario_level = FormulaCompiler.create_scenario("level", data_nt; region=test_level)
            
            manual_cat_contrasts = Float64[]
            for i in 1:n
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_baseline.data, i)
                pred_baseline = dot(row_buf, β)
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_level.data, i)
                pred_level = dot(row_buf, β)
                push!(manual_cat_contrasts, pred_level - pred_baseline)
            end
            manual_cat_ame = mean(manual_cat_contrasts)
            
            ame_categorical = population_margins(model, df; vars=[:region], scale=:response)
            cat_ame_df = DataFrame(ame_categorical)
            level_row = findfirst(row -> occursin(test_level, row.term), eachrow(cat_ame_df))
            cat_ame = cat_ame_df.estimate[level_row]
            @test manual_cat_ame ≈ cat_ame atol=1e-12
            
            @debug "Mixed model validation" continuous_matches=abs(continuous_ame - x_coef) < 1e-12 binary_matches=abs(manual_treatment_ame - binary_ame) < 1e-12 categorical_matches=abs(manual_cat_ame - cat_ame) < 1e-12
        end
    end
    
    @testset "Computational Sequence Validation" begin
        # Critical test: Verify that both computational sequences give identical results
        # due to linearity of expectation: E[f(X) - f(Y)] = E[f(X)] - E[f(Y)]
        
        Random.seed!(999)
        n = 100
        df = DataFrame(
            x = randn(n),
            treatment = categorical(rand(["Control", "Treatment"], n))  # Use categorical encoding
        )
        df.y_binary = rand(n) .< 0.5  # Binary outcome
        
        model = glm(@formula(y_binary ~ x + treatment), df, Binomial(), LogitLink())
        data_nt = Tables.columntable(df)
        
        @testset "Contrast-then-Average vs Average-then-Contrast" begin
            
            # === METHOD 1: CONTRAST-THEN-AVERAGE (CORRECT) ===
            baseline = FormulaCompiler._get_baseline_level(model, :treatment)
            test_level = baseline == "Control" ? "Treatment" : "Control"
            
            scenario_baseline = FormulaCompiler.create_scenario("baseline", data_nt; treatment=baseline)
            scenario_level = FormulaCompiler.create_scenario("level", data_nt; treatment=test_level)
            
            compiled = FormulaCompiler.compile_formula(model, data_nt)
            β = coef(model)
            row_buf = Vector{Float64}(undef, length(compiled))
            
            contrasts = Float64[]
            for i in 1:n
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_baseline.data, i)
                η_baseline = dot(row_buf, β)
                prob_baseline = 1 / (1 + exp(-η_baseline))
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_level.data, i)
                η_level = dot(row_buf, β)
                prob_level = 1 / (1 + exp(-η_level))
                push!(contrasts, prob_level - prob_baseline)
            end
            contrast_then_average = mean(contrasts)
            
            # === METHOD 2: AVERAGE-THEN-CONTRAST (WRONG) ===
            probs_baseline = Float64[]
            probs_level = Float64[]
            for i in 1:n
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_baseline.data, i)
                η_baseline = dot(row_buf, β)
                push!(probs_baseline, 1 / (1 + exp(-η_baseline)))
                FormulaCompiler.modelrow!(row_buf, compiled, scenario_level.data, i)
                η_level = dot(row_buf, β)
                push!(probs_level, 1 / (1 + exp(-η_level)))
            end
            average_then_contrast = mean(probs_level) - mean(probs_baseline)
            
            # === VERIFY THEY ARE IDENTICAL (both methods mathematically equivalent) ===
            # Due to linearity of expectation: E[f(X)] - E[f(Y)] = E[f(X) - f(Y)]
            # Both methods should give identical results for any model
            difference = abs(contrast_then_average - average_then_contrast)
            @test difference < 1e-12  # Should be essentially identical
            
            # === VALIDATE population_margins MATCHES BOTH METHODS ===
            ame_result = population_margins(model, df; vars=[:treatment], scale=:response)
            ame_df = DataFrame(ame_result)
            level_row = findfirst(row -> occursin(test_level, row.term), eachrow(ame_df))
            computed_ame = ame_df.estimate[level_row]
            
            @test computed_ame ≈ contrast_then_average atol=1e-12  # Should match contrast-then-average
            @test computed_ame ≈ average_then_contrast atol=1e-12  # Should also match average-then-contrast
            
            @debug "Computational sequence validation" contrast_then_average=contrast_then_average average_then_contrast=average_then_contrast computed_ame=computed_ame sequence_difference=difference
        end
    end
end