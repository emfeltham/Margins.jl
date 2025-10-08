# categorical_bootstrap_tests.jl - Categorical Effects Bootstrap Validation
#
# Categorical Effects Bootstrap Testing
#
# This file implements bootstrap validation specifically for categorical variables,
# which use discrete changes rather than derivatives and have different SE computation
# patterns than continuous variables.

using Test
using Random
using DataFrames
using Statistics
using CategoricalArrays
using GLM
using StatsModels
using Margins
using Printf

# Load bootstrap utilities
# Bootstrap SE validation loaded centrally in runtests.jl

"""
    make_categorical_test_data(; n=500, include_continuous=true, seed=42)

Generate test data with categorical variables for bootstrap validation.

# Arguments
- `n`: Sample size
- `include_continuous`: Whether to include continuous variables for mixed models
- `seed`: Random seed for reproducibility

# Returns
- `DataFrame` with categorical variables and appropriate outcome variable
"""
function make_categorical_test_data(; n=500, include_continuous=true, seed=42)
    Random.seed!(seed)
    
    df = DataFrame(
        # Categorical variables
        education = categorical(rand(["High School", "College", "Graduate"], n)),
        region = categorical(rand(["North", "South", "East", "West"], n)),
        treatment = categorical(rand(["Control", "Treatment"], n)),
        
        # Boolean categorical (common in econometrics)
        union_member = rand([true, false], n)
    )
    
    if include_continuous
        # Add some continuous variables
        df.experience = randn(n) .+ 5.0
        df.age = rand(25:65, n)
    end
    
    # Create realistic outcome variables
    # For continuous outcome (linear models)
    education_effect = ifelse.(df.education .== "Graduate", 2.0,
                      ifelse.(df.education .== "College", 1.0, 0.0))
    region_effect = ifelse.(df.region .== "North", 0.5,
                   ifelse.(df.region .== "South", -0.3, 0.0))
    treatment_effect = ifelse.(df.treatment .== "Treatment", 1.2, 0.0)
    union_effect = ifelse.(df.union_member, 0.8, 0.0)
    
    # Continuous outcome
    df.wage = 10.0 .+ education_effect .+ region_effect .+ treatment_effect .+ union_effect .+ 0.5 .* randn(n)
    
    if include_continuous
        df.wage .+= 0.1 .* df.experience .+ 0.02 .* (df.age .- 45)
    end
    
    # Binary outcome for logistic models
    linear_pred = -0.5 .+ 0.5 .* education_effect .+ 0.3 .* region_effect .+ 
                  0.4 .* treatment_effect .+ 0.6 .* union_effect
    
    if include_continuous
        linear_pred .+= 0.1 .* (df.experience .- 5) .+ 0.02 .* (df.age .- 45)
    end
    
    probs = 1 ./ (1 .+ exp.(-linear_pred))
    df.high_earner = [rand() < p for p in probs]
    
    return df
end

"""
    bootstrap_validate_categorical_effects(model_func, formula, data; categorical_vars=nothing, n_bootstrap=200)

Bootstrap validation for categorical variable effects.

# Arguments
- `model_func`: Model fitting function
- `formula`: Model formula containing categorical variables  
- `data`: Dataset with categorical variables
- `categorical_vars`: Categorical variables to test (auto-detected if nothing)
- `n_bootstrap`: Number of bootstrap samples

# Returns
- `NamedTuple` with bootstrap validation results for categorical effects
"""
function bootstrap_validate_categorical_effects(model_func, formula, data; categorical_vars=nothing, n_bootstrap=200)
    # Auto-detect categorical variables if not specified
    if isnothing(categorical_vars)
        categorical_vars = Symbol[]
        for col_name in names(data)
            col = data[!, col_name]
            if isa(col, CategoricalVector) || isa(col, AbstractVector{Bool})
                push!(categorical_vars, Symbol(col_name))
            end
        end
    end
    
    if isempty(categorical_vars)
        return (success=false, reason="No categorical variables found in data")
    end
    
    # Fit original model
    original_model = model_func(formula, data)
    
    # Get population margins for categorical variables
    # Categorical variable processing is performed automatically by the Margins.jl framework
    try
        original_result = population_margins(original_model, data; type=:effects, backend=:fd)
        original_df = DataFrame(original_result)
        
        # Filter for categorical variable terms
        categorical_terms = String[]
        categorical_ses = Float64[]
        
        for (i, term) in enumerate(original_df.variable)
            # Check if this term corresponds to a categorical variable
            for cat_var in categorical_vars
                if startswith(string(term), string(cat_var)) || string(term) == string(cat_var)
                    push!(categorical_terms, term)
                    push!(categorical_ses, original_df.se[i])
                    break
                end
            end
        end
        
        if isempty(categorical_terms)
            return (success=false, reason="No categorical effects found in margins result")
        end
        
        # Bootstrap computation
        boot_means, boot_ses, n_successful = bootstrap_margins_computation(
            model_func, formula, data, population_margins;
            n_bootstrap=n_bootstrap, type=:effects
        )
        
        # Match bootstrap results to categorical terms
        bootstrap_categorical_ses = Float64[]
        for term in categorical_terms
            term_idx = findfirst(t -> t == term, original_df.variable)
            if !isnothing(term_idx) && term_idx <= length(boot_ses)
                push!(bootstrap_categorical_ses, boot_ses[term_idx])
            end
        end
        
        # Validate agreement
        validation = validate_bootstrap_se_agreement(
            categorical_ses, bootstrap_categorical_ses; 
            tolerance=0.20,  # Higher tolerance for categorical effects
            var_names=categorical_terms
        )
        
        return (
            success = true,
            computed_ses = categorical_ses,
            bootstrap_ses = bootstrap_categorical_ses,
            validation = validation,
            categorical_terms = categorical_terms,
            n_bootstrap_successful = n_successful,
            categorical_vars_tested = categorical_vars
        )
        
    catch e
        return (success=false, error=e)
    end
end

"""
    run_categorical_bootstrap_test_suite(; n_bootstrap=150, verbose=true)

Comprehensive categorical effects bootstrap testing across different model types.
"""
function run_categorical_bootstrap_test_suite(; n_bootstrap=150, verbose=true)
    if verbose
        @debug "Starting Categorical Effects Bootstrap Validation Suite"
        @debug "Bootstrap samples per model: $n_bootstrap"
        @debug "="^60
    end
    
    # Define categorical test models
    categorical_test_models = [
        (
            name = "Linear: Categorical Only",
            model_func = lm,
            data_func = () -> make_categorical_test_data(n=600, include_continuous=false),
            formula = @formula(wage ~ education + region + treatment + union_member),
            expected_agreement = 0.85  # Good for linear categorical
        ),
        (
            name = "Linear: Mixed Categorical + Continuous",
            model_func = lm,
            data_func = () -> make_categorical_test_data(n=600, include_continuous=true),
            formula = @formula(wage ~ education + experience + union_member),
            expected_agreement = 0.80  # Moderate for mixed model
        ),
        (
            name = "Logistic: Categorical Predictors", 
            model_func = (f, d) -> glm(f, d, Binomial(), LogitLink()),
            data_func = () -> make_categorical_test_data(n=800, include_continuous=false),
            formula = @formula(high_earner ~ education + region + union_member),
            expected_agreement = 0.75  # Lower for GLM categorical
        ),
        (
            name = "Logistic: Mixed Model",
            model_func = (f, d) -> glm(f, d, Binomial(), LogitLink()),
            data_func = () -> make_categorical_test_data(n=800, include_continuous=true),
            formula = @formula(high_earner ~ education + experience + union_member),
            expected_agreement = 0.70  # Lower for complex GLM
        )
    ]
    
    individual_results = []
    
    for model_config in categorical_test_models
        if verbose
            @debug "Testing: $(model_config.name)"
        end
        
        data = model_config.data_func()
        
        try
            result = bootstrap_validate_categorical_effects(
                model_config.model_func, model_config.formula, data;
                n_bootstrap=n_bootstrap
            )
            
            if result.success
                agreement_rate = result.validation.agreement_rate
                meets_expectation = agreement_rate >= model_config.expected_agreement
                
                if verbose
                    @debug "Agreement Rate: $(round(agreement_rate * 100, digits=1))% (expected: ≥$(round(model_config.expected_agreement * 100, digits=1))%)"
                    @debug "Categorical Terms Tested: $(length(result.categorical_terms))"
                    @debug "Bootstrap Samples: $(result.n_bootstrap_successful)/$n_bootstrap successful"
                    
                    status = meets_expectation ? " PASSED" : " BELOW EXPECTATION"
                    @debug "Result: $status"
                end
                
                push!(individual_results, (
                    model_name = model_config.name,
                    success = true,
                    agreement_rate = agreement_rate,
                    expected_agreement = model_config.expected_agreement,
                    meets_expectation = meets_expectation,
                    n_categorical_terms = length(result.categorical_terms),
                    detailed_result = result
                ))
            else
                if verbose
                    reason = haskey(result, :reason) ? result.reason : "Unknown error"
                    @warn "  Failed: $reason"
                end
                
                push!(individual_results, (
                    model_name = model_config.name,
                    success = false,
                    agreement_rate = 0.0,
                    expected_agreement = model_config.expected_agreement,
                    meets_expectation = false,
                    n_categorical_terms = 0,
                    detailed_result = result
                ))
            end
            
        catch e
            @error "Categorical bootstrap test failed for $(model_config.name): $e"
            push!(individual_results, (
                model_name = model_config.name,
                success = false,
                agreement_rate = 0.0,
                expected_agreement = model_config.expected_agreement,
                meets_expectation = false,
                n_categorical_terms = 0,
                error = e
            ))
        end
        
        if verbose
            @debug ""  # Spacing
        end
    end
    
    # Overall assessment
    successful_models = [r for r in individual_results if r.success]
    models_meeting_expectation = [r for r in individual_results if r.meets_expectation]
    
    overall_success_rate = length(successful_models) / length(individual_results)
    expectation_success_rate = length(models_meeting_expectation) / length(individual_results)
    mean_agreement_rate = length(successful_models) > 0 ? mean([r.agreement_rate for r in successful_models]) : 0.0
    
    if verbose
        @debug "="^60
        @debug "CATEGORICAL BOOTSTRAP VALIDATION SUMMARY"
        @debug "="^60
        @debug "Total Models Tested: $(length(individual_results))"
        @debug "Successful Models: $(length(successful_models))/$(length(individual_results)) ($(round(overall_success_rate * 100, digits=1))%)"
        @debug "Models Meeting Expectations: $(length(models_meeting_expectation))/$(length(individual_results)) ($(round(expectation_success_rate * 100, digits=1))%)"
        if length(successful_models) > 0
            @debug "Mean Agreement Rate: $(round(mean_agreement_rate * 100, digits=1))%"
        end
        
        if expectation_success_rate >= 0.70  # Lower threshold for categorical effects
            @info " CATEGORICAL BOOTSTRAP VALIDATION: PASSED"
            @info "Categorical effects show acceptable bootstrap agreement!"
        else
            @warn "  CATEGORICAL BOOTSTRAP VALIDATION: NEEDS IMPROVEMENT"
            @info "Some categorical models below expectation"
        end
    end
    
    return (
        overall_success_rate = overall_success_rate,
        expectation_success_rate = expectation_success_rate,
        mean_agreement_rate = mean_agreement_rate,
        individual_results = individual_results,
        n_models_tested = length(individual_results)
    )
end

# Export categorical bootstrap testing functions
export make_categorical_test_data, bootstrap_validate_categorical_effects
export run_categorical_bootstrap_test_suite