# # Advanced Econometric Analysis with Margins.jl
#
# **Complete econometric workflows for research and policy analysis**
#
# This example demonstrates advanced features including elasticities, policy analysis,
# robust standard errors, and complex econometric modeling patterns.

using Margins, DataFrames, GLM, CategoricalArrays, Random
using Statistics, Distributions, LinearAlgebra
using StatsModels

# For robust standard errors (optional dependency)
# using CovarianceMatrices

Random.seed!(123)

println("=== Advanced Econometric Analysis ===")

# ## 1. Generate Realistic Economic Dataset

function generate_economic_data(n=3000)
    """Generate realistic dataset for labor economics analysis"""
    
    df = DataFrame(
        # Demographics
        age = rand(22:65, n),
        female = rand([0, 1], n),
        married = rand([0, 1], n),
        
        # Human capital
        education = categorical(
            sample(["Less than HS", "High School", "Some College", "Bachelor's", "Graduate"], 
                   Weights([0.10, 0.30, 0.25, 0.25, 0.10]), n)
        ),
        experience = rand(0:40, n),
        
        # Job characteristics
        union = rand([0, 1], n),
        urban = rand([0, 1], n),
        
        # Industry (with realistic frequencies)
        industry = categorical(
            sample(["Manufacturing", "Services", "Technology", "Healthcare", "Education", "Government"],
                   Weights([0.20, 0.25, 0.15, 0.15, 0.10, 0.15]), n)
        ),
        
        # Regional variables
        region = categorical(rand(["Northeast", "South", "Midwest", "West"], n)),
        unemployment_rate = rand(Normal(6.0, 2.5), n),
        
        # Firm characteristics
        firm_size = rand(LogNormal(3.5, 1.2), n)
    )
    
    # Ensure realistic bounds
    df.experience = min.(df.experience, df.age .- 18)
    df.unemployment_rate = clamp.(df.unemployment_rate, 2.0, 15.0)
    
    # Education premium structure
    education_premiums = Dict(
        "Less than HS" => 0.0,
        "High School" => 0.12,
        "Some College" => 0.25,
        "Bachelor's" => 0.45,
        "Graduate" => 0.70
    )
    edu_numeric = [education_premiums[string(edu)] for edu in df.education]
    
    # Industry wage premiums
    industry_premiums = Dict(
        "Technology" => 0.25,
        "Healthcare" => 0.15,
        "Manufacturing" => 0.10,
        "Government" => 0.08,
        "Education" => 0.05,
        "Services" => 0.0  # Reference category
    )
    industry_numeric = [industry_premiums[string(ind)] for ind in df.industry]
    
    # Regional cost of living adjustments
    region_adjustments = Dict(
        "Northeast" => 0.15,
        "West" => 0.12,
        "South" => 0.0,
        "Midwest" => 0.03
    )
    region_numeric = [region_adjustments[string(reg)] for reg in df.region]
    
    # Generate realistic log wages using Mincer equation with extensions
    log_wage = 2.2 +                                    # Base wage
               0.08 * df.age +                          # Age (experience proxy)
               -0.0008 * df.age.^2 +                    # Age squared (diminishing returns)
               edu_numeric +                            # Education premium
               0.025 * df.experience +                  # Experience effect
               -0.0003 * df.experience.^2 +             # Experience squared
               -0.18 * df.female +                      # Gender wage gap
               0.05 * df.married +                      # Marriage premium
               0.12 * df.union +                        # Union premium
               0.08 * df.urban +                        # Urban premium
               industry_numeric +                       # Industry effects
               region_numeric +                         # Regional adjustments
               0.12 * log.(df.firm_size) +              # Firm size premium
               -0.025 * df.unemployment_rate +          # Labor market conditions
               rand(Normal(0, 0.35), n)                 # Random error
    
    df.log_wage = log_wage
    df.wage = exp.(log_wage)
    
    # Create binary outcome: promotion probability
    promotion_logit = -2.5 +
                     0.04 * df.age +
                     0.8 * edu_numeric +
                     0.035 * df.experience +
                     -0.4 * df.female +
                     0.3 * df.union +
                     0.2 * df.urban +
                     0.15 * log.(df.firm_size) +
                     rand(Normal(0, 0.8), n)
    
    df.promotion = [rand() < (1/(1+exp(-logit))) ? 1 : 0 for logit in promotion_logit]
    
    return df
end

data = generate_economic_data(3000)
println("Economic dataset: $(nrow(data)) observations")
println("Variables: $(names(data))")

# ## 2. Labor Economics Analysis: Wage Determination

println("\n=== Wage Determination Analysis ===")

# Fit wage equation with interactions
wage_model = lm(@formula(log_wage ~ age + I(age^2) + education + experience + I(experience^2) + 
                         female + married + union + urban + industry + region + 
                         log(firm_size) + unemployment_rate + female*education), data)

println("Wage model R² = $(round(r2(wage_model), digits=3))")

# ### Population Analysis with Subgroups

println("\n--- Population Average Effects ---")

# Overall average marginal effects
overall_ame = population_margins(wage_model, data; type=:effects, 
                                vars=[:age, :experience, :unemployment_rate])
println("Overall population effects:")
println(DataFrame(overall_ame))

# Gender wage gap analysis
gender_effects = population_margins(wage_model, data; type=:effects, groups=:female)
println("\nEffects by gender:")
gender_df = DataFrame(gender_effects)
println(gender_df[gender_df.term .== "experience", [:over_female, :estimate, :se, :p_value]])

# Education-specific analysis
education_effects = population_margins(wage_model, data; type=:effects, 
                                     vars=[:experience], over=:education)
println("\nExperience returns by education level:")
edu_df = DataFrame(education_effects)
println(edu_df[!, [:over_education, :estimate, :se, :conf_low, :conf_high]])

# ### Advanced Profile Analysis

println("\n--- Profile Analysis: Policy Scenarios ---")

# Career progression scenarios
career_scenarios = Dict(
    :age => [25, 35, 45, 55],
    :experience => [2, 12, 22, 32],
    :education => ["High School", "Bachelor's", "Graduate"],
    :female => [0, 1]
)

career_predictions = profile_margins(wage_model, data, cartesian_grid(data; age=[25,35,45,55], experience=[2,12,22,32], education=["High School","Bachelor's","Graduate"], female=[0,1]);
                                   type=:predictions, scale=:response)
career_df = DataFrame(career_predictions)

# Focus on wage levels by education and gender
println("Predicted wages by career stage, education, and gender:")
wage_comparison = career_df[!, [:at_age, :at_education, :at_female, :estimate]]
wage_comparison.wage = exp.(wage_comparison.estimate)  # Convert from log wage
println(wage_comparison[1:12, :])  # Show first 12 rows

# Gender wage gap quantification
gap_analysis = profile_margins(wage_model, data, cartesian_grid(data; education=["High School", "Bachelor's", "Graduate"], experience=[5, 15, 25], female=[0, 1]);
    type=:predictions
)

gap_df = DataFrame(gap_analysis)
println("\nGender wage gap analysis (log points):")
for edu in ["High School", "Bachelor's", "Graduate"]
    for exp in [5, 15, 25]
        male_wage = gap_df[(gap_df.at_education .== edu) .& 
                          (gap_df.at_experience .== exp) .& 
                          (gap_df.at_female .== 0), :estimate][1]
        female_wage = gap_df[(gap_df.at_education .== edu) .& 
                            (gap_df.at_experience .== exp) .& 
                            (gap_df.at_female .== 1), :estimate][1]
        gap = male_wage - female_wage
        println("$edu, $exp years exp: $(round(gap, digits=3)) log points ($(round((exp(gap)-1)*100, digits=1))%)")
    end
end

# ## 3. Elasticity Analysis

println("\n=== Elasticity Analysis ===")

# Population average elasticities
elasticities = population_margins(wage_model, data; type=:effects, 
                                measure=:elasticity, 
                                vars=[:age, :experience])
println("Population average elasticities:")
println(DataFrame(elasticities))

# Elasticities by education level
edu_elasticities = profile_margins(wage_model, data, cartesian_grid(data; education=["High School", "Bachelor's", "Graduate"]);
    type=:effects,
    measure=:elasticity,
    vars=[:age, :experience]
)
println("\nElasticities by education level:")
elas_df = DataFrame(edu_elasticities)
println(elas_df[!, [:at_education, :term, :estimate, :se]])

# Semi-elasticity: unemployment rate effect
unemployment_semi = population_margins(wage_model, data;
                                     measure=:semielasticity_eydx,
                                     vars=[:unemployment_rate])
semi_df = DataFrame(unemployment_semi)
println("\nUnemployment rate semi-elasticity:")
println("$(round(semi_df.estimate[1], digits=3)) = $(round(semi_df.estimate[1]*100, digits=1))% wage change per percentage point unemployment")

# ## 4. Binary Outcome Analysis: Promotion Probability

println("\n=== Promotion Probability Analysis ===")

# Fit logistic model for promotion
promotion_model = glm(@formula(promotion ~ age + education + experience + 
                              female + union + urban + log(firm_size) + 
                              female*education), 
                     data, Binomial(), LogitLink())

println("Promotion model fitted (logistic regression)")

# Average marginal effects on probability scale
promotion_ame = population_margins(promotion_model, data; type=:effects, scale=:response)
println("\nAverage marginal effects on promotion probability:")
println(DataFrame(promotion_ame))

# Promotion probability by demographic scenarios
promotion_scenarios = profile_margins(promotion_model, data, cartesian_grid(data; education=["High School", "Bachelor's", "Graduate"], experience=[5, 15, 25], female=[0, 1]);
    type=:predictions,
    scale=:response  # Probability scale
)

promo_df = DataFrame(promotion_scenarios)
println("\nPromotion probabilities by education, experience, and gender:")
println(promo_df[!, [:at_education, :at_experience, :at_female, :estimate]])

# ## 5. Policy Counterfactual Analysis

println("\n=== Policy Counterfactual Analysis ===")

# Current vs. policy scenario: universal college education
current_scenario = population_margins(wage_model, data;
    scenarios=Dict(:education => mix("High School" => 0.35, "Some College" => 0.25, 
                                    "Bachelor's" => 0.30, "Graduate" => 0.10)),
    type=:predictions
)

policy_scenario = population_margins(wage_model, data;
    scenarios=Dict(:education => mix("High School" => 0.10, "Some College" => 0.15,
                                    "Bachelor's" => 0.60, "Graduate" => 0.15)),
    type=:predictions
)

current_wage = exp(DataFrame(current_scenario).estimate[1])
policy_wage = exp(DataFrame(policy_scenario).estimate[1])
policy_impact = (policy_wage / current_wage - 1) * 100

println("Policy analysis: Increasing college graduation rates")
println("Current average wage: \$$(round(current_wage, digits=0))")
println("Policy scenario wage: \$$(round(policy_wage, digits=0))")
println("Policy impact: $(round(policy_impact, digits=1))% wage increase")

# ## 6. Robust Standard Errors (if CovarianceMatrices.jl available)

println("\n=== Robust Standard Errors ===")
println("(Requires CovarianceMatrices.jl for full functionality)")

# Example of how to use robust SEs when available:
# robust_model = glm(formula, data, Normal(), vcov=HC1())
# robust_effects = population_margins(robust_model, data; type=:effects)

# For now, show manual covariance specification
println("Standard covariance-based analysis completed.")
println("For robust/clustered SEs, install CovarianceMatrices.jl and use vcov parameter in glm().")

# ## 7. Summary and Best Practices

println("\n=== Analysis Summary ===")
println("Advanced econometric analysis completed:")
println("1. Wage determination with complex interactions")
println("2. Population vs. profile comparison across subgroups")  
println("3. Elasticity analysis for policy interpretation")
println("4. Binary outcome analysis (promotion probability)")
println("5. Policy counterfactual using categorical mixtures")
println("6. Framework for robust standard errors")

println("\n=== Best Practices Demonstrated ===")
println("• Use population analysis for average treatment effects")
println("• Use profile analysis for concrete policy scenarios")
println("• Elasticities provide scale-free effect interpretation")
println("• Categorical mixtures enable realistic population scenarios")
println("• Delta-method SEs maintain statistical rigor throughout")
println("• All analyses support subgroup decomposition and stratification")