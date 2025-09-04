# # Basic Workflow with Margins.jl
# 
# **Essential patterns for new users**
#
# This example demonstrates the fundamental Margins.jl workflow using the clean 2×2 framework:
# Population vs Profile × Effects vs Predictions

using Margins, DataFrames, GLM, Random
using Statistics

# Set seed for reproducible results
Random.seed!(06515)

println("=== Margins.jl Basic Workflow ===")
println("Demonstrating the 2×2 framework: Population vs Profile × Effects vs Predictions")

# ## 1. Data Setup

# Generate sample dataset
n = 1000
data = DataFrame(
    # Continuous variables
    age = rand(25:65, n),
    income = rand(30000:120000, n),
    
    # Categorical variables  
    education = rand(["High School", "College", "Graduate"], n),
    region = rand(["North", "South", "East", "West"], n),
    
    # Binary outcome
    treatment = rand([0, 1], n)
)

# Generate realistic outcome variable
# Outcome depends on age, income, education, and treatment
education_effects = Dict("High School" => 0.0, "College" => 0.3, "Graduate" => 0.6)
edu_numeric = [education_effects[edu] for edu in data.education]

data.outcome = 50 .+ 
               0.2 .* (data.age .- 45) .+
               0.00001 .* (data.income .- 75000) .+
               edu_numeric .+
               3.0 .* data.treatment .+
               2.0 .* randn(n)

println("Dataset: $(nrow(data)) observations, $(ncol(data)) variables")
println("First 5 rows:")
println(first(data, 5))

# ## 2. Model Fitting

# Fit linear model
model = lm(@formula(outcome ~ age + income + education + region + treatment), data)

println("\n=== Model Summary ===")
println(model)

# ## 3. The 2×2 Framework

println("\n=== The 2×2 Framework ===")
println("Population vs Profile × Effects vs Predictions")

# ### Population Analysis (AME/AAP)
# Average effects/predictions across your observed sample distribution

println("\n--- Population Analysis ---")

# Population Average Marginal Effects (AME)
ame_result = population_margins(model, data; type=:effects)
println("Population Average Marginal Effects:")
println(DataFrame(ame_result))

# Population Average Adjusted Predictions (AAP)
aap_result = population_margins(model, data; type=:predictions)
println("\nPopulation Average Adjusted Predictions:")
println(DataFrame(aap_result))

# ### Profile Analysis (MEM/APM)  
# Effects/predictions at specific covariate scenarios

println("\n--- Profile Analysis ---")

# Marginal Effects at Sample Means (MEM)
mem_result = profile_margins(model, data; at=:means, type=:effects)
println("Marginal Effects at Sample Means:")
println(DataFrame(mem_result))

# Adjusted Predictions at Sample Means (APM)
apm_result = profile_margins(model, data; at=:means, type=:predictions)
println("\nAdjusted Predictions at Sample Means:")
println(DataFrame(apm_result))

# ## 4. Profile Specification Patterns

println("\n=== Profile Specification Examples ===")

# ### Scenario Analysis
# Effects at specific policy scenarios

scenarios = Dict(
    :age => [30, 45, 60],           # Different age groups
    :treatment => [0, 1],           # With/without treatment
    :education => ["High School", "College", "Graduate"]  # Education levels
)

scenario_effects = profile_margins(model, data; at=scenarios, type=:effects)
scenario_df = DataFrame(scenario_effects)

println("Treatment effects across age and education scenarios:")
treatment_effects = scenario_df[scenario_df.term .== "treatment", :]
println(treatment_effects[!, [:at_age, :at_education, :at_treatment, :estimate, :se]])

# ### Custom Reference Grid
# Maximum control using DataFrame specification

custom_grid = DataFrame(
    age = [35, 50, 35],
    income = [50000, 75000, 100000],  
    education = ["College", "College", "Graduate"],
    region = ["North", "South", "East"],
    treatment = [1, 1, 1]  # All treated
)

custom_predictions = profile_margins(model, custom_grid; type=:predictions)
println("\nPredictions at custom scenarios:")
println(DataFrame(custom_predictions))

# ## 5. Subgroup Analysis

println("\n=== Subgroup Analysis ===")

# Effects within education groups
education_effects = population_margins(model, data; type=:effects, groups=:education)
println("Average marginal effects by education level:")
println(DataFrame(education_effects))

# Effects within region and education groups
region_edu_effects = population_margins(model, data; 
                                       type=:effects, 
                                       over=[:region, :education],
                                       vars=[:treatment])  # Focus on treatment effect
println("\nTreatment effects by region and education:")
println(DataFrame(region_edu_effects))

# ## 6. Comparing Population vs Profile

println("\n=== Population vs Profile Comparison ===")

# For linear models, these should be very similar
pop_treatment = DataFrame(population_margins(model, data; type=:effects, vars=[:treatment]))
prof_treatment = DataFrame(profile_margins(model, data; at=:means, type=:effects, vars=[:treatment]))

println("Treatment effect comparison:")
println("Population (AME): $(round(pop_treatment.estimate[1], digits=3)) ± $(round(pop_treatment.se[1], digits=3))")
println("Profile (MEM):    $(round(prof_treatment.estimate[1], digits=3)) ± $(round(prof_treatment.se[1], digits=3))")
println("Difference:       $(round(abs(pop_treatment.estimate[1] - prof_treatment.estimate[1]), digits=4))")

# ## 7. Working with Results

println("\n=== Working with Results ===")

# MarginsResult implements Tables.jl interface
result = population_margins(model, data; type=:effects)

# Convert to DataFrame for analysis
df = DataFrame(result)
println("Result structure:")
println(names(df))

# Select significant effects (p < 0.05)
significant = df[df.p_value .< 0.05, :]
println("\nStatistically significant effects (p < 0.05):")
println(significant[!, [:term, :estimate, :se, :p_value]])

# Export to CSV (example)
# using CSV
# CSV.write("marginal_effects.csv", df)

println("\n=== Basic Workflow Complete ===")
println("Key takeaways:")
println("1. Population analysis gives average effects across your sample")  
println("2. Profile analysis gives effects at specific scenarios")
println("3. Use population for population parameters, profile for concrete interpretation")
println("4. Both approaches provide delta-method standard errors")
println("5. Results integrate seamlessly with DataFrames ecosystem")