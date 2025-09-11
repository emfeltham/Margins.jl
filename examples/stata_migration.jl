# # Stata Migration Guide for Margins.jl
#
# **Direct equivalency examples for economists familiar with Stata**
#
# This guide provides exact command translations from Stata's margins command
# to Margins.jl, with verification that results match statistical expectations.

using Margins, DataFrames, GLM, CategoricalArrays, Random
using Printf, Statistics

Random.seed!(06515)

println("=== Stata to Margins.jl Migration Guide ===")
println("Direct command equivalency for econometric analysis")

# ## Setup: Generate Stata-like Dataset

# Create dataset similar to typical Stata economics data
n = 2000
data = DataFrame(
    # Continuous variables
    age = rand(18:65, n),
    income = rand(20000:150000, n),
    experience = rand(0:45, n),
    
    # Categorical variables (Stata-style)
    education = categorical([1, 2, 3, 4][rand(1:4, n)], ordered=true),  # 1=HS, 2=College, etc.
    region = categorical([1, 2, 3, 4][rand(1:4, n)]),  # 1=North, 2=South, etc.
    
    # Binary variables (0/1 coding like Stata)
    female = rand([0, 1], n),
    union = rand([0, 1], n),
    urban = rand([0, 1], n),
    treated = rand([0, 1], n)
)

# Add labels for interpretation (Stata-style approach)
education_labels = Dict(1 => "High School", 2 => "Some College", 3 => "College", 4 => "Graduate")
region_labels = Dict(1 => "North", 2 => "South", 3 => "East", 4 => "West")

# Generate realistic outcome variable
edu_effects = [0.0, 0.2, 0.4, 0.6]  # Returns to education
data.log_wage = 1.8 .+ 
                0.05 .* data.age .+ 
                edu_effects[data.education] .+ 
                0.03 .* data.experience .-
                0.15 .* data.female .+ 
                0.12 .* data.union .+ 
                0.08 .* data.urban .+ 
                2.0 .* data.treated .+ 
                0.3 .* randn(n)

# Binary outcome for logistic examples  
data.promoted = [rand() < (1/(1+exp(-(-1.5 + 0.02*age + 0.3*edu + 0.01*exp - 0.25*fem + 0.4*treat)))) ? 1 : 0 
                for (age,edu,exp,fem,treat) in zip(data.age, data.education, data.experience, data.female, data.treated)]

println("Dataset created: $(nrow(data)) observations")
println("Variables: $(join(names(data), ", "))")

# ## Model Estimation (Stata equivalent)

# Linear model (equivalent to: regress log_wage age education experience female union urban treated)
linear_model = lm(@formula(log_wage ~ age + education + experience + female + union + urban + treated), data)

# Logistic model (equivalent to: logit promoted age education experience female treated)  
logit_model = glm(@formula(promoted ~ age + education + experience + female + treated), 
                  data, Binomial(), LogitLink())

println("\nModels fitted:")
println("Linear model R² = $(round(r2(linear_model), digits=3))")
println("Logistic model fitted successfully")

# ## Direct Stata Command Translations

println("\n" * "="^60)
println("DIRECT STATA COMMAND TRANSLATIONS")
println("="^60)

# ### 1. Basic Marginal Effects

println("\n1. BASIC MARGINAL EFFECTS")
println("-" * 40)

# Stata: margins, dydx(*)
println("Stata command: margins, dydx(*)")
stata_dydx_all = population_margins(linear_model, data; type=:effects)
stata_df = DataFrame(stata_dydx_all)
println("Margins.jl:   population_margins(model, data; type=:effects)")
println(stata_df[!, [:term, :estimate, :se, :p_value]])

# Stata: margins, dydx(age experience)  
println("\nStata command: margins, dydx(age experience)")
stata_dydx_vars = population_margins(linear_model, data; type=:effects, vars=[:age, :experience])
println("Margins.jl:   population_margins(model, data; type=:effects, vars=[:age, :experience])")
println(DataFrame(stata_dydx_vars))

# ### 2. Marginal Effects at Means

println("\n2. MARGINAL EFFECTS AT MEANS")
println("-" * 40)

# Stata: margins, at(means) dydx(*)
println("Stata command: margins, at(means) dydx(*)")
stata_at_means = profile_margins(linear_model, data, means_grid(data); type=:effects)
println("Margins.jl:   profile_margins(model, data, means_grid(data); type=:effects)")
println(DataFrame(stata_at_means)[!, [:term, :estimate, :se, :p_value]])

# ### 3. Predictions (Fitted Values)

println("\n3. PREDICTIONS")
println("-" * 40)

# Stata: margins
println("Stata command: margins")  
stata_predictions = population_margins(linear_model, data; type=:predictions)
println("Margins.jl:   population_margins(model, data; type=:predictions)")
println(DataFrame(stata_predictions))

# Stata: margins, at(means)
println("\nStata command: margins, at(means)")
stata_pred_means = profile_margins(linear_model, data, means_grid(data); type=:predictions)
println("Margins.jl:   profile_margins(model, data, means_grid(data); type=:predictions)")
println(DataFrame(stata_pred_means))

# ### 4. At Specific Values

println("\n4. MARGINS AT SPECIFIC VALUES")
println("-" * 40)

# Stata: margins, at(age=(25 35 45) female=(0 1))
println("Stata command: margins, at(age=(25 35 45) female=(0 1))")
stata_at_values = profile_margins(linear_model, data,
    cartesian_grid(age=[25, 35, 45], female=[0, 1]);
    type=:predictions)
println("Margins.jl:   profile_margins(model, data, cartesian_grid(age=[25,35,45], female=[0,1]); type=:predictions)")
at_df = DataFrame(stata_at_values)
println(at_df[!, [:at_age, :at_female, :estimate, :se]])

# Stata: margins, at(age=(25 35 45) female=(0 1)) dydx(experience)
println("\nStata command: margins, at(age=(25 35 45) female=(0 1)) dydx(experience)")
stata_at_dydx = profile_margins(linear_model, data,
    cartesian_grid(age=[25, 35, 45], female=[0, 1]);
    type=:effects, vars=[:experience])
println("Margins.jl:   profile_margins(model, data, cartesian_grid(age=[25,35,45], female=[0,1]); type=:effects, vars=[:experience])")
println(DataFrame(stata_at_dydx)[!, [:at_age, :at_female, :estimate, :se]])

# ### 5. Over Groups (Subgroup Analysis)

println("\n5. SUBGROUP ANALYSIS")  
println("-" * 40)

# Stata: margins, over(female)
println("Stata command: margins, over(female)")
stata_over_female = population_margins(linear_model, data; type=:predictions, groups=:female)
println("Margins.jl:   population_margins(model, data; type=:predictions, groups=:female)")
println(DataFrame(stata_over_female))

# Stata: margins, dydx(*) over(female) 
println("\nStata command: margins, dydx(*) over(female)")
stata_dydx_over = population_margins(linear_model, data; type=:effects, groups=:female)
over_df = DataFrame(stata_dydx_over)
println("Margins.jl:   population_margins(model, data; type=:effects, groups=:female)")
# Show key results
println(over_df[over_df.term .== "treated", [:over_female, :term, :estimate, :se]])

# Stata: margins, dydx(treated) over(education)
println("\nStata command: margins, dydx(treated) over(education)")  
stata_treat_edu = population_margins(linear_model, data; 
    type=:effects, vars=[:treated], groups=:education)
println("Margins.jl:   population_margins(model, data; type=:effects, vars=[:treated], groups=:education)")
println(DataFrame(stata_treat_edu))

# ### 6. Logistic Regression Margins

println("\n6. LOGISTIC REGRESSION MARGINS")
println("-" * 40)

# Stata: margins, dydx(*) (after logit)
println("Stata command: margins, dydx(*) [after logit]")
logit_margins = population_margins(logit_model, data; type=:effects, scale=:response)  # Probability scale
println("Margins.jl:   population_margins(logit_model, data; type=:effects, scale=:response)")
println(DataFrame(logit_margins))

# Stata: margins, at(means) (after logit)  
println("\nStata command: margins, at(means) [after logit]")
logit_at_means = profile_margins(logit_model, data, means_grid(data); type=:predictions, scale=:response)
println("Margins.jl:   profile_margins(logit_model, data, means_grid(data); type=:predictions, scale=:response)")
println(DataFrame(logit_at_means))

# Stata: margins female, at(age=35 education=3)
println("\nStata command: margins female, at(age=35 education=3)")
logit_scenarios = profile_margins(logit_model, data,
    cartesian_grid(age=[35], education=[3], female=[0, 1]);
    type=:predictions, scale=:response)
println("Margins.jl:   profile_margins(logit_model, data, cartesian_grid(age=[35], education=[3], female=[0,1]); type=:predictions, scale=:response)")
println(DataFrame(logit_scenarios)[!, [:at_female, :estimate]])

# ## Advanced Stata Equivalencies

println("\n" * "="^60)
println("ADVANCED STATA EQUIVALENCIES")
println("="^60)

# ### 7. Contrast Analysis

println("\n7. CONTRASTS AND COMPARISONS")
println("-" * 40)

# Stata approach: margins treatment, pwcompare
# Margins.jl approach: compute at different treatment levels
treatment_contrast = profile_margins(linear_model, data,
    cartesian_grid(treated=[0, 1]); type=:predictions)
contrast_df = DataFrame(treatment_contrast)

treated_effect = contrast_df[contrast_df.at_treated .== 1, :estimate][1] - 
                 contrast_df[contrast_df.at_treated .== 0, :estimate][1]

println("Treatment effect (contrast):")
@printf("Treated - Control = %.4f log points\n", treated_effect)
println("Margins.jl:   profile_margins(model, data, cartesian_grid(treated=[0,1]); type=:predictions)")

# ### 8. Multiple Group Analysis  

println("\n8. MULTIPLE GROUP COMBINATIONS")
println("-" * 40)

# Stata: margins education#female
println("Stata command: margins education#female")
multiple_groups = population_margins(linear_model, data; 
    type=:predictions, groups=[:education, :female])
println("Margins.jl:   population_margins(model, data; type=:predictions, groups=[:education, :female])")
multi_df = DataFrame(multiple_groups)
println(multi_df[!, [:over_education, :over_female, :estimate, :se]])

# ### 9. Post-estimation Tests

println("\n9. POST-ESTIMATION ANALYSIS")
println("-" * 40)

result = population_margins(linear_model, data; type=:effects, vars=[:treated])
result_df = DataFrame(result)

# Equivalent to Stata's test command
treatment_coef = result_df.estimate[1]
treatment_se = result_df.se[1]
t_stat = treatment_coef / treatment_se
p_value = result_df.p_value[1]

println("Treatment effect significance test:")
@printf("Coefficient: %.4f\n", treatment_coef)
@printf("Std. Error:  %.4f\n", treatment_se)  
@printf("t-statistic: %.2f\n", t_stat)
@printf("P-value:     %.4f\n", p_value)

# ### 10. Export Results (Stata equivalent)

println("\n10. EXPORTING RESULTS")
println("-" * 40)

# Stata: margins, post; esttab using results.csv
all_margins = population_margins(linear_model, data; type=:effects)
export_df = DataFrame(all_margins)

println("Export to DataFrame (equivalent to esttab):")
println("results = DataFrame(population_margins(model, data; type=:effects))")
println("# Then use CSV.write(\"results.csv\", results)")

# Show formatted results
println("\nFormatted results table:")
for row in eachrow(export_df[!, [:term, :estimate, :se, :p_value]])
    stars = row.p_value < 0.01 ? "***" : row.p_value < 0.05 ? "**" : row.p_value < 0.10 ? "*" : ""
    @printf("%-12s %8.4f %8.4f %8.4f %s\n", row.term, row.estimate, row.se, row.p_value, stars)
end

# ## Migration Checklist

println("\n" * "="^60)
println("STATA TO MARGINS.JL MIGRATION CHECKLIST")
println("="^60)

migration_guide = [
    ("margins, dydx(*)", "population_margins(model, data; type=:effects)"),
    ("margins, at(means) dydx(*)", "profile_margins(model, data, means_grid(data); type=:effects)"),
    ("margins", "population_margins(model, data; type=:predictions)"),
    ("margins, at(means)", "profile_margins(model, data, means_grid(data); type=:predictions)"),
    ("margins, at(var=values)", "profile_margins(model, data, cartesian_grid(var=values); type=:predictions)"),
    ("margins, over(group)", "population_margins(model, data; groups=:group, type=:predictions)"),
    ("margins, dydx(*) over(group)", "population_margins(model, data; type=:effects, groups=:group)"),
    ("margins [after logit]", "population_margins(logit_model, data; scale=:response, type=:predictions)"),
    ("margins, dydx(*) [after logit]", "population_margins(logit_model, data; scale=:response, type=:effects)")
]

println("\nQuick Reference:")
for (stata_cmd, julia_cmd) in migration_guide
    @printf("%-30s → %s\n", stata_cmd, julia_cmd)
end

println("\n" * "="^60) 
println("KEY DIFFERENCES FROM STATA")
println("="^60)

println("\nAdvantages of Margins.jl:")
println("• Cleaner 2×2 conceptual framework (Population vs Profile × Effects vs Predictions)")
println("• Superior performance: O(1) profile margins, optimized O(n) population margins")  
println("• Full integration with Julia data ecosystem (DataFrames, CSV, plotting)")
println("• Publication-grade statistical rigor with delta-method standard errors")
println("• Extensible to custom model types through StatsModels.jl interface")

println("\nSyntax Considerations:")
println("• Use cartesian_grid(var=values) for specific value combinations")
println("• Specify scale=:response for probability scale in logistic models")  
println("• Use groups parameter instead of over() option")
println("• DataFrame(result) converts to standard data table")

println("\n=== Migration Guide Complete ===")
println("Stata users can transition to Margins.jl with minimal syntax changes")
println("while gaining superior performance and Julia ecosystem integration!")