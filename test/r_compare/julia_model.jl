# julia_model.jl
# Julia model estimation for R comparison study
# Loads data from CSV (same file as R script)

using GLM, Margins, DataFrames, CSV
using CategoricalArrays

println("="^80)
println("Julia Model Estimation - R Comparison Study")
println("="^80)

# Load data from CSV (same file R will use)
println("\nLoading data from CSV...")
# test/r_compare/
df = CSV.read("r_comparison_data.csv", DataFrame; pool=false, stringtype=String);
df = copy(df)
println("Loaded: $(nrow(df)) rows, $(ncol(df)) columns")

# Convert data types to match Julia expectations
println("\nConverting data types...")

# Boolean variables (from 0/1 to Bool)
for col in [:response, :socio4, :same_building, :kin431, :coffee_cultivation,
            :isindigenous_p, :man_p, :maj_catholic, :maj_indigenous]
    if col in Symbol.(names(df))
        df[!, col] = convert(Vector{Bool}, df[!, col])
    end
end

# Categorical variables (from String to CategoricalArray)
# Convert explicitly to String first to handle InlineString types from CSV
for col in [:relation, :religion_c_p, :village_code, :perceiver, :alter1, :alter2,
            :man_x, :religion_c_x, :isindigenous_x]
    if col in Symbol.(names(df))
        df[!, col] = categorical(df[!, col])
    end
end

# Integer variables (ensure proper Int type)
for col in [:num_common_nbs, :age_p, :schoolyears_p, :population, :degree_h, :degree_p]
    if col in Symbol.(names(df))
        df[!, col] = convert(Vector{Int}, df[!, col])
    end
end

println("Data types converted successfully")

# Verify categorical levels
println("\n" * "="^80)
println("Categorical variable levels:")
println("="^80)
println("relation: ", levels(df[!, :relation]))
println("religion_c_p: ", levels(df[!, :religion_c_p]))
println("village_code: ", levels(df[!, :village_code]))

# R-compatible complex model
fx = @formula(response ~
    # Base effects
    socio4 + dists_p_inv + dists_a_inv + are_related_dists_a_inv +

    # Binary × continuous interactions
    socio4 & (dists_p_inv + are_related_dists_a_inv + dists_a_inv) +

    # Individual-level variables with interactions
    (age_p + wealth_d1_4_p + schoolyears_p + man_p +
     same_building + population + hhi_religion + hhi_indigenous +
     coffee_cultivation + market) & (1 + socio4 + are_related_dists_a_inv) +

    # Categorical variables
    relation + religion_c_p +
    relation & socio4 +
    religion_c_p & are_related_dists_a_inv +

    # Tie-level homophily measures
    degree_a_mean + degree_h + age_a_mean + schoolyears_a_mean + wealth_d1_4_a_mean +

    # Continuous × continuous interactions
    age_h & age_h_nb_1_socio +
    schoolyears_h & schoolyears_h_nb_1_socio +
    wealth_d1_4_h & wealth_d1_4_h_nb_1_socio +

    # Tie-level interactions with socio4
    (degree_a_mean + degree_h + age_a_mean + schoolyears_a_mean +
     wealth_d1_4_a_mean) & socio4 +

    # Composition effects
    hhi_religion & are_related_dists_a_inv +
    hhi_indigenous & are_related_dists_a_inv
)

# Fit model
println("\n" * "="^80)
println("Fitting logistic regression model...")
println("="^80)
@time m = fit(GeneralizedLinearModel, fx, df, Bernoulli(), LogitLink())

println("\nModel fitted successfully!")
println("Number of coefficients: ", length(coef(m)))

# Export model coefficients for comparison
println("\nExporting model coefficients...")
coef_names = coefnames(m)
coef_vals = coef(m)
coef_se = stderror(m)
coef_df = DataFrame(
    term = coef_names,
    estimate = coef_vals,
    std_error = coef_se
)
CSV.write("julia_coefficients.csv", coef_df)

# Test all marginal effects specifications
cg = cartesian_grid(socio4 = [false, true], are_related_dists_a_inv = [1, 1/6])

println("\n" * "="^80)
println("APM (Adjusted Predictions at Profiles):")
println("="^80)
@time apm_result = profile_margins(m, df, cg; type=:predictions)
display(DataFrame(apm_result))

println("\n" * "="^80)
println("MEM (Marginal Effects at Profiles, pairwise contrasts):")
println("="^80)
@time mem_result = profile_margins(m, df, cg; type=:effects, contrasts=:pairwise)
display(DataFrame(mem_result))

println("\n" * "="^80)
println("AAP (Average Adjusted Predictions):")
println("="^80)
@time aap_result = population_margins(m, df; type=:predictions)
display(DataFrame(aap_result))

println("\n" * "="^80)
println("AME (Average Marginal Effects - all variables):")
println("="^80)
@time ame_result = population_margins(m, df; type=:effects)
println("Computed AME for $(nrow(DataFrame(ame_result))) variables")

println("\n" * "="^80)
println("AME (single variable - age_h):")
println("="^80)
@time ame_age = population_margins(m, df; type=:effects, vars=[:age_h])
display(DataFrame(ame_age))

println("\n" * "="^80)
println("AME (with scenario - wealth variables at are_related_dists_a_inv=1/6):")
println("="^80)
@time ame_scenario = population_margins(
    m, df; type=:effects,
    vars=[:wealth_d1_4_p, :wealth_d1_4_h],
    scenarios=(are_related_dists_a_inv=[1/6],)
)
display(DataFrame(ame_scenario))

# Export results for comparison with R
println("\n" * "="^80)
println("Exporting results...")
println("="^80)
CSV.write("julia_apm.csv", DataFrame(apm_result))
CSV.write("julia_mem.csv", DataFrame(mem_result))
CSV.write("julia_aap.csv", DataFrame(aap_result))
CSV.write("julia_ame.csv", DataFrame(ame_result))
CSV.write("julia_ame_age.csv", DataFrame(ame_age))
CSV.write("julia_ame_scenario.csv", DataFrame(ame_scenario))

println("\nAll results exported to current directory")
println("\n✓ Julia analysis complete!")
println("\nNext steps:")
println("1. Update factor levels in r_model.R (if not done already)")
println("2. Run: Rscript r_model.R")
println("3. Run: julia --project=. compare_results.jl")
