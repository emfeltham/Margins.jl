# generate_data.jl
# Generate synthetic data for R comparison study
# Ensures identical data is used in both Julia and R analyses

using DataFrames, CSV
using CategoricalArrays

# Generate data using existing synthetic data generator
include("../support/generate_large_synthetic_data.jl")
df = generate_synthetic_dataset(5_000; seed = 08540)

println("Dataset generated: $(nrow(df)) rows, $(ncol(df)) columns")

# Export data to CSV
println("\nExporting data to CSV...")
CSV.write("r_comparison_data.csv", df)
println("Data exported to: r_comparison_data.csv")

# CRITICAL: Document categorical variable levels for R
# R must use identical factor levels and reference categories
println("\n" * "="^80)
println("=== CATEGORICAL VARIABLE LEVELS (for R factor conversion) ===")
println("="^80)
println("relation levels: ", levels(df[!, :relation]))
println("religion_c_p levels: ", levels(df[!, :religion_c_p]))
println("village_code levels: ", levels(df[!, :village_code]))
println("man_x levels: ", levels(df[!, :man_x]))
println("religion_c_x levels: ", levels(df[!, :religion_c_x]))
println("isindigenous_x levels: ", levels(df[!, :isindigenous_x]))
println("\nNOTE: First level in each array is the reference category.")
println("R factors MUST use identical level order for coefficient matching.")
println("="^80)

# Document data types for verification
println("\n" * "="^80)
println("=== DATA TYPES ===")
println("="^80)
println("\nBoolean variables (exported as 0/1 integers):")
for col in [:response, :socio4, :same_building, :kin431, :coffee_cultivation,
            :isindigenous_p, :man_p, :maj_catholic, :maj_indigenous]
    if col in names(df)
        println("  $col: ", eltype(df[!, col]))
    end
end

println("\nCategorical variables (exported as strings):")
for col in [:relation, :religion_c_p, :village_code, :perceiver,
            :man_x, :religion_c_x, :isindigenous_x]
    if col in names(df)
        println("  $col: ", eltype(df[!, col]), " (", length(levels(df[!, col])), " levels)")
    end
end

println("\nInteger variables:")
for col in [:num_common_nbs, :age_p, :schoolyears_p, :population, :degree_h, :degree_p]
    if col in names(df)
        println("  $col: ", eltype(df[!, col]))
    end
end

println("\n✓ Data generation complete!")
println("\nNext steps:")
println("1. Run: julia --project=. julia_model.jl")
println("2. Update factor levels in r_model.R based on levels printed above")
println("3. Run: Rscript r_model.R")
println("4. Run: julia --project=. compare_results.jl")
