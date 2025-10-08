# generate_data_large.jl
# Generate large synthetic dataset for performance benchmarking
# Separate from correctness validation workflow to avoid interference

using DataFrames, CSV
using CategoricalArrays

# Generate large dataset for performance testing
include("../support/generate_large_synthetic_data.jl")
df = generate_synthetic_dataset(500_000; seed = 08540)

println("Dataset generated: $(nrow(df)) rows, $(ncol(df)) columns")

# Export data to CSV
println("\nExporting data to CSV...")
CSV.write("r_comparison_data_large.csv", df)
println("Data exported to: r_comparison_data_large.csv")

# Document categorical variable levels for reference
println("\n" * "="^80)
println("=== CATEGORICAL VARIABLE LEVELS ===")
println("="^80)
println("relation levels: ", levels(df[!, :relation]))
println("religion_c_p levels: ", levels(df[!, :religion_c_p]))
println("village_code levels: ", levels(df[!, :village_code]))
println("man_x levels: ", levels(df[!, :man_x]))
println("religion_c_x levels: ", levels(df[!, :religion_c_x]))
println("isindigenous_x levels: ", levels(df[!, :isindigenous_x]))
println("="^80)

println("\n✓ Large dataset generation complete!")
println("\nNext steps:")
println("1. Run: make performance-large")
