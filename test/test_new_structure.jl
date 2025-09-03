# Test script for new MarginsResult structure

using Pkg
Pkg.activate(".")

using DataFrames, GLM
using Margins

# Create simple test data
n = 100
df = DataFrame(
    y = randn(n),
    x1 = randn(n), 
    x2 = randn(n)
)

# Fit model
model = lm(@formula(y ~ x1 + x2), df)

println("Testing new MarginsResult structure...")

# Test population margins
println("\n=== Population Margins ===")
pop_result = population_margins(model, df; type=:effects, vars=[:x1, :x2])

println("MarginsResult fields:")
println("  estimates: ", typeof(pop_result.estimates), " with ", length(pop_result.estimates), " elements")
println("  standard_errors: ", typeof(pop_result.standard_errors), " with ", length(pop_result.standard_errors), " elements")
println("  terms: ", pop_result.terms)
println("  profile_values: ", pop_result.profile_values)
println("  analysis_type: ", get(pop_result.metadata, :analysis_type, "missing"))

# Test DataFrame conversion with different formats
println("\nDataFrame(result) - auto format:")
df_auto = DataFrame(pop_result)
println(first(df_auto, 3))

println("\nDataFrame(result; format=:compact):")
df_compact = DataFrame(pop_result; format=:compact)
println(first(df_compact, 3))

println("\nDataFrame(result; format=:confidence):")
df_conf = DataFrame(pop_result; format=:confidence)
println(first(df_conf, 3))

# Test show method
println("\nInteractive display:")
show(pop_result)

println("\n\n=== Profile Margins ===")
prof_result = profile_margins(model, df; at=:means, type=:effects, vars=[:x1])

println("Profile MarginsResult fields:")
println("  estimates: ", typeof(prof_result.estimates), " with ", length(prof_result.estimates), " elements")
println("  terms: ", prof_result.terms)
println("  profile_values: ", prof_result.profile_values)
println("  analysis_type: ", get(prof_result.metadata, :analysis_type, "missing"))

# Test DataFrame conversion
println("\nDataFrame(prof_result) - auto format (should be :profile):")
df_prof_auto = DataFrame(prof_result)
println(df_prof_auto)

println("\nDataFrame(prof_result; format=:standard):")
df_prof_std = DataFrame(prof_result; format=:standard)
println(df_prof_std)

# Test error on invalid format
println("\nTesting invalid format error:")
try
    DataFrame(pop_result; format=:profile)
    println("ERROR: Should have failed!")
catch e
    println("✅ Correctly caught error: ", e)
end

println("\n✅ All tests completed!")