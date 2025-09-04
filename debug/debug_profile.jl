using Pkg
Pkg.activate(".")

using Margins, GLM, DataFrames, StatsModels, CategoricalArrays
using Random

# Create small test dataset - SIMPLIFIED TO ONLY CONTINUOUS VARIABLES
Random.seed!(123)
data = DataFrame(
    y = randn(10),
    x1 = randn(10),
    x2 = randn(10)
)

model = lm(@formula(y ~ x1 + x2), data)

println("Data size: ", nrow(data))

# Debug: check reference grid building
data_nt = Tables.columntable(data)
reference_grid = Margins._build_reference_grid(:means, data_nt)
println("Reference grid:")
println(reference_grid)
println("Reference grid size: ", nrow(reference_grid))

# Debug: check profiles conversion
profiles = [Dict(pairs(row)) for row in eachrow(reference_grid)]
println("Number of profiles: ", length(profiles))
println("Profile 1: ", profiles[1])

# Debug: check refgrid_data building
profile = profiles[1]
refgrid_data = Margins._build_refgrid_data(profile, data_nt)
println("refgrid_data:")
for (k, v) in pairs(refgrid_data)
    println("  $k: $v (length=$(length(v)))")
end

# Run profile margins
println("\n=== Running profile margins ===")

# Let me intercept the profile conversion step to see where the bug is
data_nt_for_profile = Tables.columntable(data) 
reference_grid_in_profile = Margins._build_reference_grid(:means, data_nt_for_profile)
println("Reference grid in profile function:")
println(reference_grid_in_profile)
println("Size: $(nrow(reference_grid_in_profile))")

profiles_in_profile = [Dict(pairs(row)) for row in eachrow(reference_grid_in_profile)]
println("Number of profiles after conversion: ", length(profiles_in_profile))

# Let's also check eachrow behavior on our data
println("Checking eachrow behavior on original data:")
for (i, row) in enumerate(eachrow(data))
    println("  Row $i: $row")
    if i > 3 break end  # Only show first few
end

println("Checking eachrow behavior on reference_grid:")
for (i, row) in enumerate(eachrow(reference_grid_in_profile))
    println("  Row $i: $row")
end

grid = means_grid(data)
result = profile_margins(model, data, grid, type=:effects, vars=[:x1])
df = DataFrame(result)
println("Number of results: ", nrow(df))
println("First few estimates: ", df.estimate[1:min(5, end)])

# Check if any columns have unexpected length
for (k, v) in pairs(refgrid_data) 
    if length(v) != 1
        println("WARNING: Column $k has length $(length(v)), expected 1")
    end
end