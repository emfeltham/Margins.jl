#!/usr/bin/env julia

# Debug script to understand context indices

using Margins, GLM, DataFrames, CategoricalArrays

# Create test data with clear group differences
n = 20
data = DataFrame(
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,  # HS: 1-10
         11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],  # College: 11-20
    x1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # HS: all x1=1
          2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],   # College: all x1=2
    education = CategoricalArray([repeat(["HS"], 10); repeat(["College"], 10)])
)

println("Data with clear group differences:")
println(data)

data_nt = Tables.columntable(data)
println("\nData by group:")
println("HS rows: ", findall(==("HS"), data.education))
println("College rows: ", findall(==("College"), data.education))

# Test group parsing
println("\n" * repeat("=", 50))
println("DEBUGGING GROUP INDICES")
println(repeat("=", 50))

try
    group_specs = Margins._parse_groups_specification(:education, data_nt)
    println("Group specs: ", group_specs)
    
    for (i, group_spec) in enumerate(group_specs)
        context_data, context_indices = Margins._create_context_data(data_nt, Dict(), group_spec)
        println("Group $i ($(group_spec)):")
        println("  Indices: ", context_indices)
        println("  N obs: ", length(context_indices))
        println("  x1 values: ", unique(context_data.x1))
        println("  y values: ", context_data.y)
    end
catch e
    println("ERROR: ", e)
end

# Fit model and test
model = lm(@formula(y ~ x1 + education), data)
println("\nModel coefficients: ", coef(model))

# Test marginal effects
println("\n" * repeat("=", 30))
println("MARGINAL EFFECTS TEST")
println(repeat("=", 30))

base_result = population_margins(model, data; type=:effects, vars=[:x1])
println("Base result: ", DataFrame(base_result))

grouped_result = population_margins(model, data; type=:effects, vars=[:x1], groups=:education)
println("Grouped result: ", DataFrame(grouped_result))