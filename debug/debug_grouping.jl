#!/usr/bin/env julia

# Debug script to understand grouping parsing

using Margins, GLM, DataFrames, CategoricalArrays

# Create test data 
n = 100
data = DataFrame(
    y = randn(n),
    x1 = randn(n),
    x2 = randn(n),
    education = CategoricalArray(rand(["HS", "College"], n)),
    gender = CategoricalArray(rand(["Male", "Female"], n))
)

data_nt = Tables.columntable(data)

println("Data summary:")
println("Education unique values: ", unique(data.education))
println("Gender unique values: ", unique(data.gender))
println("Education count: HS=", sum(data.education .== "HS"), ", College=", sum(data.education .== "College"))
println("Gender count: Male=", sum(data.gender .== "Male"), ", Female=", sum(data.gender .== "Female"))

# Test what the group parsing functions return
println("\n" * repeat("=", 50))
println("DEBUGGING GROUP PARSING")
println(repeat("=", 50))

# Test 1: Simple grouping
println("\nTest 1: Simple grouping - :education")
try
    # Access the internal parsing function
    group_specs = Margins._parse_groups_specification(:education, data_nt)
    println("Group specs: ", group_specs)
    println("Number of groups: ", length(group_specs))
    for (i, spec) in enumerate(group_specs)
        println("  Group $i: ", spec)
    end
catch e
    println("ERROR: ", e)
end

# Test 2: Cross-tabulation
println("\nTest 2: Cross-tabulation - [:education, :gender]")
try
    group_specs = Margins._parse_groups_specification([:education, :gender], data_nt)
    println("Group specs: ", group_specs)
    println("Number of groups: ", length(group_specs))
    for (i, spec) in enumerate(group_specs)
        println("  Group $i: ", spec)
    end
catch e
    println("ERROR: ", e)
end

# Test 3: Test context data creation
println("\n" * repeat("-", 30))
println("DEBUGGING CONTEXT DATA CREATION")
println(repeat("-", 30))

try
    # Test with simple grouping
    group_specs = Margins._parse_groups_specification(:education, data_nt)
    
    for (i, group_spec) in enumerate(group_specs)
        context_data = Margins._create_context_data(data_nt, Dict(), group_spec)
        println("Group $i ($(group_spec)): n_obs = ", length(first(context_data)))
        if haskey(context_data, :education)
            println("  Education values: ", unique(context_data.education))
        end
    end
catch e
    println("ERROR: ", e)
end