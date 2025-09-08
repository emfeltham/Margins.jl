# Debug script to see which code path is being taken
using Margins, GLM, DataFrames, Tables

println("=== Checking Which Code Path Is Taken ===\n")

# Create test data like the allocation test
n_rows = 1000
df = DataFrame(
    y = randn(n_rows),
    x = randn(n_rows)
)
model = lm(@formula(y ~ x), df)

println("Calling population_margins with n_rows = $n_rows")

# Trace the execution by modifying the functions temporarily
# Let's add some debug prints to see which path is taken
result = population_margins(model, df; type=:effects, vars=[:x])

println("Result obtained successfully")
println("Estimate: $(result.estimates[1])")
println("Standard Error: $(result.standard_errors[1])")

# Try to manually trace which function gets called
data_nt = Tables.columntable(df)
engine = Margins.get_or_build_engine(model, data_nt, [:x], GLM.vcov)

println("\nTesting _compute_variable_ame_unified directly:")
var_type = Margins._detect_variable_type(engine.data_nt, :x)
println("Variable type detected: $var_type")

rows = 1:n_rows
scale = :response
backend = :ad

if var_type === :continuous
    println("Should call _compute_continuous_ame")
    ame_val, gradient = Margins._compute_continuous_ame(engine, :x, rows, scale, backend)
    println("Direct call result - AME: $ame_val, Gradient norm: $(sum(abs.(gradient)))")
end