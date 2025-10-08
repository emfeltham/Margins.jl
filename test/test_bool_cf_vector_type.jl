# Check what type of CounterfactualVector is used for Bool variables
using Margins, GLM, DataFrames, Tables
using Margins: build_engine, PopulationUsage, HasDerivatives

n = 100
df_bool = DataFrame(y = randn(n), x = randn(n), treated = rand([false, true], n))
model_bool = lm(@formula(y ~ x + treated), df_bool)
data_nt_bool = Tables.columntable(df_bool)
engine_bool = build_engine(PopulationUsage, HasDerivatives, model_bool, data_nt_bool, [:treated], GLM.vcov, :ad)

println("ContrastEvaluator type: ", typeof(engine_bool.contrast))
println("Counterfactuals: ", typeof(engine_bool.contrast.counterfactuals))
println("Variables: ", engine_bool.contrast.vars)

# Get the counterfactual vector for :treated
cf_tuple = engine_bool.contrast.counterfactuals
println("\nCounterfactual tuple length: ", length(cf_tuple))
for (i, cf) in enumerate(cf_tuple)
    println("  [$i] ", typeof(cf))
end

# Check if there's a categorical_level_maps
println("\nCategorical level maps: ", engine_bool.contrast.categorical_level_maps)
println("Has key :treated? ", haskey(engine_bool.contrast.categorical_level_maps, :treated))
