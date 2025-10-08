# Investigate the type differences between boolean and string ContrastPairs
using Margins, GLM, DataFrames, Tables, CategoricalArrays
using Margins: generate_contrast_pairs

n = 100

# Boolean categorical
df_bool = DataFrame(
    y = randn(n),
    x = randn(n),
    treated = rand([false, true], n)
)
model_bool = lm(@formula(y ~ x + treated), df_bool)
data_nt_bool = Tables.columntable(df_bool)

# String categorical
df_string = DataFrame(
    y = randn(n),
    x = randn(n),
    group = categorical(rand(["Control", "Treatment"], n))
)
model_string = lm(@formula(y ~ x + group), df_string)
data_nt_string = Tables.columntable(df_string)

# Generate contrast pairs
rows = collect(1:50)
var_col_bool = getproperty(data_nt_bool, :treated)
var_col_string = getproperty(data_nt_string, :group)

pairs_bool = generate_contrast_pairs(var_col_bool, rows, :baseline, model_bool, :treated, data_nt_bool)
pairs_string = generate_contrast_pairs(var_col_string, rows, :baseline, model_string, :group, data_nt_string)

println("="^80)
println("ContrastPair Type Analysis")
println("="^80)

println("\nBoolean categorical:")
println("  Number of pairs: ", length(pairs_bool))
println("  Type of pairs:   ", typeof(pairs_bool))
if length(pairs_bool) > 0
    println("  First pair type: ", typeof(pairs_bool[1]))
    println("  First pair:      ", pairs_bool[1])
    println("  level1 type:     ", typeof(pairs_bool[1].level1))
    println("  level2 type:     ", typeof(pairs_bool[1].level2))
    println("  level1 value:    ", pairs_bool[1].level1)
    println("  level2 value:    ", pairs_bool[1].level2)
end

println("\nString categorical:")
println("  Number of pairs: ", length(pairs_string))
println("  Type of pairs:   ", typeof(pairs_string))
if length(pairs_string) > 0
    println("  First pair type: ", typeof(pairs_string[1]))
    println("  First pair:      ", pairs_string[1])
    println("  level1 type:     ", typeof(pairs_string[1].level1))
    println("  level2 type:     ", typeof(pairs_string[1].level2))
    println("  level1 value:    ", pairs_string[1].level1)
    println("  level2 value:    ", pairs_string[1].level2)
end

println("\n" * "="^80)
println("Analysis:")
println("="^80)

if !isempty(pairs_bool) && !isempty(pairs_string)
    t1 = typeof(pairs_bool[1])
    t2 = typeof(pairs_string[1])

    if t1 != t2
        println("\n⚠️  ContrastPair types differ!")
        println("    Boolean: $t1")
        println("    String:  $t2")
        println("\n    This type difference may cause type instability in the batch function,")
        println("    leading to allocations for boolean categoricals.")
    else
        println("\n✓ ContrastPair types are the same")
    end

    lt1 = typeof(pairs_bool[1].level1)
    lt2 = typeof(pairs_string[1].level1)

    if lt1 != lt2
        println("\n⚠️  Level types differ!")
        println("    Boolean levels: $lt1")
        println("    String levels:  $lt2")
        println("\n    The batch function may not be type-stable for boolean levels,")
        println("    causing allocations when passing levels to FC primitives.")
    else
        println("\n✓ Level types are the same")
    end
end

println("\n" * "="^80)
