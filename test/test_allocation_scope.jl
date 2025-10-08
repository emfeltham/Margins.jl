# Investigate where allocations occur in the call stack
using Margins, GLM, DataFrames, Tables, BenchmarkTools, CategoricalArrays
using Margins: build_engine, PopulationUsage, HasDerivatives
using Margins: generate_contrast_pairs, categorical_contrast_ame_batch!

n = 100
df = DataFrame(
    y = randn(n),
    x = randn(n),
    group = categorical(rand(["Control", "Treatment"], n))
)
model = lm(@formula(y ~ x + group), df)
data_nt = Tables.columntable(df)
engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:group], GLM.vcov, :ad)

rows = collect(1:50)
var_col = getproperty(data_nt, :group)
contrast_pairs = generate_contrast_pairs(var_col, rows, :baseline, model, :group, data_nt)

println("\n" * "="^80)
println("Allocation Scope Analysis")
println("="^80)

println("\n1. Pre-allocate arrays at outer scope (correct pattern)")
println("-"^80)

# Pre-allocate once (what _process_single_categorical_variable! should do)
n_contrasts = length(contrast_pairs)
results_ame = Vector{Float64}(undef, n_contrasts)
results_se = Vector{Float64}(undef, n_contrasts)
gradient_matrix = Matrix{Float64}(undef, n_contrasts, length(coef(model)))

# Warmup
for _ in 1:10
    categorical_contrast_ame_batch!(
        results_ame, results_se, gradient_matrix,
        engine.contrast_buf, engine.contrast_grad_buf, engine.contrast_grad_accum,
        engine.contrast, :group, contrast_pairs,
        engine.β, engine.Σ, engine.link,
        rows, nothing
    )
end

# Benchmark - should be 0 alloc
b1 = @benchmark categorical_contrast_ame_batch!(
    $results_ame, $results_se, $gradient_matrix,
    $(engine.contrast_buf), $(engine.contrast_grad_buf), $(engine.contrast_grad_accum),
    $(engine.contrast), :group, $contrast_pairs,
    $(engine.β), $(engine.Σ), $(engine.link),
    $rows, nothing
) samples=100

println("Zero-allocation pattern (arrays pre-allocated):")
println("  Allocations: ", b1.allocs)
println("  Memory:      ", b1.memory, " bytes")

println("\n2. Allocate arrays inside function (current _compute_categorical_contrasts)")
println("-"^80)

function allocate_inside()
    # This is what _compute_categorical_contrasts currently does
    n_contrasts = length(contrast_pairs)
    results_ame = Vector{Float64}(undef, n_contrasts)
    results_se = Vector{Float64}(undef, n_contrasts)
    gradient_matrix = Matrix{Float64}(undef, n_contrasts, length(engine.β))

    categorical_contrast_ame_batch!(
        results_ame, results_se, gradient_matrix,
        engine.contrast_buf, engine.contrast_grad_buf, engine.contrast_grad_accum,
        engine.contrast, :group, contrast_pairs,
        engine.β, engine.Σ, engine.link,
        rows, nothing
    )

    return results_ame, results_se, gradient_matrix
end

# Warmup
for _ in 1:10
    allocate_inside()
end

b2 = @benchmark allocate_inside() samples=100

println("Current pattern (arrays allocated inside):")
println("  Allocations: ", b2.allocs)
println("  Memory:      ", b2.memory, " bytes")

println("\n3. Analysis")
println("-"^80)

println("\nAllocation breakdown:")
println("  Pre-allocated:  ", b1.allocs, " allocs (hot loop)")
println("  Inside function: ", b2.allocs, " allocs (includes array allocation)")
println("  Difference:      ", b2.allocs - b1.allocs, " allocs from array allocation")

println("\nConclusion:")
println("  Moving array allocation to caller (_process_single_categorical_variable!)")
println("  will eliminate ", b2.allocs - b1.allocs, " allocations from the hot path.")
println("\n  Cost: ", b2.allocs - b1.allocs, " allocations per VARIABLE (not per contrast)")
println("  Benefit: Hot loop remains 0-alloc")

println("\n" * "="^80)
