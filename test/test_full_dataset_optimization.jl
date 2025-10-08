using Margins, GLM, DataFrames, CategoricalArrays, Tables

println("Testing FULL DATASET optimization (no per-row allocations)")
println("=" ^ 60)

function test_full_dataset(n)
    df = DataFrame(
        y = randn(n),
        x1 = randn(n),
        x2 = randn(n),
        group = categorical(rand(["A", "B", "C", "D"], n)),
        treatment = rand([false, true], n)
    )

    model = lm(@formula(y ~ x1 * group + x2 * treatment), df)

    # Warmup
    population_margins(model, df; type=:effects)

    # Test
    stats = @timed population_margins(model, df; type=:effects)
    return (n, stats.time * 1000, stats.bytes / 1024)
end

# Test different sizes
println("\nTesting with FULL dataset (should have O(1) allocations):")
for n in [100, 1000, 10000]
    n_rows, time_ms, alloc_kb = test_full_dataset(n)
    println("n=$n_rows: $(round(time_ms, digits=2))ms, $(round(alloc_kb, digits=1))KB")
    println("  Per-row: $(round(alloc_kb * 1024 / n_rows, digits=1)) bytes/row")
end

println("\n" * "=" ^ 60)
println("EXPECTED BEHAVIOR:")
println("- Allocations should NOT scale linearly with n")
println("- Per-row bytes should DECREASE as n increases")
println("- This proves O(1) allocation complexity")

# Now test with the internal function directly to isolate the issue
println("\nDirect test of _compute_categorical_contrasts:")

n = 10000
df = DataFrame(
    y = randn(n),
    x = randn(n),
    group = categorical(rand(["A", "B", "C"], n))
)

model = lm(@formula(y ~ x * group), df)
data_nt = Tables.columntable(df)

engine = Margins.build_engine(
    Margins.PopulationUsage,
    Margins.HasDerivatives,
    model,
    data_nt,
    [:x, :group],
    vcov
)

# Test with FULL rows (should use optimization)
rows_all = 1:n
Margins._compute_categorical_contrasts(engine, :group, rows_all, :response, :fd, :baseline)  # warmup
stats = @timed Margins._compute_categorical_contrasts(engine, :group, rows_all, :response, :fd, :baseline)
println("\nFull dataset (1:$n): $(round(stats.time*1000, digits=2))ms, $(round(stats.bytes/1024, digits=1))KB")
println("  Per-row: $(round(stats.bytes/n, digits=1)) bytes/row")

# Test with subset (can't use optimization)
rows_subset = collect(1:div(n,2))
stats = @timed Margins._compute_categorical_contrasts(engine, :group, rows_subset, :response, :fd, :baseline)
println("\nSubset (1:$(div(n,2))): $(round(stats.time*1000, digits=2))ms, $(round(stats.bytes/1024, digits=1))KB")
println("  Per-row: $(round(stats.bytes/(div(n,2)), digits=1)) bytes/row")