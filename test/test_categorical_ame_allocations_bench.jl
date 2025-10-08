using Margins, GLM, DataFrames, CategoricalArrays, BenchmarkTools

println("=" ^ 60)
println("Categorical AME Allocation Benchmark")
println("=" ^ 60)

# Setup test data
n = 1000
df = DataFrame(
    y = randn(n),
    x = randn(n),
    group = categorical(repeat(["A", "B", "C"], inner=div(n,3)+1)[1:n]),
    treatment = rand([false, true], n)
)

# Test 1: Multi-level categorical (3 levels)
println("\n## Test 1: Multi-level categorical (group: A, B, C)")
println("-" ^ 60)

model1 = lm(@formula(y ~ x + group), df)
data_nt1 = Tables.columntable(df)
engine1 = Margins.build_engine(
    Margins.PopulationUsage,
    Margins.HasDerivatives,
    model1,
    data_nt1,
    [:group],
    (m) -> vcov(m),
    :ad
)
rows1 = collect(1:100)

# Warm up
for _ in 1:10
    Margins._compute_categorical_contrasts(engine1, :group, rows1, :response, :ad, :baseline)
end

# Benchmark
result1 = @benchmark Margins._compute_categorical_contrasts($engine1, :group, $rows1, :response, :ad, :baseline) samples=1000

println("Time (median):     ", median(result1.times) / 1000, " μs")
println("Time (min):        ", minimum(result1.times) / 1000, " μs")
println("Allocations:       ", result1.allocs)
println("Memory:            ", result1.memory, " bytes")

# Test 2: Boolean categorical
println("\n## Test 2: Boolean categorical (treatment: true/false)")
println("-" ^ 60)

model2 = lm(@formula(y ~ x + treatment), df)
data_nt2 = Tables.columntable(df)
engine2 = Margins.build_engine(
    Margins.PopulationUsage,
    Margins.HasDerivatives,
    model2,
    data_nt2,
    [:treatment],
    (m) -> vcov(m),
    :ad
)
rows2 = collect(1:100)

# Warm up
for _ in 1:10
    Margins._compute_categorical_contrasts(engine2, :treatment, rows2, :response, :ad, :baseline)
end

# Benchmark
result2 = @benchmark Margins._compute_categorical_contrasts($engine2, :treatment, $rows2, :response, :ad, :baseline) samples=1000

println("Time (median):     ", median(result2.times) / 1000, " μs")
println("Time (min):        ", minimum(result2.times) / 1000, " μs")
println("Allocations:       ", result2.allocs)
println("Memory:            ", result2.memory, " bytes")

# Test 3: Categorical with interaction
println("\n## Test 3: Categorical × continuous interaction (x * group)")
println("-" ^ 60)

model3 = lm(@formula(y ~ x * group), df)
data_nt3 = Tables.columntable(df)
engine3 = Margins.build_engine(
    Margins.PopulationUsage,
    Margins.HasDerivatives,
    model3,
    data_nt3,
    [:group],
    (m) -> vcov(m),
    :ad
)
rows3 = collect(1:50)

# Warm up
for _ in 1:10
    Margins._compute_categorical_contrasts(engine3, :group, rows3, :response, :ad, :baseline)
end

# Benchmark
result3 = @benchmark Margins._compute_categorical_contrasts($engine3, :group, $rows3, :response, :ad, :baseline) samples=1000

println("Time (median):     ", median(result3.times) / 1000, " μs")
println("Time (min):        ", minimum(result3.times) / 1000, " μs")
println("Allocations:       ", result3.allocs)
println("Memory:            ", result3.memory, " bytes")

# Summary
println("\n" * "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println("Test                           Allocs    Memory (bytes)")
println("-" ^ 60)
println("Multi-level categorical        ", lpad(result1.allocs, 6), "    ", lpad(result1.memory, 10))
println("Boolean categorical            ", lpad(result2.allocs, 6), "    ", lpad(result2.memory, 10))
println("Categorical × interaction      ", lpad(result3.allocs, 6), "    ", lpad(result3.memory, 10))
println("=" ^ 60)

# Check if we achieved zero allocations
if result1.allocs == 0 && result2.allocs == 0 && result3.allocs == 0
    println("\n✓ ZERO ALLOCATIONS ACHIEVED!")
else
    println("\n⚠ Non-zero allocations detected (expected during Phase 4)")
end
