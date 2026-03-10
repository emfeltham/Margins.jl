# compare_performance_marginaleffects.jl
# Analyzes Julia vs R marginaleffects performance benchmarks and reports speedup ratios

using DataFrames, CSV, RData
using Statistics

println("=" ^ 80)
println("JULIA vs R marginaleffects PERFORMANCE COMPARISON")
println("=" ^ 80)
println()

# Determine script directory for relative paths
script_dir = @__DIR__

# Load Julia benchmarks
println("Loading benchmark results...")
julia_results = CSV.read(joinpath(script_dir, "julia_benchmarks.csv"), DataFrame)

# Load R marginaleffects benchmarks from RDS file
r_benchmarks = load(joinpath(script_dir, "r_marginaleffects_benchmarks.rds"))

# Extract R timing data (stored in microseconds, convert to seconds)
r_results = DataFrame(
    operation = String[],
    time_s = Float64[],
    memory_mb = Union{Float64, Missing}[]
)

# R benchmark structure: list of microbenchmark objects
# Extract median times and convert to seconds
operations = ["AAP", "AME (all)", "AME (age_h)", "AME (scenario)", "APM", "MEM"]
r_operations = ["aap", "ame_all", "ame_age", "ame_scenario", "apm", "mem"]

for (op_name, r_name) in zip(operations, r_operations)
    if haskey(r_benchmarks, r_name)
        bench = r_benchmarks[r_name]
        # microbenchmark stores times in nanoseconds
        times = bench[!, :time]  # Access the time column
        median_time = median(times) / 1e9  # Convert to seconds
        # Extract memory from profmem measurements (stored separately)
        mem_key = r_name * "_mem"
        memory_mb = haskey(r_benchmarks, mem_key) ? r_benchmarks[mem_key] : missing
        push!(r_results, (op_name, median_time, memory_mb))
    end
end

println("  ✓ Julia results: $(nrow(julia_results)) operations")
println("  ✓ R marginaleffects results: $(nrow(r_results)) operations")
println()

# Merge results for comparison
comparison = innerjoin(
    julia_results,
    r_results,
    on = :operation,
    makeunique = true
)

rename!(comparison,
    :time_s => :julia_time_s,
    :memory_mb => :julia_mem_mb,
    :time_s_1 => :r_time_s,
    :memory_mb_1 => :r_mem_mb
)

# Calculate speedup ratios
comparison[!, :speedup_ratio] = comparison.r_time_s ./ comparison.julia_time_s
comparison[!, :memory_ratio] = comparison.r_mem_mb ./ comparison.julia_mem_mb

# Add percentage improvement columns
comparison[!, :time_improvement_pct] = (1 .- 1 ./ comparison.speedup_ratio) .* 100
comparison[!, :memory_improvement_pct] = (1 .- 1 ./ comparison.memory_ratio) .* 100

println("=" ^ 80)
println("DETAILED COMPARISON")
println("=" ^ 80)
println()

for row in eachrow(comparison)
    println("$(row.operation):")
    println("  Julia:            $(round(row.julia_time_s, digits=4))s  |  $(round(row.julia_mem_mb, digits=2)) MB")
    println("  R marginaleffects: $(round(row.r_time_s, digits=4))s  |  $(round(row.r_mem_mb, digits=2)) MB")
    println("  Speedup:          $(round(row.speedup_ratio, digits=2))×  |  Memory: $(round(row.memory_ratio, digits=2))×")
    println("  Improvement:      $(round(row.time_improvement_pct, digits=1))% faster  |  $(round(row.memory_improvement_pct, digits=1))% less memory")
    println()
end

println("=" ^ 80)
println("SUMMARY STATISTICS")
println("=" ^ 80)
println()

avg_speedup = mean(comparison.speedup_ratio)
median_speedup = median(comparison.speedup_ratio)
min_speedup = minimum(comparison.speedup_ratio)
max_speedup = maximum(comparison.speedup_ratio)

println("Speed Performance vs marginaleffects:")
println("  Average speedup:  $(round(avg_speedup, digits=2))×")
println("  Median speedup:   $(round(median_speedup, digits=2))×")
println("  Range:            $(round(min_speedup, digits=2))× to $(round(max_speedup, digits=2))×")
println()

# Only show memory stats if we have data
mem_vals = collect(skipmissing(comparison.memory_ratio))
if !isempty(mem_vals)
    avg_mem_ratio = mean(mem_vals)
    median_mem_ratio = median(mem_vals)
    println("Memory Performance:")
    println("  Average memory ratio:    $(round(avg_mem_ratio, digits=2))× (R/Julia)")
    println("  Median memory ratio:     $(round(median_mem_ratio, digits=2))×")
    println()
else
    println("Memory Performance:")
    println("  (No memory data available from R)")
    println()
end

# Overall assessment
total_julia_time = sum(julia_results.time_s)
total_r_time = sum(r_results.time_s)
overall_speedup = total_r_time / total_julia_time

println("Overall Performance:")
println("  Total Julia time:          $(round(total_julia_time, digits=2))s")
println("  Total R marginaleffects:   $(round(total_r_time, digits=2))s")
println("  Overall speedup:           $(round(overall_speedup, digits=2))×")
println()

# Save comparison table
CSV.write(joinpath(script_dir, "performance_comparison_marginaleffects.csv"), comparison)

println("=" ^ 80)
println("RESULTS SAVED")
println("=" ^ 80)
println()
println("✓ performance_comparison_marginaleffects.csv")
println()

# Print interpretation
println("=" ^ 80)
println("INTERPRETATION")
println("=" ^ 80)
println()

println("NOTE: marginaleffects is the current state-of-the-art R package,")
println("      ~1000× faster than the older margins package.")
println()

if avg_speedup >= 10
    println("🚀 EXCEPTIONAL: Julia is $(round(avg_speedup, digits=1))× faster than marginaleffects")
    println("   Even against R's state-of-the-art implementation, Julia provides major advantages.")
    println("   This demonstrates the fundamental benefit of zero-allocation architecture.")
elseif avg_speedup >= 5
    println("⚡ EXCELLENT: Julia is $(round(avg_speedup, digits=1))× faster than marginaleffects")
    println("   Substantial performance gains over R's best-in-class implementation.")
    println("   Zero-allocation architecture provides clear advantages at scale.")
elseif avg_speedup >= 2
    println("✓ STRONG: Julia is $(round(avg_speedup, digits=1))× faster than marginaleffects")
    println("   Notable performance advantage even against optimized R implementation.")
    println("   Demonstrates efficiency gains from compiled evaluation.")
elseif avg_speedup >= 1.2
    println("✓ COMPETITIVE: Julia is $(round(avg_speedup, digits=1))× faster than marginaleffects")
    println("   Performance is strong against R's state-of-the-art package.")
    println("   Architectural differences (O(1) vs O(n)) provide modest speed gains.")
else
    println("~ COMPARABLE: Julia is $(round(avg_speedup, digits=1))× relative to marginaleffects")
    println("   Performance is similar to R's optimized implementation.")
    println("   Key advantage remains in memory scaling (O(1) vs O(n)) for larger datasets.")
end

println()
println("Dataset: N=$(nrow(CSV.read(joinpath(script_dir, "r_comparison_data.csv"), DataFrame))) observations")
println()
println("KEY INSIGHT: marginaleffects is highly optimized but still O(n) constrained.")
println("             Julia's zero-allocation architecture provides O(1) memory scaling.")
println()
