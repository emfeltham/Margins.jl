# Allocation scaling validation for marginal effects computation
using Test, BenchmarkTools, Margins, DataFrames, GLM, StatsModels
include("test_utilities.jl")  # This provides make_test_data

@info "Allocation Scaling Validation (Methodological Replication)"

# Sample sizes replicate original benchmark methodology
dataset_sizes = [100, 1000, 10000]

results = []
for n_rows in dataset_sizes
    @info "Evaluating sample size n = $n_rows"
    
    # Data generation methodology consistent with original validation
    data = make_test_data(n=n_rows)
    model = fit(LinearModel, @formula(continuous_response ~ x + y), data)
    
    # Compilation overhead elimination
    result_warmup = population_margins(model, data; type=:effects, vars=[:x, :y])
    
    # Performance measurement with methodological consistency
    bench = @benchmark population_margins($model, $data; type=:effects, vars=[:x, :y]) samples=10 evals=1
    
    min_allocs = minimum(bench).allocs
    allocs_per_row = min_allocs / n_rows
    time_ms = minimum(bench).time / 1e6
    
    push!(results, (n_rows, min_allocs, allocs_per_row, time_ms))
    @info "Memory allocation: $min_allocs bytes ($allocs_per_row per observation)"
    @info "Execution time: $(round(time_ms, digits=2)) ms"
end

# Scaling analysis following original validation methodology
if length(results) >= 2
    @info "Computational Scaling Analysis:"
    first_result = results[1]
    last_result = results[end]
    
    first_n, first_allocs = first_result[1], first_result[2]
    last_n, last_allocs = last_result[1], last_result[2]
    
    size_ratio = last_n / first_n
    alloc_ratio = last_allocs / first_allocs
    
    @info "Dataset size scaling factor: $(size_ratio)x"
    @info "Memory allocation scaling factor: $(alloc_ratio)x"
    
    # Validation criterion consistent with original methodology
    test_condition = alloc_ratio < 100
    @info "Scaling criterion satisfaction (allocation ratio < 100): $test_condition"
    
    if test_condition
        @info "Validation successful: Memory allocation scaling within acceptable bounds"
    else
        @info "Validation failed: Memory allocation scaling exceeds acceptable thresholds"
        
        # Detailed diagnostic information
        @info "Comprehensive scaling diagnostics:"
        for (i, (n, allocs, per_row, time)) in enumerate(results)
            @info "Configuration $i: n=$n, allocations=$allocs, per-observation=$per_row"
        end
    end
end