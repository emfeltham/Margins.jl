# Debug script to isolate population margins scaling bottleneck
using Margins, GLM, DataFrames, BenchmarkTools, Tables, FormulaCompiler

println("=== Population Margins Scaling Bottleneck Analysis ===\n")

function test_scaling_components(n_rows)
    println("Testing with n_rows = $n_rows")
    
    # Create test data
    df = DataFrame(
        y = randn(n_rows),
        x = randn(n_rows)
    )
    data_nt = Tables.columntable(df)
    model = lm(@formula(y ~ x), df)
    
    # Build engine components like population_margins does
    engine = Margins.get_or_build_engine(model, data_nt, [:x], GLM.vcov)
    rows = 1:n_rows
    
    println("  Engine built successfully")
    
    # Test 1: FormulaCompiler marginal_effects_mu! scaling
    println("  Testing FormulaCompiler.marginal_effects_mu! scaling:")
    g_buf = Vector{Float64}(undef, 1)
    
    # Warmup
    FormulaCompiler.marginal_effects_mu!(g_buf, engine.de, engine.β, 1; link=engine.link, backend=:fd)
    
    # Test single call allocation
    single_alloc = @allocated FormulaCompiler.marginal_effects_mu!(g_buf, engine.de, engine.β, 1; link=engine.link, backend=:fd)
    println("    Single call: $single_alloc bytes")
    
    # Test loop allocation (this should reveal scaling)
    function test_loop(g_buf, de, β, link, rows)
        for row in rows
            FormulaCompiler.marginal_effects_mu!(g_buf, de, β, row; link=link, backend=:fd)
        end
    end
    
    # Warmup
    test_loop(g_buf, engine.de, engine.β, engine.link, 1:min(10, n_rows))
    
    # Measure loop allocation
    loop_alloc = @allocated test_loop(g_buf, engine.de, engine.β, engine.link, 1:min(100, n_rows))
    loop_per_call = loop_alloc / min(100, n_rows)
    println("    Loop (100 calls): $loop_alloc bytes total, $loop_per_call bytes per call")
    
    # Test 2: _accumulate_weighted_ame_gradient! scaling (suspected bottleneck)
    println("  Testing _accumulate_weighted_ame_gradient! scaling:")
    
    gβ_sum = Vector{Float64}(undef, length(engine.β))
    weights = ones(Float64, n_rows)
    
    # Warmup
    Margins._accumulate_weighted_ame_gradient!(gβ_sum, engine.de, engine.β, 1:min(10, n_rows), :x, weights; 
                                               link=engine.link, backend=:fd)
    
    # Test small subset
    small_subset = 1:min(100, n_rows)
    small_alloc = @allocated Margins._accumulate_weighted_ame_gradient!(gβ_sum, engine.de, engine.β, small_subset, :x, weights; 
                                                                        link=engine.link, backend=:fd)
    println("    Small subset ($(length(small_subset)) rows): $small_alloc bytes")
    
    # Test full dataset if manageable
    if n_rows <= 1000
        full_alloc = @allocated Margins._accumulate_weighted_ame_gradient!(gβ_sum, engine.de, engine.β, rows, :x, weights; 
                                                                          link=engine.link, backend=:fd)
        println("    Full dataset ($n_rows rows): $full_alloc bytes")
        println("    Allocation ratio: $(full_alloc / small_alloc)x")
    else
        println("    Skipping full dataset test (too large)")
    end
    
    # Test 3: _compute_continuous_ame scaling (next level up)
    println("  Testing _compute_continuous_ame scaling:")
    
    if n_rows <= 1000
        continuous_alloc = @allocated Margins._compute_continuous_ame(engine, :x, rows, :response, :fd)
        println("    Full _compute_continuous_ame ($n_rows rows): $continuous_alloc bytes")
    else
        println("    Skipping _compute_continuous_ame test (too large)")
    end
    
    println("  ---")
    return (single_alloc, loop_per_call, small_alloc)
end

# Test different dataset sizes
sizes = [100, 500, 1000, 2000]
results = []

for size in sizes
    result = test_scaling_components(size)
    push!(results, (size, result...))
    println()
end

println("=== SCALING ANALYSIS SUMMARY ===")
println("Size | Single Call | Loop Per Call | Gradient Accumulation (100 rows)")
println("-----|-------------|---------------|----------------------------------")
for (size, single, loop, grad) in results
    println("$size | $single bytes | $loop bytes | $grad bytes")
end

# Calculate scaling factors
if length(results) >= 2
    println("\n=== SCALING FACTORS ===")
    first_size, first_results = results[1][1], results[1][2:end]
    last_size, last_results = results[end][1], results[end][2:end]
    
    size_ratio = last_size / first_size
    println("Dataset size increase: $(size_ratio)x")
    
    for (i, name) in enumerate(["Single Call", "Loop Per Call", "Gradient Accumulation"])
        if first_results[i] > 0 && last_results[i] > 0
            alloc_ratio = last_results[i] / first_results[i]
            println("$name allocation increase: $(alloc_ratio)x")
        end
    end
end