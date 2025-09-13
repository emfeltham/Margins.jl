# Find remaining allocation scaling sources
using Margins, GLM, DataFrames, Tables, BenchmarkTools
include("test_utilities.jl")

println("=== Finding Remaining Allocation Sources ===\n")

function debug_population_margins_step_by_step(n_rows)
    println("Debugging with n_rows = $n_rows")
    
    # Create exact same data as allocation test
    data = make_test_data(n=n_rows)
    model = fit(LinearModel, @formula(continuous_response ~ x + y), data)
    data_nt = Tables.columntable(data)
    
    println("  Step 1: Engine creation")
    engine_alloc = @allocated engine = Margins.get_or_build_engine(model, data_nt, [:x, :y], GLM.vcov)
    println("    Engine creation: $engine_alloc bytes")
    
    println("  Step 2: _ame_continuous_and_categorical call")
    # This is what population_margins calls for effects
    rows = 1:n_rows
    
    # Warmup
    df_warmup, G_warmup = Margins._ame_continuous_and_categorical(engine, data_nt, :response, :ad, :effect)
    
    # Measure
    step2_alloc = @allocated df, G = Margins._ame_continuous_and_categorical(engine, data_nt, :response, :ad, :effect)
    println("    _ame_continuous_and_categorical: $step2_alloc bytes")
    
    println("  Step 3: Individual variable processing")
    # Test each variable individually
    for var in [:x, :y]
        # Warmup
        Margins._compute_variable_ame_unified(engine, var, rows, :response, :ad)
        
        # Measure
        var_alloc = @allocated Margins._compute_variable_ame_unified(engine, var, rows, :response, :ad)
        println("    Variable $var processing: $var_alloc bytes")
    end
    
    return (engine_alloc, step2_alloc)
end

# Test with different sizes
sizes = [100, 1000, 10000]
results = []

for n in sizes
    result = debug_population_margins_step_by_step(n)
    push!(results, (n, result...))
    println()
end

println("=== SCALING ANALYSIS ===")
println("Size | Engine Creation | Main Function")
println("-----|----------------|---------------")
for (n, engine_alloc, main_alloc) in results
    println("$n | $engine_alloc bytes | $main_alloc bytes")
end

if length(results) >= 2
    println("\n=== SCALING FACTORS ===")
    first_n, first_engine, first_main = results[1]
    last_n, last_engine, last_main = results[end]
    
    size_ratio = last_n / first_n
    engine_ratio = last_engine / first_engine if first_engine > 0 else 1
    main_ratio = last_main / first_main if first_main > 0 else 1
    
    println("Dataset size increase: $(size_ratio)x")
    println("Engine creation scaling: $(engine_ratio)x")
    println("Main function scaling: $(main_ratio)x")
    
    if main_ratio > 10
        @info "Main function exhibits significant scaling behavior"
    elseif engine_ratio > 10  
        @info "Engine creation exhibits significant scaling behavior"
    else
        @info "Both components demonstrate acceptable scaling characteristics"
    end
end