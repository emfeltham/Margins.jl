
using StatsModels
using CategoricalArrays: CategoricalValue, levelcode
using Margins: fixed_effects_form


import CategoricalArrays
using Revise
using Margins

begin
    using Test
    using Random
    using DataFrames, CategoricalArrays
    using Distributions, Statistics, GLM, MixedModels
    using RDatasets
    import LinearAlgebra.dot
    import LinearAlgebra.diag
    using StandardizedPredictors
end

using BenchmarkTools
using GLM, DataFrames

include("compiled_formula.jl")

n = 50_000
large_df = DataFrame(
    x = randn(n), 
    y = randn(n), 
    z = abs.(randn(n)) .+ 1,
    group = categorical(rand(["A", "B", "C"], n))
);
data = Tables.columntable(large_df);

model = lm(@formula(y ~ x + x^2 + log(z) + group), large_df);
compiled = compile_formula(model);
direct_func = get_direct_function(compiled);
mm = modelmatrix(model);

# compiled.output_width
# this is probably the number of levels too!

# Test full matrix evaluation time
row_vec = Vector{Float64}(undef, size(mm, 2))
fill!(row_vec, 0.0);

# this really is no allocations
@btime direct_func(row_vec, data, 1);  # Target: ~10ns, 0 allocations

mm1 = vec(mm[1, :])
row_vec

hcat(mm1, row_vec)

@assert isapprox(mm1, row_vec)

######

function generate_loop_function(compiled_formula, n)
    func_name = compiled_formula.function_name
    loop_name = Symbol("loop_$(compiled_formula.formula_hash)")
    
    loop_code = """
    function $loop_name(row_vec, data, n)
        @inbounds for i in 1:n
            $func_name(row_vec, data, i)
        end
    end
    """
    
    eval(Meta.parse(loop_code))
    return getproperty(Main, loop_name)
end

loop_func = generate_loop_function(compiled, n)
fill!(row_vec, 0.0)
@btime loop_func(row_vec, data, n)  # Should be 0 allocations

mm = modelmatrix(model);

@assert row_vec == mm[n, :];

### generated

# Test the @generated approach
formula_val, output_width, column_names = compile_formula_generated(model)
row_vec_gen = Vector{Float64}(undef, output_width);

# Single call test
@btime modelrow!(row_vec_gen, formula_val, data, 1)

# Verify correctness  
mm_row = modelmatrix(model)[1, :]
println("Correct:   ", mm_row)
println("Generated: ", row_vec_gen)
println("Match: ", isapprox(mm_row, row_vec_gen))



# Loop test - can use your existing loop generation but with the @generated function
function generate_loop_function_generated(formula_val, n)
    loop_name = Symbol("loop_generated_$(typeof(formula_val).parameters[1])")
    
    loop_code = """
    function $loop_name(row_vec, formula_val, data, n)
        @inbounds for i in 1:n
            ultra_fast_modelrow!(row_vec, formula_val, data, i)
        end
    end
    """
    
    eval(Meta.parse(loop_code))
    return getproperty(Main, loop_name)
end

loop_func_gen = generate_loop_function_generated(formula_val, n)
@btime loop_func_gen(row_vec_gen, formula_val, data, n)

### debug

# Test the categorical function directly
test_instr = CategoricalColumn(5, :group, [0.0 0.0; 1.0 0.0; 0.0 1.0], 2)
generated_lines = generate_instruction_code(test_instr)

println("Generated categorical lines:")
for (i, line) in enumerate(generated_lines)
    println("$i: '$line'")
end