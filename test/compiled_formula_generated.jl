# compiled_formula_generated.jl
# New @generated interface

function compile_formula_generated(model)
    # Register the formula and return the Val type for @generated dispatch
    formula_val = register_formula!(model)
    
    # Get metadata for the compiled formula
    instructions, column_names, output_width = FORMULA_CACHE[typeof(formula_val).parameters[1]]
    
    return (formula_val, output_width, column_names)
end

###############################################################################
# Zero-Allocation Evaluation Interface
###############################################################################

# Step 2: Registration function
const FORMULA_CACHE = Dict{UInt64, Tuple{Vector{Instruction}, Vector{Symbol}, Int}}()

"""
    compile_to_instructions(model) -> (instructions, column_names, output_width)

Extract instruction building from model - same logic as working compile_formula
"""
function compile_to_instructions(model)
    rhs = fixed_effects_form(model).rhs
    
    instructions = Instruction[]
    column_names = Symbol[]
    position = 1
    
    # Use exact working parsing logic
    for term in rhs.terms
        position = parse_term!(instructions, column_names, term, position)
    end
    
    output_width = size(modelmatrix(model), 2)  # Keep the workaround
    
    return instructions, unique(column_names), output_width
end

"""
    register_formula!(model) -> Val{formula_hash}

Register formula in cache and return Val type for @generated dispatch
"""
function register_formula!(model)
    formula_hash = hash(string(fixed_effects_form(model).rhs))
    instructions, column_names, output_width = compile_to_instructions(model)
    FORMULA_CACHE[formula_hash] = (instructions, column_names, output_width)
    return Val(formula_hash)
end

"""
    modelrow!(row_vec, ::Val{formula_hash}, data, row_idx)

@generated function for zero-allocation formula evaluation
"""
@generated function modelrow!(row_vec, ::Val{formula_hash}, data, row_idx) where formula_hash
    instructions, column_names, output_width = FORMULA_CACHE[formula_hash]
    
    # Use exact working code generation functions
    body_lines = String[]
    for instr in instructions
        append!(body_lines, generate_instruction_code(instr))
    end
    
    code_exprs = [Meta.parse(line) for line in body_lines]
    
    return quote
        @inbounds begin
            $(code_exprs...)
        end
        return row_vec
    end
end

"""
Example usage:

```julia
using GLM, DataFrames
df = DataFrame(x = randn(1000), y = randn(1000), z = randn(1000))
model = lm(@formula(y ~ x + x^2 + log(z) + x*z), df)

formula_val, output_width, column_names = compile_formula_generated(model)
row_vec_gen = Vector{Float64}(undef, output_width);

# Single call test
@btime modelrow!(row_vec_gen, formula_val, data, 1)

# Verify correctness  
mm_row = modelmatrix(model)[1, :]
println("Correct:   ", mm_row)
println("Generated: ", row_vec_gen)
println("Match: ", isapprox(mm_row, row_vec_gen))
```

This will update `row_vec_gen` without allocating.
"""
