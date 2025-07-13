
###############################################################################
# Core Types
###############################################################################

"""
Instruction types for formula compilation
"""
abstract type Instruction end

struct BinaryOp <: Instruction
    position::Int
    op::Function
    left_source::Union{Int, Symbol}
    right_source::Union{Int, Symbol}
end

struct UnaryOp <: Instruction
    position::Int
    op::Function
    source::Union{Int, Symbol}
end

struct SetConstant <: Instruction
    position::Int
    value::Float64
end

struct CopyColumn <: Instruction
    position::Int
    column::Symbol
end

struct PowerColumn <: Instruction
    position::Int
    column::Symbol
    exponent::Float64
end

struct FunctionColumn <: Instruction
    position::Int
    func::Function
    column::Symbol
end

struct ProductColumns <: Instruction
    position::Int
    columns::Vector{Symbol}
end

struct CategoricalColumn <: Instruction
    position::Int
    column::Symbol
    contrast_matrix::Matrix{Float64}
    num_cols::Int
end

"""
Compiled formula representation - no function storage
"""
struct CompiledFormula
    function_name::Symbol  # Name of generated function
    instructions::Vector{Instruction}  # For debugging
    output_width::Int
    column_names::Vector{Symbol}
    formula_hash::UInt64  # For identification
end

###############################################################################
# Formula Parsing - Convert StatsModels terms to instructions
###############################################################################

"""
    compile_formula(model) -> CompiledFormula

Main entry point: convert model formula to compiled evaluator
"""
function compile_formula(model)
    rhs = fixed_effects_form(model).rhs  # Get right-hand side terms
    
    instructions = Instruction[]
    column_names = Symbol[]
    position = 1
    
    # Parse each term into instructions
    for term in rhs.terms
        position = parse_term!(instructions, column_names, term, position)
    end
    
    # output_width = position - 1
    # TEMP workaround
    output_width = size(modelmatrix(model), 2)
    
    # Create unique hash for this formula
    formula_hash = hash(string(rhs))
    
    # Generate and eval specialized function
    function_name = generate_function(instructions, output_width, formula_hash)
    
    return CompiledFormula(function_name, instructions, output_width, unique(column_names), formula_hash)
end

"""
Parse individual terms into instructions
"""
function parse_term!(instructions, column_names, term, position)
    if term isa InterceptTerm
        if hasintercept(term)  # InterceptTerm{true}
            push!(instructions, SetConstant(position, 1.0))
            return position + 1
        else
            return position  # InterceptTerm{false} - no output
        end
        
    elseif term isa ConstantTerm
        push!(instructions, SetConstant(position, Float64(term.n)))
        return position + 1
        
    elseif term isa Union{ContinuousTerm, Term}
        push!(column_names, term.sym)
        push!(instructions, CopyColumn(position, term.sym))
        return position + 1
        
    elseif term isa FunctionTerm
        return parse_function_term!(instructions, column_names, term, position)
        
    elseif term isa InteractionTerm
        return parse_interaction_term!(instructions, column_names, term, position)
        
    elseif term isa CategoricalTerm
        return parse_categorical_term!(instructions, column_names, term, position)
        
    else
        error("Unsupported term type: $(typeof(term))")
    end
end

function parse_function_term!(instructions, column_names, term, position)
    func = term.f
    args = term.args
    
    if length(args) == 1
        # Single argument function: f(x) or f(expression)
        if args[1] isa Union{ContinuousTerm, Term}
            # Simple case: f(column)
            column = args[1].sym
            push!(column_names, column)
            push!(instructions, FunctionColumn(position, func, column))
            return position + 1
        else
            # Complex case: f(expression) - need to parse recursively
            arg_pos = parse_expression!(instructions, column_names, args[1], position)
            push!(instructions, UnaryOp(position + arg_pos, func, arg_pos))
            return position + arg_pos + 1
        end
        
    elseif length(args) == 2
        # Binary operation: x^2, x+y, etc.
        if func === (^) && args[2] isa ConstantTerm
            # Power function: x^2
            column = args[1].sym
            exponent = Float64(args[2].n)
            push!(column_names, column)
            push!(instructions, PowerColumn(position, column, exponent))
            return position + 1
        else
            # General binary operation: x+y, x*y, etc.
            left_pos = parse_expression!(instructions, column_names, args[1], position)
            right_pos = parse_expression!(instructions, column_names, args[2], position + left_pos)
            push!(instructions, BinaryOp(position + left_pos + right_pos, func, left_pos, right_pos))
            return position + left_pos + right_pos + 1
        end
        
    else
        error("Functions with $(length(args)) arguments not yet supported: $term")
    end
end

"""
Parse any expression (term, constant, or nested function) into instructions
"""
function parse_expression!(instructions, column_names, expr, start_position)
    if expr isa Union{ContinuousTerm, Term}
        # Simple column reference
        push!(column_names, expr.sym)
        push!(instructions, CopyColumn(start_position, expr.sym))
        return 1  # Used 1 position
        
    elseif expr isa ConstantTerm
        # Constant value
        push!(instructions, SetConstant(start_position, Float64(expr.n)))
        return 1  # Used 1 position
        
    elseif expr isa FunctionTerm
        # Recursive function call
        return parse_function_term!(instructions, column_names, expr, start_position)
        
    elseif expr isa InteractionTerm
        # Interaction term
        columns = Symbol[]
        for component in expr.terms
            if component isa Union{ContinuousTerm, Term}
                push!(columns, component.sym)
                push!(column_names, component.sym)
            else
                error("Complex interactions not yet supported: $expr")
            end
        end
        push!(instructions, ProductColumns(start_position, columns))
        return 1  # Used 1 position
        
    else
        error("Unsupported expression type: $(typeof(expr))")
    end
end

function parse_interaction_term!(instructions, column_names, term, position)
    # Simple case: product of continuous variables
    columns = Symbol[]
    
    for component in term.terms
        if component isa Union{ContinuousTerm, Term}
            push!(columns, component.sym)
            push!(column_names, component.sym)
        else
            error("Complex interactions not yet supported: $term")
        end
    end
    
    push!(instructions, ProductColumns(position, columns))
    return 1  # Used 1 position
end

function parse_categorical_term!(instructions, column_names, term, position)
    push!(column_names, term.sym)
    
    # Extract contrast matrix
    contrast_matrix = Matrix{Float64}(term.contrasts.matrix)
    num_cols = size(contrast_matrix, 2)
    
    push!(instructions, CategoricalColumn(position, term.sym, contrast_matrix, num_cols))
    return num_cols  # Used num_cols positions
end

###############################################################################
# Code Generation - Convert instructions to specialized function
###############################################################################

"""
Generate and eval specialized function directly into global namespace
"""
function generate_function(instructions, output_width, formula_hash)
    # Build function body as string
    body_lines = String[]
    
    for instr in instructions
        append!(body_lines, generate_instruction_code(instr))
    end
    
    # Create unique function name based on formula
    func_name = Symbol("compiled_formula_$(abs(formula_hash))")
    
    # Combine into complete function
    func_code = """
    function $func_name(row_vec, data, row_idx)
        $(join(body_lines, "\n        "))
        return row_vec
    end
    """
    
    # Eval directly into Main namespace
    eval(Meta.parse(func_code))
    
    # Return the function name (symbol) not the function itself
    return func_name
end

"""
Generate code lines for each instruction type
"""
function generate_instruction_code(instr::SetConstant)
    return ["@inbounds row_vec[$(instr.position)] = $(instr.value)"]
end

function generate_instruction_code(instr::CopyColumn)
    return ["@inbounds row_vec[$(instr.position)] = Float64(data.$(instr.column)[row_idx])"]
end

function generate_instruction_code(instr::PowerColumn)
    col = instr.column
    pos = instr.position
    exp = instr.exponent
    
    if exp == 2.0
        return [
            "@inbounds val_$(col) = Float64(data.$(col)[row_idx])",
            "@inbounds row_vec[$pos] = val_$(col) * val_$(col)"
        ]
    elseif exp == 3.0
        return [
            "@inbounds val_$(col) = Float64(data.$(col)[row_idx])",
            "@inbounds row_vec[$pos] = val_$(col) * val_$(col) * val_$(col)"
        ]
    else
        return [
            "@inbounds val_$(col) = Float64(data.$(col)[row_idx])",
            "@inbounds row_vec[$pos] = val_$(col)^$exp"
        ]
    end
end

function generate_instruction_code(instr::FunctionColumn)
    col = instr.column
    func_name = instr.func
    pos = instr.position
    
    return [
        "@inbounds val_$(col) = Float64(data.$(col)[row_idx])",
        "@inbounds row_vec[$pos] = $func_name(val_$(col))"
    ]
end

function generate_instruction_code(instr::ProductColumns)
    lines = String[]
    pos = instr.position
    
    # Generate value extraction for each column
    for col in instr.columns
        push!(lines, "@inbounds val_$(col) = Float64(data.$(col)[row_idx])")
    end
    
    # Generate product
    product_expr = join(["val_$col" for col in instr.columns], " * ")
    push!(lines, "@inbounds row_vec[$pos] = $product_expr")
    
    return lines
end

function generate_instruction_code(instr::BinaryOp)
    lines = String[]
    pos = instr.position
    op_symbol = instr.op
    
    # Get left operand
    if instr.left_source isa Symbol
        push!(lines, "@inbounds left_val = Float64(data.$(instr.left_source)[row_idx])")
    else
        push!(lines, "@inbounds left_val = row_vec[$(instr.left_source)]")
    end
    
    # Get right operand  
    if instr.right_source isa Symbol
        push!(lines, "@inbounds right_val = Float64(data.$(instr.right_source)[row_idx])")
    else
        push!(lines, "@inbounds right_val = row_vec[$(instr.right_source)]")
    end
    
    # Apply operation with safe handling
    if op_symbol === (+)
        push!(lines, "@inbounds row_vec[$pos] = left_val + right_val")
    elseif op_symbol === (-)
        push!(lines, "@inbounds row_vec[$pos] = left_val - right_val")
    elseif op_symbol === (*)
        push!(lines, "@inbounds row_vec[$pos] = left_val * right_val")
    elseif op_symbol === (/)
        push!(lines, "@inbounds row_vec[$pos] = right_val != 0.0 ? left_val / right_val : left_val")
    elseif op_symbol === (^)
        push!(lines, "@inbounds row_vec[$pos] = left_val^right_val")
    else
        push!(lines, "@inbounds row_vec[$pos] = $op_symbol(left_val, right_val)")
    end
    
    return lines
end

function generate_instruction_code(instr::UnaryOp)
    lines = String[]
    pos = instr.position
    op_symbol = instr.op
    
    # Get operand
    if instr.source isa Symbol
        push!(lines, "@inbounds val = Float64(data.$(instr.source)[row_idx])")
    else
        push!(lines, "@inbounds val = row_vec[$(instr.source)]")
    end
    
    # Apply operation with safe handling
    if op_symbol === log
        push!(lines, "@inbounds row_vec[$pos] = val > 0.0 ? log(val) : log(abs(val) + 1e-16)")
    elseif op_symbol === sqrt
        push!(lines, "@inbounds row_vec[$pos] = sqrt(abs(val))")
    else
        push!(lines, "@inbounds row_vec[$pos] = $op_symbol(val)")
    end
    
    return lines
end

function generate_instruction_code(instr)
    # This stores intermediate values for complex expressions
    lines = String[]
    
    if instr.value_source isa Symbol
        push!(lines, "@inbounds temp_$(instr.temp_id) = Float64(data.$(instr.value_source)[row_idx])")
    else
        push!(lines, "@inbounds temp_$(instr.temp_id) = row_vec[$(instr.value_source)]")
    end
    
    return lines
end

###############################################################################

function generate_instruction_code(instr::CategoricalColumn)
    lines = String[]
    col = instr.column
    pos = instr.position
    
    # Store contrast matrix as a local constant (embed in generated function)
    matrix_var = "contrast_$(col)_matrix"
    push!(lines, "@inbounds cat_val = data.$(col)[row_idx]")
    push!(lines, "@inbounds level_code = cat_val isa CategoricalValue ? levelcode(cat_val) : 1")
    
    # Embed matrix values directly to avoid runtime lookups
    for j in 1:instr.num_cols
        for i in 1:size(instr.contrast_matrix, 1)
            matrix_val = instr.contrast_matrix[i, j]
            if i == 1
                push!(lines, "@inbounds row_vec[$(pos + j - 1)] = level_code == $i ? $matrix_val :")
            elseif i == size(instr.contrast_matrix, 1)
                push!(lines, "                                        $matrix_val")
            else
                push!(lines, "                                        level_code == $i ? $matrix_val :")
            end
        end
    end
    
    return lines
end

# Correct, but allocates
# function generate_instruction_code(instr::CategoricalColumn)
#     lines = String[]
#     col = instr.column
#     pos = instr.position
    
#     push!(lines, "@inbounds cat_val = data.$(col)[row_idx]")
#     push!(lines, "@inbounds level_code = cat_val isa CategoricalValue ? levelcode(cat_val) : 1")
    
#     # Simple array indexing instead of ternary
#     for j in 1:instr.num_cols
#         values = [instr.contrast_matrix[i, j] for i in 1:size(instr.contrast_matrix, 1)]
#         values_str = "[" * join(string.(values), ", ") * "]"
#         push!(lines, "@inbounds row_vec[$(pos + j - 1)] = $values_str[level_code]")
#     end
    
#     return lines
# end


# compiled_formula.jl - Direct code generation for zero-allocation formula evaluation

###############################################################################
# Zero-Allocation Evaluation Interface
###############################################################################

"""
    zero_alloc_modelrow!(row_vec, compiled_formula, data, row_idx)

Ultra-fast zero-allocation row vector evaluation using direct function call
"""
function zero_alloc_modelrow!(row_vec, compiled_formula::CompiledFormula, data, row_idx)
    # Call the generated function directly by name
    func = getproperty(Main, compiled_formula.function_name)
    return func(row_vec, data, row_idx)
end

"""
    get_direct_function(compiled_formula) -> Function

Get the compiled function for direct calling (even faster)
"""
function get_direct_function(compiled_formula::CompiledFormula)
    return getproperty(Main, compiled_formula.function_name)
end

###############################################################################
# Usage Example and Testing
###############################################################################

"""
Example usage:

```julia
using GLM, DataFrames
df = DataFrame(x = randn(1000), y = randn(1000), z = randn(1000))
model = lm(@formula(y ~ x + x^2 + log(z) + x*z), df)

# One-time compilation (expensive)
compiled = compile_formula(model)

# Setup for fast evaluation
row_vec = Vector{Float64}(undef, compiled.output_width)
data = Tables.columntable(df)

# Method 1: Through wrapper (still fast)
zero_alloc_modelrow!(row_vec, compiled, data, 1)

# Method 2: Direct function call (fastest)
direct_func = get_direct_function(compiled)
direct_func(row_vec, data, 1)  # Target: ~10ns, 0 allocations

# Method 3: Call by name directly (also fastest)
compiled_formula_123456789(row_vec, data, 1)  # Where 123456789 is the hash
```

The generated function will be named something like `compiled_formula_123456789`
based on the hash of your formula, and you can call it directly for maximum speed.
"""

export CompiledFormula, compile_formula, zero_alloc_modelrow!
