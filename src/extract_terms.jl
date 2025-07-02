# extract_terms.jl
# seemingly correct functions to extract the columns of the model matrix
# for an input variable.
# (potential: work towards a faster version of Margins.jl that parses the 
#  model matrix)
# the functions pass a bunch of tests

using StatsModels, DataFrames

function extractcols(terms, var_symbol::Symbol)
    indices = Int[]
    for (i, t) in enumerate(terms)
        term_variables = StatsModels.termvars(t)
        if var_symbol in term_variables
            push!(indices, i)
        end
    end
    return indices
end

# Method for raw formula RHS (needs collect)
function extractcols(formula_rhs::Union{AbstractTerm, Tuple}, var_symbol::Symbol)
    extractcols(collect(formula_rhs), var_symbol)
end

# Method for MatrixTerm (from fitted model)
function extractcols(matrix_term::MatrixTerm, var_symbol::Symbol)
    extractcols(matrix_term.terms, var_symbol)
end

# Method for fitted model directly
function extractcols(model::StatsModels.TableRegressionModel, var_symbol::Symbol)
    extractcols(formula(model).rhs, var_symbol)
end

## example

# df = DataFrame(x = 1:10, y = rand(10), z = repeat(["A", "B"], 5))
# f = @formula(y ~ x + z + x&z + log(x))

# m_test = lm(f, df);
# f_rhs = f.rhs;

# # Find which model matrix columns involve :x
# x_indices = extractcols(f_rhs, :x)
# y_indices = extractcols(f_rhs, :z)

# tt = formula(m_test).rhs.terms;

# extractcols(f_rhs, :x)
# extractcols(tt, :x)

using Test
using StatsModels, DataFrames, GLM

# Include the extractcols function
function extractcols(terms, var_symbol::Symbol)
    indices = Int[]
    for (i, t) in enumerate(terms)
        term_variables = StatsModels.termvars(t)
        if var_symbol in term_variables
            push!(indices, i)
        end
    end
    return indices
end

# Method for raw formula RHS (needs collect)
function extractcols(formula_rhs::Union{AbstractTerm, Tuple}, var_symbol::Symbol)
    extractcols(collect(formula_rhs), var_symbol)
end

# Method for MatrixTerm (from fitted model)
function extractcols(matrix_term::MatrixTerm, var_symbol::Symbol)
    extractcols(matrix_term.terms, var_symbol)
end

# Method for fitted model directly
function extractcols(model::StatsModels.TableRegressionModel, var_symbol::Symbol)
    extractcols(formula(model).rhs, var_symbol)
end
    
# Create test data with various types
n = 100
df = DataFrame(
    x1 = randn(n),
    x2 = randn(n),
    x3 = randn(n),
    x4 = randn(n),
    x5 = randn(n),
    cat1 = repeat(["A", "B", "C", "D", "E"], inner=div(n,2))[1:n],
    cat2 = repeat(["X", "Y"], inner=div(n,2))[1:n],
    y = randn(n)
);

@testset "Basic Terms" begin
    f = @formula(y ~ x1 + x2 + cat1)
    model = lm(f, df)
    
    # Test with fitted model - focus on the processed formula
    x1_indices = extractcols(formula(model).rhs, :x1)
    @test !isempty(x1_indices)  # Should find x1
    @test length(x1_indices) == 1  # Should appear once (main effect only)
    
    x2_indices = extractcols(formula(model).rhs, :x2)
    @test !isempty(x2_indices)  # Should find x2
    @test length(x2_indices) == 1  # Should appear once (main effect only)
    
    cat1_indices = extractcols(formula(model).rhs, :cat1)
    @test !isempty(cat1_indices)  # Should find cat1
    @test length(cat1_indices) == 1  # Should appear once (main effect only)
    
    # Test variable not in formula
    x3_indices = extractcols(formula(model).rhs, :x3)
    @test isempty(x3_indices)
    
    # Test model method gives same result as formula(model).rhs
    @test extractcols(model, :x1) == x1_indices
end

@testset "Two-way Interactions" begin
    f = @formula(y ~ x1 + x2 + x1&x2)
    model = lm(f, df)
    
    # x1 should appear in main effect and interaction
    x1_indices = extractcols(formula(model).rhs, :x1)
    @test length(x1_indices) >= 2  # Should appear in at least 2 terms
    
    # x2 should appear in main effect and interaction
    x2_indices = extractcols(formula(model).rhs, :x2)
    @test length(x2_indices) >= 2  # Should appear in at least 2 terms
    
    # Test model method
    @test extractcols(model, :x1) == x1_indices
end

@testset "Three-way and Higher Interactions" begin
    f = @formula(y ~ x1 + x2 + x3 + x1&x2&x3)
    model = lm(f, df)
    
    # Each variable should appear in main effect and 3-way interaction
    x1_indices = extractcols(formula(model).rhs, :x1)
    @test length(x1_indices) >= 2  # Should appear in at least 2 terms
    
    x2_indices = extractcols(formula(model).rhs, :x2)
    @test length(x2_indices) >= 2  # Should appear in at least 2 terms
    
    x3_indices = extractcols(formula(model).rhs, :x3)
    @test length(x3_indices) >= 2  # Should appear in at least 2 terms
    
    # Test 4-way interaction
    f4 = @formula(y ~ x1 + x2 + x3 + x4 + x1&x2&x3&x4)
    model4 = lm(f4, df)
    
    x1_indices_4way = extractcols(formula(model4).rhs, :x1)
    @test length(x1_indices_4way) >= 2  # Should appear in at least 2 terms
end

@testset "Function Transformations" begin
    f = @formula(y ~ x1 + (x2)^3 + inv(x3))
    model = lm(f, df)
    
    # Test variables appear in their transformed terms
    x1_indices = extractcols(formula(model).rhs, :x1)
    @test !isempty(x1_indices)  # Should find x1
    @test length(x1_indices) == 1  # Should appear once (main effect only)
    
    x2_indices = extractcols(formula(model).rhs, :x2)
    @test !isempty(x2_indices)  # Should find x2 in log transformation
    @test length(x2_indices) == 1  # Should appear once
    
    x3_indices = extractcols(formula(model).rhs, :x3)
    @test !isempty(x3_indices)  # Should find x3 in sqrt transformation
    @test length(x3_indices) == 1  # Should appear once
    
    # Test model method
    @test extractcols(model, :x2) == x2_indices
end

@testset "Complex Function Interactions" begin
    f = @formula(y ~ (x1)^2 + x2 + inv(x1)&x2)
    model = lm(f, df)
    
    # x1 should appear in log(x1) main effect and interaction
    x1_indices = extractcols(f.rhs, :x1)
    @test sort(x1_indices) == [1, 3]
    
    # x2 should appear in main effect and interaction
    x2_indices = extractcols(f.rhs, :x2)
    @test sort(x2_indices) == [2, 3]
    
    # Test with model (should find variables but indices may differ)
    model_x1_indices = extractcols(model, :x1)
    @test !isempty(model_x1_indices)  # Should find x1 somewhere
    @test length(model_x1_indices) >= 2  # Should appear in multiple terms
end

@testset "Nested Function Calls" begin
    f = @formula(y ~ exp(inv(x1)) + sin(cos(x2)))
    model = lm(f, df)
    
    # x1 should be found in nested exp(log(x1))
    x1_indices = extractcols(formula(model).rhs, :x1)
    @test !isempty(x1_indices)  # Should find x1
    @test length(x1_indices) == 1  # Should appear once
    
    # x2 should be found in nested sin(cos(x2))
    x2_indices = extractcols(formula(model).rhs, :x2)
    @test !isempty(x2_indices)  # Should find x2
    @test length(x2_indices) == 1  # Should appear once
    
    # Test model method
    @test extractcols(model, :x1) == x1_indices
end

# this test is kinda irrelvant
@testset "Functions with Multiple Arguments" begin
    # Test functions that take multiple arguments
    f = @formula(y ~ x1 + x2 + (x1 + x2))
    model = lm(f, df)
    
    # Both x1 and x2 should appear in multiple terms
    x1_indices = extractcols(formula(model).rhs, :x1)
    @test length(x1_indices) >= 1  # Should appear in at least 2 terms
    
    x2_indices = extractcols(formula(model).rhs, :x2)
    @test length(x2_indices) >= 1  # Should appear in at least 2 terms
end

@testset "Mixed Categorical and Continuous Interactions" begin
    f = @formula(y ~ x1 + cat1 + x1&cat1 + inv(x2)&cat2)
    model = lm(f, df)
    
    # x1 should appear in main effect and interaction with cat1
    x1_indices = extractcols(formula(model).rhs, :x1)
    @test length(x1_indices) >= 2  # Should appear in at least 2 terms
    
    # cat1 should appear in main effect and interaction with x1
    cat1_indices = extractcols(formula(model).rhs, :cat1)
    @test length(cat1_indices) >= 2  # Should appear in at least 2 terms
    
    # x2 should appear in function interaction with cat2
    x2_indices = extractcols(formula(model).rhs, :x2)
    @test !isempty(x2_indices)  # Should find x2
    
    # cat2 should appear in interaction with log(x2)
    cat2_indices = extractcols(formula(model).rhs, :cat2)
    @test !isempty(cat2_indices)  # Should find cat2
end

@testset "Complex Multi-way Interactions with Functions" begin
    f = @formula(y ~ (x1)^2 + inv(x2) + exp(x3) + (x1)^2&inv(x2)&exp(x3))
    model = lm(f, df)
    
    # Each variable should appear in its function form and 3-way interaction
    x1_indices = extractcols(formula(model).rhs, :x1)
    @test length(x1_indices) >= 2  # Should appear in at least 2 terms
    
    x2_indices = extractcols(formula(model).rhs, :x2)
    @test length(x2_indices) >= 2  # Should appear in at least 2 terms
    
    x3_indices = extractcols(formula(model).rhs, :x3)
    @test length(x3_indices) >= 2  # Should appear in at least 2 terms
end

@testset "Very Complex Formula" begin
    # Kitchen sink formula with everything
    f = @formula(
        y ~ x1 + x2 + x3 + cat1 + cat2 + 
        inv(x1) + (x2^2) + 
        x1&x2 + x1&cat1 + cat1&cat2 +
        inv(x1)&(x2^2) + 
        x1&x2&x3 +
        inv(x1)&cat1&x2
    )
    
    model = lm(f, df)
    
    # x1 should appear in many terms
    x1_indices = extractcols(formula(model).rhs, :x1)
    @test length(x1_indices) >= 6  # Should appear in at least 6 terms
    
    # x2 should also appear in many terms
    x2_indices = extractcols(formula(model).rhs, :x2)
    @test length(x2_indices) >= 5  # Should appear in at least 5 terms
    
    # Test that a variable not in the formula returns empty
    x5_indices = extractcols(formula(model).rhs, :x5)
    @test isempty(x5_indices)
end

@testset "Edge Cases" begin
    # Formula with only intercept (no RHS terms)
    f_empty = @formula(y ~ 1)
    model_empty = lm(f_empty, df)
    
    @test isempty(extractcols(formula(model_empty).rhs, :x1))
    @test isempty(extractcols(model_empty, :x1))
    
    # Single term formula
    f_single = @formula(y ~ x1)
    model_single = lm(f_single, df)
    
    @test !isempty(extractcols(formula(model_single).rhs, :x1))
    @test isempty(extractcols(formula(model_single).rhs, :x2))
end

@testset "Input Type Consistency" begin
    # Test that all input methods give same results
    f = @formula(y ~ x1 + (x2)^4 + x1&x2)
    model = lm(f, df)
    
    # Test that model methods are consistent with each other
    f = @formula(y ~ x1 + inv(x2) + x1&x2)
    model = lm(f, df)
    
    for var in [:x1, :x2]
        result_model = extractcols(model, var)
        result_formula = extractcols(formula(model).rhs, var)
        
        # Both model methods should give same result
        @test result_model == result_formula
        @test !isempty(result_model)  # Should find the variable
    end
end
