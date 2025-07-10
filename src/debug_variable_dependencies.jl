# Debug script to check if we're identifying all affected columns correctly

using Margins: AMEWorkspace, InplaceModeler, modelmatrix!
using Margins: prepare_finite_differences_fixed!

"""
    debug_variable_dependencies(model, df, variable::Symbol)

Debug which columns a variable should affect in the model matrix.
"""
function debug_variable_dependencies(model, df, variable::Symbol)
    println("=== Debugging variable dependencies for :$variable ===")
    
    # Build workspace
    ws = AMEWorkspace(model, df)
    
    # Get what our selective update thinks are affected columns
    if haskey(ws.variable_plans, variable)
        affected_cols_selective = ws.variable_plans[variable]
        println("Selective update thinks $variable affects columns: $affected_cols_selective")
    else
        println("Variable $variable not found in variable_plans!")
        return
    end
    
    # Build full model matrix with original data
    X_orig = modelmatrix(model)
    
    # Build model matrix with perturbed variable
    df_pert = copy(df)
    if eltype(df[!, variable]) <: Real && eltype(df[!, variable]) != Bool
        # Continuous variable - add small perturbation
        h = std(df[!, variable]) * 1e-6
        df_pert[!, variable] = df[!, variable] .+ h
    else
        println("Variable $variable is not continuous - skipping perturbation test")
        return
    end
    
    # Build perturbed model matrix (the "ground truth")
    ipm = InplaceModeler(model, nrow(df))
    X_pert = similar(X_orig)
    modelmatrix!(ipm, Tables.columntable(df_pert), X_pert)
    
    # Find which columns actually changed
    actually_changed = Int[]
    for col in 1:size(X_orig, 2)
        if !all(X_orig[:, col] .≈ X_pert[:, col])
            push!(actually_changed, col)
        end
    end
    
    println("Ground truth: $variable actually affects columns: $actually_changed")
    
    # Compare
    missing_cols = setdiff(actually_changed, affected_cols_selective)
    extra_cols = setdiff(affected_cols_selective, actually_changed)
    
    if !isempty(missing_cols)
        println("❌ MISSING COLUMNS: $missing_cols")
        
        # Show what terms these columns correspond to
        for col in missing_cols
            term_info = get_term_for_column(ws.mapping, col)
            println("   Column $col: $term_info")
        end
    end
    
    if !isempty(extra_cols)
        println("⚠️  EXTRA COLUMNS: $extra_cols")
    end
    
    if isempty(missing_cols) && isempty(extra_cols)
        println("✅ Column dependencies look correct!")
    end
    
    return (
        selective = affected_cols_selective,
        ground_truth = actually_changed,
        missing = missing_cols,
        extra = extra_cols
    )
end

"""
    compare_finite_differences(model, df, variable::Symbol)

Compare our finite difference computation against ground truth.
"""
function compare_finite_differences(model, df, variable::Symbol)
    println("=== Comparing finite difference computations for :$variable ===")
    
    # Build workspace and step size
    ws = AMEWorkspace(model, df)
    ipm = InplaceModeler(model, nrow(df))
    
    orig_values = ws.base_data[variable]
    var_scale = std(orig_values)
    h = var_scale * 1e-6
    h = max(h, 1e-8)
    
    println("Using step size h = $h")
    
    # Our selective approach
    prepare_finite_differences_fixed!(ws, variable, h, ipm)
    our_finite_diff = copy(ws.finite_diff_matrix)
    
    # Ground truth approach - build both matrices manually
    X_base = modelmatrix(model)
    
    df_pert = copy(df)
    df_pert[!, variable] = df[!, variable] .+ h
    X_pert = similar(X_base)
    modelmatrix!(ipm, Tables.columntable(df_pert), X_pert)
    
    ground_truth_finite_diff = (X_pert - X_base) / h
    
    # Compare
    max_diff = maximum(abs.(our_finite_diff - ground_truth_finite_diff))
    affected_cols = ws.variable_plans[variable]
    
    println("Maximum difference in finite differences: $max_diff")
    
    if max_diff > 1e-10
        println("❌ Significant differences found!")
        
        # Show where the differences are
        for col in affected_cols
            col_diff = maximum(abs.(our_finite_diff[:, col] - ground_truth_finite_diff[:, col]))
            if col_diff > 1e-10
                println("   Column $col: max diff = $col_diff")
            end
        end
    else
        println("✅ Finite differences match ground truth!")
    end
    
    return (
        our_result = our_finite_diff,
        ground_truth = ground_truth_finite_diff,
        max_diff = max_diff
    )
end

"""
    prepare_finite_differences_fixed_fallback!(ws::AMEWorkspace, variable::Symbol, h::Real, 
                                     ipm::InplaceModeler)

FALLBACK VERSION: Use full matrix construction for finite differences until we fix selective updates.
"""
function prepare_finite_differences_fixed_fallback!(ws::AMEWorkspace, variable::Symbol, h::Real, 
                                          ipm::InplaceModeler)
    # Validate that variable is continuous and pre-allocated
    if !haskey(ws.pert_vectors, variable)
        throw(ArgumentError(
            "Variable $variable not found in perturbation vectors. " *
            "Only continuous (non-Bool) variables are supported."
        ))
    end
    
    # Create perturbed data
    current_var_values = ws.base_data[variable]
    pert_vector = ws.pert_vectors[variable]
    
    # Fill perturbation vector
    @inbounds for i in eachindex(pert_vector)
        pert_vector[i] = current_var_values[i] + h
    end
    
    # Create perturbed data structure
    pert_data = create_perturbed_data(ws.base_data, variable, pert_vector)
    
    # FALLBACK: Use full matrix construction for now
    # This should give us the correct finite differences
    modelmatrix!(ipm, pert_data, ws.finite_diff_matrix)
    
    # Compute finite differences: (X_perturbed - X_baseline) / h
    invh = 1.0 / h
    
    @inbounds for col in 1:size(ws.finite_diff_matrix, 2)
        for row in 1:size(ws.finite_diff_matrix, 1)
            baseline_val = ws.work_matrix[row, col]
            perturbed_val = ws.finite_diff_matrix[row, col]
            
            if !isfinite(baseline_val) || !isfinite(perturbed_val)
                ws.finite_diff_matrix[row, col] = 0.0
                continue
            end
            
            raw_diff = perturbed_val - baseline_val
            finite_diff = raw_diff * invh
            
            if isfinite(finite_diff)
                ws.finite_diff_matrix[row, col] = clamp(finite_diff, -1e6, 1e6)
            else
                ws.finite_diff_matrix[row, col] = 0.0
            end
        end
    end
    
    # Set unaffected columns to zero (this is mathematically correct)
    affected_cols = ws.variable_plans[variable]
    total_cols = size(ws.finite_diff_matrix, 2)
    unaffected_cols = setdiff(1:total_cols, affected_cols)
    
    @inbounds for col in unaffected_cols
        for row in 1:size(ws.finite_diff_matrix, 1)
            ws.finite_diff_matrix[row, col] = 0.0
        end
    end
end

function compare_finite_differences_fallback(model, df, variable::Symbol)
    println("=== Comparing finite difference computations for :$variable ===")
    
    # Build workspace and step size
    ws = AMEWorkspace(model, df)
    ipm = InplaceModeler(model, nrow(df))
    
    orig_values = ws.base_data[variable]
    var_scale = std(orig_values)
    h = var_scale * 1e-6
    h = max(h, 1e-8)
    
    println("Using step size h = $h")
    
    # Our selective approach
    prepare_finite_differences_fixed_fallback!(ws, variable, h, ipm)
    our_finite_diff = copy(ws.finite_diff_matrix)
    
    # Ground truth approach - build both matrices manually
    X_base = modelmatrix(model)
    
    df_pert = copy(df)
    df_pert[!, variable] = df[!, variable] .+ h
    X_pert = similar(X_base)
    modelmatrix!(ipm, Tables.columntable(df_pert), X_pert)
    
    ground_truth_finite_diff = (X_pert - X_base) / h
    
    # Compare
    max_diff = maximum(abs.(our_finite_diff - ground_truth_finite_diff))
    affected_cols = ws.variable_plans[variable]
    
    println("Maximum difference in finite differences: $max_diff")
    
    if max_diff > 1e-10
        println("❌ Significant differences found!")
        
        # Show where the differences are
        for col in affected_cols
            col_diff = maximum(abs.(our_finite_diff[:, col] - ground_truth_finite_diff[:, col]))
            if col_diff > 1e-10
                println("   Column $col: max diff = $col_diff")
            end
        end
    else
        println("✅ Finite differences match ground truth!")
    end
    
    return (
        our_result = our_finite_diff,
        ground_truth = ground_truth_finite_diff,
        max_diff = max_diff
    )
end

model = m;

# For the 3-way interaction test case
debug_variable_dependencies(model, df, :x)
compare_finite_differences(model, df, :x)

using Margins: create_perturbed_data, get_variable_term_ranges

# fallback version to test error
compare_finite_differences_fallback(model, df, :x)
# yes, this version is correct

# deeper diagnosis

# Diagnose the selective update bug

using Margins: modelmatrix_with_base!

"""
    diagnose_modelmatrix_with_base(model, df, variable::Symbol)

Deep dive into what's going wrong with modelmatrix_with_base!
"""
function diagnose_modelmatrix_with_base(model, df, variable::Symbol)
    println("=== Diagnosing modelmatrix_with_base! for :$variable ===")
    
    # Setup
    ws = AMEWorkspace(model, df)
    ipm = InplaceModeler(model, nrow(df))
    
    # Get step size and create perturbation
    orig_values = ws.base_data[variable]
    h = std(orig_values) * 1e-6
    h = max(h, 1e-8)
    
    pert_vector = orig_values .+ h
    pert_data = create_perturbed_data(ws.base_data, variable, pert_vector)
    
    # Ground truth: Full matrix construction
    X_ground_truth = similar(ws.base_matrix)
    modelmatrix!(ipm, pert_data, X_ground_truth)
    
    # Our approach: Selective update with base sharing
    X_selective = similar(ws.base_matrix)
    modelmatrix_with_base!(ipm, pert_data, X_selective, ws.work_matrix, [variable], ws.mapping)
    
    # Compare them
    max_diff = maximum(abs.(X_selective - X_ground_truth))
    println("Maximum difference between selective and ground truth: $max_diff")
    
    if max_diff > 1e-10
        println("❌ Significant differences found!")
        
        # Find which columns are wrong
        affected_cols = ws.variable_plans[variable]
        println("Variable $variable should affect columns: $affected_cols")
        
        for col in 1:size(X_selective, 2)
            col_diff = maximum(abs.(X_selective[:, col] - X_ground_truth[:, col]))
            if col_diff > 1e-10
                println("   Column $col: max diff = $col_diff")
                
                # Get term info for this column
                try
                    term, local_col = get_term_for_column(ws.mapping, col)
                    println("     Term: $term (local column $local_col)")
                catch e
                    println("     Could not get term info: $e")
                end
                
                # Show if this column should be affected or not
                if col in affected_cols
                    println("     ⚠️  This column SHOULD be updated by $variable")
                else
                    println("     ⚠️  This column should NOT be affected by $variable")
                end
            end
        end
    else
        println("✅ Selective update matches ground truth!")
    end
    
    return (
        ground_truth = X_ground_truth,
        selective = X_selective,
        max_diff = max_diff
    )
end

"""
    inspect_interaction_structure(model, df)

Look at the model structure to understand the interaction terms.
"""
function inspect_interaction_structure(model, df)
    println("=== Model Structure Analysis ===")
    
    # Get the formula
    f = formula(model)
    println("Formula: $f")
    
    # Build workspace to see column mapping
    ws = AMEWorkspace(model, df)
    
    println("\\nColumn mapping:")
    for (i, (term, range)) in enumerate(ws.mapping.term_info)
        println("  Columns $range: $term")
    end
    
    println("\\nVariable plans:")
    for (var, cols) in ws.variable_plans
        println("  :$var affects columns: $cols")
    end
    
    return ws.mapping
end

"""
    test_individual_terms(model, df, variable::Symbol)

Test selective update on each term individually to isolate the problem.
"""
function test_individual_terms(model, df, variable::Symbol)
    println("=== Testing Individual Terms for :$variable ===")
    
    ws = AMEWorkspace(model, df)
    ipm = InplaceModeler(model, nrow(df))
    
    # Get perturbation data
    orig_values = ws.base_data[variable]
    h = std(orig_values) * 1e-6
    pert_vector = orig_values .+ h
    pert_data = create_perturbed_data(ws.base_data, variable, pert_vector)
    
    # Test each term that involves our variable
    variable_ranges = get_variable_term_ranges(ws.mapping, variable)
    
    for (term, range) in variable_ranges
        println("\\nTesting term: $term (columns $range)")
        
        # Build matrices for just this term
        n_rows = size(ws.base_matrix, 1)
        term_width = length(range)
        
        baseline_term = Matrix{Float64}(undef, n_rows, term_width)
        selective_term = Matrix{Float64}(undef, n_rows, term_width)
        ground_truth_term = Matrix{Float64}(undef, n_rows, term_width)
        
        # Evaluate this term with different approaches
        try
            # Baseline (current data)
            evaluate_single_term!(term, ws.base_data, baseline_term, ipm)
            
            # Ground truth (full evaluation with perturbed data)
            evaluate_single_term!(term, pert_data, ground_truth_term, ipm)
            
            # Selective approach (this might be where the bug is)
            # For now, let's just use ground truth to see what it should be
            selective_term .= ground_truth_term
            
            # Compare
            term_diff = maximum(abs.(selective_term - ground_truth_term))
            println("  Term difference: $term_diff")
            
            if term_diff > 1e-10
                println("  ❌ This term has issues!")
            else
                println("  ✅ This term looks OK")
            end
            
        catch e
            println("  ❌ Error evaluating term: $e")
        end
    end
end

"""
    evaluate_single_term!(term, data, output, ipm)

Helper to evaluate a single term with given data.
"""
function evaluate_single_term!(term, data, output, ipm)
    # This is a simplified version - you might need to adapt based on your EfficientModelMatrices setup
    fn_i = Ref(1)
    int_i = Ref(1)
    _cols!(term, data, output, 1, ipm, fn_i, int_i)
end

# Run these diagnostics on your failing test case

# Assuming you have the model and df from the failing 3-way interaction test
println("Step 1: Inspect the model structure")
mapping = inspect_interaction_structure(model, df)

println("\\n" * "="^60)
println("Step 2: Diagnose modelmatrix_with_base!")
result = diagnose_modelmatrix_with_base(model, df, :x)

println("\\n" * "="^60)
println("Step 3: Test individual terms")
test_individual_terms(model, df, :x)

# Let's also check what the exact formula is
println("\\n" * "="^60)
println("Step 4: Formula analysis")
println("Model formula: ", formula(model))
println("RHS terms: ", formula(model).rhs)

# And let's see the model matrix structure
println("\\n" * "="^60)
println("Step 5: Matrix structure")
X = modelmatrix(model)
println("Model matrix size: $(size(X))")
println("First few column names (if available):")
try
    println(coefnames(model))
catch
    println("Coefficient names not available")
end