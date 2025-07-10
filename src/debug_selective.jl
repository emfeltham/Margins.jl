# Detailed debugging tools for selective update investigation

using Margins.EfficientModelMatrices
using EfficientModelMatrices: _cols!

"""
    debug_selective_vs_full_evaluation(model, df, variable::Symbol)

Trace exactly where selective evaluation diverges from full evaluation.
"""
function debug_selective_vs_full_evaluation(model, df, variable::Symbol)
    println("=== DEBUGGING SELECTIVE UPDATE FOR :$variable ===")
    
    # Setup
    ws = AMEWorkspace(model, df)
    ipm = InplaceModeler(model, nrow(df))
    
    # Create perturbation
    orig_values = ws.base_data[variable]
    h = std(orig_values) * 1e-6
    pert_vector = orig_values .+ h
    pert_data = create_perturbed_data(ws.base_data, variable, pert_vector)
    
    println("Step size h = $h")
    println("Variable $variable affects columns: $(ws.variable_plans[variable])")
    
    # --- FULL EVALUATION (Ground Truth) ---
    println("\\n🔍 FULL EVALUATION:")
    X_full = similar(ws.base_matrix)
    modelmatrix!(ipm, pert_data, X_full)
    
    # --- SELECTIVE EVALUATION (Our Approach) ---
    println("\\n🔍 SELECTIVE EVALUATION:")
    X_selective = copy(ws.base_matrix)  # Start with base
    
    # Call modelmatrix_with_base! with detailed logging
    println("Calling modelmatrix_with_base!...")
    debug_modelmatrix_with_base!(ipm, pert_data, X_selective, ws.base_matrix, [variable], ws.mapping)
    
    # --- COMPARE RESULTS ---
    println("\\n📊 COMPARISON:")
    max_diff = maximum(abs.(X_selective - X_full))
    println("Maximum difference: $max_diff")
    
    # Focus on the problematic columns
    affected_cols = ws.variable_plans[variable]
    for col in affected_cols
        col_diff = maximum(abs.(X_selective[:, col] - X_full[:, col]))
        if col_diff > 1e-10
            println("❌ Column $col: max diff = $col_diff")
            
            # Show term info
            term_info = "unknown"
            try
                for (term, range) in ws.mapping.term_info
                    if col in range
                        term_info = string(term)
                        break
                    end
                end
            catch
            end
            println("   Term: $term_info")
            
            # Show first few values for debugging
            println("   Full eval [1:3]:      $(X_full[1:3, col])")
            println("   Selective eval [1:3]: $(X_selective[1:3, col])")
            
            # If this is an interaction, let's debug its components
            if col == 8  # The problematic 3-way interaction
                debug_interaction_components(pert_data, ws, col, ipm)
            end
        else
            println("✅ Column $col: OK (diff = $col_diff)")
        end
    end
    
    return X_full, X_selective
end

"""
    debug_modelmatrix_with_base!(ipm, data, X_target, X_base, changed_vars, mapping)

Instrumented version of modelmatrix_with_base! with detailed logging.
"""
function debug_modelmatrix_with_base!(ipm, data, X_target, X_base, changed_vars, mapping)
    println("  📝 Input: changed_vars = $changed_vars")
    
    # Get affected columns
    total_cols = size(X_target, 2)
    changed_cols = Set{Int}()
    
    for var in changed_vars
        var_cols = get_variable_columns_flat(mapping, var)
        union!(changed_cols, var_cols)
        println("  📝 Variable :$var affects columns: $var_cols")
    end
    
    changed_cols = sort(collect(changed_cols))
    unchanged_cols = get_unchanged_columns(mapping, changed_vars, total_cols)
    
    println("  📝 Total changed columns: $changed_cols")
    println("  📝 Unchanged columns: $unchanged_cols")
    
    # Share memory for unchanged columns
    for col in unchanged_cols
        X_target[:, col] = view(X_base, :, col)
    end
    println("  ✅ Shared memory for $(length(unchanged_cols)) unchanged columns")
    
    # Update changed columns using EfficientModelMatrices function
    if !isempty(changed_cols)
        println("  🔄 Updating $(length(changed_cols)) changed columns...")
        
        # This is where the bug likely is - let's trace what happens
        debug_eval_columns_for_variables!(changed_vars, data, X_target, mapping, ipm)
    end
    
    return X_target
end

"""
    debug_eval_columns_for_variables!(variables, data, X, mapping, ipm)

Instrumented version of eval_columns_for_variables! with detailed logging.
"""
function debug_eval_columns_for_variables!(variables, data, X, mapping, ipm)
    println("    🔍 eval_columns_for_variables! called with: $variables")
    
    if isempty(variables)
        return
    end
    
    # Get all affected columns across all variables
    all_affected_cols = Set{Int}()
    var_term_map = Dict{Symbol, Vector{Tuple{AbstractTerm, UnitRange{Int}}}}()
    
    for var in variables
        var_ranges = get_variable_term_ranges(mapping, var)
        var_term_map[var] = var_ranges
        
        var_cols = get_variable_columns_flat(mapping, var)
        union!(all_affected_cols, var_cols)
        
        println("    📝 Variable :$var has term ranges:")
        for (term, range) in var_ranges
            println("      - $term → columns $range")
        end
    end
    
    affected_cols = sort(collect(all_affected_cols))
    println("    📝 All affected columns: $affected_cols")
    
    # Initialize counters
    fn_i = Ref(1)
    int_i = Ref(1)
    
    # Process each unique term that needs updating
    processed_terms = Set{AbstractTerm}()
    
    for var in variables
        for (term, range) in var_term_map[var]
            if term ∉ processed_terms && !isempty(range)
                println("    🔄 Processing term: $term (columns $range)")
                
                # This is the critical call - let's see what happens here
                debug_cols_selective!(term, data, X, first(range), affected_cols, ipm, fn_i, int_i)
                
                push!(processed_terms, term)
            end
        end
    end
end

"""
    debug_cols_selective!(term, data, X, j, affected_cols, ipm, fn_i, int_i)

Instrumented version of _cols_selective! with detailed logging.
"""
function debug_cols_selective!(term::AbstractTerm, data, X, j, affected_cols, ipm, fn_i, int_i)
    println("      🎯 _cols_selective! for term: $term")
    println("         Starting at column $j")
    println("         Affected columns: $affected_cols")
    
    # Get the width of this term
    w = width(term)
    println("         Term width: $w")
    
    if w == 0
        println("         ⏭️  Zero width, skipping")
        return j
    end
    
    # Determine which columns this term would write to
    term_cols = collect(j:(j + w - 1))
    println("         Term columns: $term_cols")
    
    # Find intersection with affected columns
    cols_to_update = intersect(term_cols, affected_cols)
    println("         Columns to update: $cols_to_update")
    
    if isempty(cols_to_update)
        println("         ⏭️  No columns to update, skipping")
        return j + w
    end
    
    # Create a temporary matrix to hold the full term evaluation
    temp_matrix = Matrix{Float64}(undef, size(X, 1), w)
    println("         📋 Created temp matrix: $(size(temp_matrix))")
    
    # Evaluate the full term into temporary matrix
    println("         🧮 Evaluating term with data...")
    
    # Show what data the term is getting (for debugging)
    if term isa InteractionTerm && length(term.terms) == 3
        println("         🔍 3-way interaction components:")
        for (i, component) in enumerate(term.terms)
            if component isa Term
                val = data[component.sym]
                println("           Component $i ($component): $(val[1:3])... (first 3 values)")
            else
                println("           Component $i: $component")
            end
        end
    end
    
    _cols!(term, data, temp_matrix, 1, ipm, fn_i, int_i)
    
    println("         ✅ Term evaluation complete")
    println("         📊 Result [1:3, :]: $(temp_matrix[1:3, :])")
    
    # Copy only the affected columns to the target matrix
    for col in cols_to_update
        local_col = col - j + 1  # Column index within this term
        X[:, col] = temp_matrix[:, local_col]
        println("         📝 Copied column $col (local $local_col)")
    end
    
    return j + w
end

"""
    debug_interaction_components(data, ws, col, ipm)

Debug the components of the 3-way interaction to see what values they have.
"""
function debug_interaction_components(data, ws, col, ipm)
    println("\\n🔬 DEBUGGING 3-WAY INTERACTION (Column $col):")
    
    # Find the 3-way interaction term
    interaction_term = nothing
    for (term, range) in ws.mapping.term_info
        if col in range && term isa InteractionTerm && length(term.terms) == 3
            interaction_term = term
            break
        end
    end
    
    if interaction_term === nothing
        println("   ❌ Could not find 3-way interaction term for column $col")
        return
    end
    
    println("   📝 Term: $interaction_term")
    println("   📝 Components: $(interaction_term.terms)")
    
    # Check each component's values
    for (i, component) in enumerate(interaction_term.terms)
        println("   🔍 Component $i: $component")
        
        if component isa Term
            var_name = component.sym
            if haskey(data, var_name)
                values = data[var_name]
                println("      Values [1:5]: $(values[1:min(5, length(values))])")
                println("      Type: $(typeof(values))")
                println("      Length: $(length(values))")
            else
                println("      ❌ Variable :$var_name not found in data")
            end
        else
            println("      Complex component: $(typeof(component))")
        end
    end
    
    # Now let's manually compute what the interaction should be
    println("   🧮 Manual computation check:")
    try
        # This is a simplified check - you might need to adapt based on your exact term structure
        if length(interaction_term.terms) == 3
            comp1, comp2, comp3 = interaction_term.terms
            if all(c isa Term for c in [comp1, comp2, comp3])
                val1 = data[comp1.sym]
                val2 = data[comp2.sym] 
                val3 = data[comp3.sym]
                
                manual_result = val1 .* val2 .* val3
                println("      Manual result [1:3]: $(manual_result[1:3])")
            end
        end
    catch e
        println("      ❌ Manual computation failed: $e")
    end
end


function minimal_interaction_test()
    println("\\n🧪 MINIMAL INTERACTION TEST")
    
    # Test the 3-way interaction computation in isolation
    ws = AMEWorkspace(model, df)
    ipm = InplaceModeler(model, nrow(df))
    
    # Get the original data
    orig_data = ws.base_data
    
    # Create perturbed data
    h = std(orig_data[:x]) * 1e-6
    pert_data = merge(orig_data, (:x => orig_data[:x] .+ h,))
    
    println("Original x[1:3]: $(orig_data[:x][1:3])")
    println("Perturbed x[1:3]: $(pert_data[:x][1:3])")
    println("Change: $(h)")
    
    # Find the 3-way interaction term
    interaction_term = nothing
    for (term, range) in ws.mapping.term_info
        if term isa InteractionTerm && length(term.terms) == 3
            interaction_term = term
            println("Found 3-way interaction: $term")
            break
        end
    end
    
    if interaction_term !== nothing
        # Evaluate this term with original data
        println("\\n📊 Evaluating with ORIGINAL data:")
        temp1 = Matrix{Float64}(undef, 3, 1)  # Just first 3 rows for debugging
        fn_i, int_i = Ref(1), Ref(1)
        
        # Create minimal data for testing
        mini_orig = (
            x = orig_data[:x][1:3],
            d = orig_data[:d][1:3], 
            z = orig_data[:z][1:3]
        )
        
        _cols!(interaction_term, mini_orig, temp1, 1, ipm, fn_i, int_i)
        println("Result with original data: $(temp1[:, 1])")
        
        # Evaluate with perturbed data
        println("\\n📊 Evaluating with PERTURBED data:")
        temp2 = Matrix{Float64}(undef, 3, 1)
        fn_i, int_i = Ref(1), Ref(1)
        
        mini_pert = (
            x = pert_data[:x][1:3],
            d = pert_data[:d][1:3],
            z = pert_data[:z][1:3]
        )
        
        _cols!(interaction_term, mini_pert, temp2, 1, ipm, fn_i, int_i)
        println("Result with perturbed data: $(temp2[:, 1])")
        
        println("\\n📈 Expected finite difference: $((temp2[:, 1] - temp1[:, 1]) / h)")
    end
end

#######

# Test script to debug the 3-way interaction issue

# Run this on your failing test case model and data
println("🚀 Starting detailed debugging of selective update...")

# Make sure you have the model and df from the failing test case loaded
# Then run:
X_full, X_selective = debug_selective_vs_full_evaluation(model, df, :x)

# This will produce detailed logging showing:
# 1. Which columns are identified as needing updates
# 2. How the selective update processes each term
# 3. What data each interaction component receives
# 4. Where the computation diverges from ground truth

println("\\n🎯 Focus areas for investigation:")
println("1. Does the 3-way interaction get the right component data?")
println("2. Is _cols! being called correctly for the interaction term?") 
println("3. Are the component values what we expect?")
println("4. Is the Kronecker product computation working correctly?")

# Additional focused debugging
println("\\n" * "="^60)
println("🔍 ADDITIONAL COMPONENT-LEVEL DEBUGGING")

# Let's also create a minimal reproduction
minimal_interaction_test()

println("\\n✅ Debugging complete. Check the logs above to identify where the selective update goes wrong.")
