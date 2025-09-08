# Check what's actually running
using Margins, GLM, DataFrames, Tables
using FormulaCompiler

println("=== IMPLEMENTATION STATUS CHECK ===\n")

# Test data
df = DataFrame(y = randn(100), x = randn(100))
model = lm(@formula(y ~ x), df)
data_nt = Tables.columntable(df)
engine = Margins.get_or_build_engine(model, data_nt, [:x], GLM.vcov)

println("1. Check if new function exists:")
try
    methods_count = length(methods(Margins._accumulate_unweighted_ame_gradient!))
    println("   ✅ _accumulate_unweighted_ame_gradient! exists ($methods_count methods)")
catch e
    println("   ❌ Function missing: $e")
end

println("\n2. Test individual FormulaCompiler calls (should be ~0 bytes):")

# Test FormulaCompiler calls directly
gβ_temp = engine.de.fd_yminus
println("   Testing FormulaCompiler.fd_jacobian_column!:")
alloc_fc = @allocated FormulaCompiler.fd_jacobian_column!(gβ_temp, engine.de, 1, :x)
println("   FormulaCompiler call: $alloc_fc bytes")

println("\n3. Test our new function (should be ~0 bytes):")
gβ_buffer = Vector{Float64}(undef, length(engine.β))
alloc_new = @allocated Margins._accumulate_unweighted_ame_gradient!(
    gβ_buffer, engine.de, engine.β, 1:100, :x;
    link=engine.link, backend=:fd
)
println("   Our new function: $alloc_new bytes")

println("\n4. Test original _compute_continuous_ame (should be fixed):")
alloc_orig = @allocated Margins._compute_continuous_ame(engine, :x, 1:100, :response, :fd)
println("   _compute_continuous_ame: $alloc_orig bytes")

println("\n=== DIAGNOSIS ===")
if alloc_fc > 1000
    println("🔍 FormulaCompiler calls are allocating - this suggests dataset/evaluator size issue")
elseif alloc_new > 1000
    println("🔍 Our new function has bugs - implementation needs fixing") 
elseif alloc_orig > 1000
    println("🔍 Main function still broken - wrong code path being used")
else
    println("✅ Everything looks good - maybe just need full system test")
end