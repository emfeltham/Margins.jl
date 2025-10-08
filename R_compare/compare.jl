#!/usr/bin/env julia
# Compare R and Julia results after running pipelines
# Usage: julia --project=. compare.jl > compare.txt 2>&1

include("jl/compare.jl")

println("Running R vs Julia comparison...")
println()

# Only compare datasets that both pipelines produced
# Julia skips iris (multinomial models), so exclude it from comparison
results, details = compare_results(datasets=["mtcars","toothgrowth","titanic"], rdir="results_r", jldir="results_julia"; detailed = true)
println(results)

# Summary statistics
total = nrow(results)
ok_count = count(==("ok"), results.status)
diff_count = count(==("diff"), results.status) 
error_count = count(==("error"), results.status)

println()
println("=== COMPARISON SUMMARY ===")
println("Total comparisons: $total")
println("Perfect matches (ok): $ok_count ($(round(100*ok_count/total, digits=1))%)")
println("Differences (diff): $diff_count ($(round(100*diff_count/total, digits=1))%)")
println("Errors: $error_count ($(round(100*error_count/total, digits=1))%)")

if ok_count == total
    println()
    println("ALL COMPARISONS PASSED!")
elseif error_count == 0
    println()
    println("Some statistical differences found - investigation needed")
else
    println()
    println("Some comparisons failed - check implementation")
end

println("ok")
println(@subset details .!(.!:est_ok .| .!:se_ok))
println("not ok")
println(@subset details (.!:est_ok .| .!:se_ok))
