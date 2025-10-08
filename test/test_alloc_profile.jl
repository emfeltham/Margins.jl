using Profile
using GLM, Margins, DataFrames, FormulaCompiler, Tables

# Small test case
n = 1000
df = DataFrame(
    y = rand(Bool, n),
    x1 = randn(n),
    x2 = randn(n)
)

model = glm(@formula(y ~ x1 + x2), df, Binomial(), LogitLink())

# Build engine
data = Tables.columntable(df)
engine = Margins.get_or_build_engine(Margins.PopulationUsage, model, data, [:x1], GLM.vcov)

# The function to profile
gβ_sum = zeros(length(engine.β))
rows = 1:length(first(data))
var = :x1
link = LogitLink()

# Warm up
Margins._accumulate_unweighted_ame_gradient!(gβ_sum, engine, rows, var, link, :ad)

# Profile allocations
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1 Margins._accumulate_unweighted_ame_gradient!(gβ_sum, engine, rows, var, link, :ad)

# Get results
prof = Profile.Allocs.fetch()

# Print allocation sources
println("Top allocation sources:")
for (i, alloc) in enumerate(sort(prof.allocs, by=a->a.size, rev=true)[1:min(10, length(prof.allocs))])
    println("$i. $(alloc.size) bytes at:")
    for frame in alloc.stacktrace[1:min(3, length(alloc.stacktrace))]
        println("   ", frame)
    end
end