using Margins, GLM, DataFrames, Tables

# Simple test case
n = 1000
df = DataFrame(
    y = randn(n),
    x = randn(n),
    z = randn(n)
)

model = lm(@formula(y ~ x + z), df)
data_nt = Tables.columntable(df)

# First call (warmup)
result1 = population_margins(model, df; vars=[:x, :z], backend=:fd)

# Profile allocations
using Profile
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1.0 begin
    result2 = population_margins(model, df; vars=[:x, :z], backend=:fd)
end

# Get allocation profile
prof = Profile.Allocs.fetch()

# Show allocations
println("\n=== Allocation Profile ===")
for (i, alloc) in enumerate(prof.allocs)
    if !isnothing(alloc.type)
        frames = alloc.stacktrace
        if !isempty(frames)
            # Find first frame in Margins package
            for frame in frames
                if occursin("Margins", string(frame.file))
                    println("$(alloc.size) bytes: $(alloc.type) at $(frame.file):$(frame.line)")
                    break
                end
            end
        end
    end
end
