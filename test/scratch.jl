# scratch.jl

using DataFrames

df1 = DataFrame(A = 1:10^6, B = rand(10^6))
df2 = copy(df1; copycols=false)  # Minimal allocation

# Safe: replace a column (allocates only for new column)
df2[!, :A] = 10 .- df2[!, :A]

# Unsafe: mutates both df1 and df2 (avoid this)
# df2[!, :B][1] = 0.0

df1 == df2
hcat(df1[!, :A], df2[!, :A])