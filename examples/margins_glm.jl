using DataFrames, CategoricalArrays, GLM, Margins
# Robust covariance provider (optional)
# using CovarianceMatrices

df = DataFrame(y = rand(Bool, 1000),
                x = randn(1000),
                z = randn(1000),
                g = categorical(rand(["A","B"], 1000)))

m = glm(@formula(y ~ x + z + g), df, Binomial(), LogitLink())

# AME on response scale (default)
res_ame = ame(m, df; dydx=[:x, :z])
println(first(res_ame.table, 5))

# MER at representative values
res_mer = mer(m, df; dydx=[:x], at=Dict(:x=>[-1,0,1], :g=>["A","B"]))
println(first(res_mer.table, 5))

# Adjusted predictions (APR) on response or link
res_apr_mu = apr(m, df; target=:mu, at=Dict(:x=>[-2,0,2]))
res_apr_eta = apr(m, df; target=:eta, at=Dict(:x=>[-2,0,2]))
println(first(res_apr_mu.table, 3)); println(first(res_apr_eta.table, 3))

# Robust covariance via CovarianceMatrices
# res_robust = ame(m, df; dydx=[:x], vcov = HC1())
# println(res_robust.table)
