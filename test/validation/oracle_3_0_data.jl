# test/validation/oracle_3_0_data.jl
#
# The deterministic dataset of the Margins 3.0 R oracles (test/r_compare_marginaleffects/oracle_3_0/,
# `oracle_data()` in oracle_data.R there), built without an RNG so that both languages construct
# the same data. `data_checksums.csv` there holds R's column sums.

using DataFrames, CategoricalArrays

function _oracle3_data()
    i = 1:400
    d = DataFrame(x = 0.5 .+ mod.(i .* 37, 101) ./ 40,
                  w = 0.5 .+ mod.(i .* 13, 17) ./ 10,
                  z = cos.(i .* 0.7),
                  g = categorical(["a", "b", "c"][mod.(i, 3) .+ 1]),
                  sw = 0.5 .+ mod.(i .* 7, 11) ./ 5,
                  fw = Float64.(1 .+ mod.(i .* 5, 4)),
                  expo = Float64.(1 .+ mod.(i .* 11, 7)))
    isb = d.g .== "b"
    d.yb = Float64.(sin.(i .* 1.3) .+ 0.8 .* d.x .- 1.6 .+ 0.5 .* d.z .+ 0.4 .* d.x .* (d.w .- 1) .+
                    0.6 .* isb .* d.w .> 0)
    d.yc = 1 .+ 0.5 .* d.x .+ 0.3 .* d.w .+ 0.4 .* d.x .* d.w .- 0.2 .* d.z .+ sin.(i .* 2.3)
    d.cnt = floor.(d.expo .* exp.(-0.5 .+ 0.3 .* d.x .- 0.3 .* d.z .+ 0.2 .* (d.g .== "c")) .+
                   2 .* (sin.(i .* 2.1) .+ 1))
    return d
end
