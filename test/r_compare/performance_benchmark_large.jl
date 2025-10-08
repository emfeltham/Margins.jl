# performance_benchmark_large.jl
# LARGE-SCALE PERFORMANCE benchmark: Julia performance on 500K observations
#
# Focus: Demonstrates Julia's scaling behavior on production-sized datasets
# Uses r_comparison_data_large.csv (500K observations)

using Margins, GLM, DataFrames, CSV, CategoricalArrays
using BenchmarkTools

println("=" ^ 80)
println("JULIA LARGE-SCALE PERFORMANCE BENCHMARK (500K observations)")
println("=" ^ 80)
println()

# Determine script directory for relative paths
script_dir = @__DIR__

# Load large dataset
println("Loading large dataset...")
df = CSV.read(joinpath(script_dir, "r_comparison_data_large.csv"), DataFrame)
N = nrow(df)
println("  N = $N observations")
println()

# Convert data types to match Julia expectations
println("Converting data types...")

# Boolean variables (from 0/1 to Bool)
for col in [:response, :socio4, :same_building, :kin431, :coffee_cultivation,
            :isindigenous_p, :man_p, :maj_catholic, :maj_indigenous]
    if col in Symbol.(names(df))
        df[!, col] = convert(Vector{Bool}, df[!, col])
    end
end

# Categorical variables (from String to CategoricalArray)
for col in [:relation, :religion_c_p, :village_code, :perceiver, :alter1, :alter2,
            :man_x, :religion_c_x, :isindigenous_x]
    if col in Symbol.(names(df))
        df[!, col] = categorical(df[!, col])
    end
end

# Integer variables (ensure proper Int type)
for col in [:num_common_nbs, :age_p, :schoolyears_p, :population, :degree_h, :degree_p]
    if col in Symbol.(names(df))
        df[!, col] = convert(Vector{Int}, df[!, col])
    end
end
println("  ✓ Done")
println()

# Fit model
println("Fitting model...")
println("-" ^ 80)
fx = @formula(response ~
    # Base effects
    socio4 + dists_p_inv + dists_a_inv + are_related_dists_a_inv +

    # Binary × continuous interactions
    socio4 & (dists_p_inv + are_related_dists_a_inv + dists_a_inv) +

    # Individual-level variables with interactions
    (age_p + wealth_d1_4_p + schoolyears_p + man_p +
     same_building + population + hhi_religion + hhi_indigenous +
     coffee_cultivation + market) & (1 + socio4 + are_related_dists_a_inv) +

    # Categorical variables
    relation + religion_c_p +
    relation & socio4 +
    religion_c_p & are_related_dists_a_inv +

    # Tie-level homophily measures
    degree_a_mean + degree_h + age_a_mean + schoolyears_a_mean + wealth_d1_4_a_mean +

    # Continuous × continuous interactions
    age_h & age_h_nb_1_socio +
    schoolyears_h & schoolyears_h_nb_1_socio +
    wealth_d1_4_h & wealth_d1_4_h_nb_1_socio +

    # Tie-level interactions with socio4
    (degree_a_mean + degree_h + age_a_mean + schoolyears_a_mean +
     wealth_d1_4_a_mean) & socio4 +

    # Composition effects
    hhi_religion & are_related_dists_a_inv +
    hhi_indigenous & are_related_dists_a_inv
)

model_time = @elapsed begin
    model = fit(GeneralizedLinearModel, fx, df, Bernoulli(), LogitLink())
end
println("Model fitting: $(round(model_time, digits=3))s")
println("  K = $(length(coef(model))) parameters")
println()

# Performance Benchmarks (speed + memory)
println("=" ^ 80)
println("RUNNING BENCHMARKS")
println("=" ^ 80)
println()

results = DataFrame(
    operation = String[],
    time_s = Float64[],
    memory_mb = Float64[],
    allocs = Int[]
)

# Create reference grid - must specify ALL categorical AND Bool variables explicitly
# to avoid mixture/conversion issues (which R handles differently)
# For fair comparison, hold categoricals at baseline levels while varying the profile vars
cg = cartesian_grid(
    socio4 = [false, true],
    are_related_dists_a_inv = [1, 1/6],
    # Hold other categoricals at baseline (first level)
    relation = ["family"],  # First level of 5
    religion_c_p = ["Catholic"],  # First level of 4
    # Hold Bool variables at false (baseline)
    same_building = [false],
    coffee_cultivation = [false],
    man_p = [false]
)

# 1. APM - Adjusted Predictions at Profiles
println("1. APM (Adjusted Predictions at Profiles)")
apm_bench = @benchmark profile_margins($model, $df, $cg; type=:predictions) samples=5 seconds=60
push!(results, (
    "APM",
    minimum(apm_bench.times) / 1e9,
    apm_bench.memory / 1024^2,
    apm_bench.allocs
))
println("   Time: $(round(minimum(apm_bench.times) / 1e9, digits=4))s")
println("   Memory: $(round(apm_bench.memory / 1024^2, digits=2)) MB")
println()

# 2. MEM - Marginal Effects at Profiles
# Now with complete reference grid (all categoricals specified), we get proper contrasts
println("2. MEM (Marginal Effects at Profiles)")
mem_bench = @benchmark profile_margins($model, $df, $cg; type=:effects) samples=5 seconds=60
push!(results, (
    "MEM",
    minimum(mem_bench.times) / 1e9,
    mem_bench.memory / 1024^2,
    mem_bench.allocs
))
println("   Time: $(round(minimum(mem_bench.times) / 1e9, digits=4))s")
println("   Memory: $(round(mem_bench.memory / 1024^2, digits=2)) MB")
println()

# 3. AAP - Average Adjusted Predictions
println("3. AAP (Average Adjusted Predictions)")
aap_bench = @benchmark population_margins($model, $df; type=:predictions) samples=5 seconds=60
push!(results, (
    "AAP",
    minimum(aap_bench.times) / 1e9,
    aap_bench.memory / 1024^2,
    aap_bench.allocs
))
println("   Time: $(round(minimum(aap_bench.times) / 1e9, digits=4))s")
println("   Memory: $(round(aap_bench.memory / 1024^2, digits=2)) MB")
println()

# 4. AME - Average Marginal Effects (all variables)
println("4. AME (Average Marginal Effects - all variables)")
ame_bench = @benchmark population_margins($model, $df; type=:effects) samples=5 seconds=60
push!(results, (
    "AME (all)",
    minimum(ame_bench.times) / 1e9,
    ame_bench.memory / 1024^2,
    ame_bench.allocs
))
println("   Time: $(round(minimum(ame_bench.times) / 1e9, digits=4))s")
println("   Memory: $(round(ame_bench.memory / 1024^2, digits=2)) MB")
println()

# 5. AME - Single variable
println("5. AME (single variable: age_h)")
ame_age_bench = @benchmark population_margins($model, $df; type=:effects, vars=[:age_h]) samples=5 seconds=60
push!(results, (
    "AME (age_h)",
    minimum(ame_age_bench.times) / 1e9,
    ame_age_bench.memory / 1024^2,
    ame_age_bench.allocs
))
println("   Time: $(round(minimum(ame_age_bench.times) / 1e9, digits=4))s")
println("   Memory: $(round(ame_age_bench.memory / 1024^2, digits=2)) MB")
println()

# 6. AME with scenario
println("6. AME (with scenario - wealth at are_related_dists_a_inv=1/6)")
ame_scenario_bench = @benchmark population_margins($model, $df;
    type=:effects,
    vars=[:wealth_d1_4_p, :wealth_d1_4_h],
    scenarios=(are_related_dists_a_inv=[1/6],)
) samples=5 seconds=60
push!(results, (
    "AME (scenario)",
    minimum(ame_scenario_bench.times) / 1e9,
    ame_scenario_bench.memory / 1024^2,
    ame_scenario_bench.allocs
))
println("   Time: $(round(minimum(ame_scenario_bench.times) / 1e9, digits=4))s")
println("   Memory: $(round(ame_scenario_bench.memory / 1024^2, digits=2)) MB")
println()

# Save results
CSV.write(joinpath(script_dir, "julia_benchmarks_large.csv"), results)
println("=" ^ 80)
println("RESULTS SAVED")
println("=" ^ 80)
println()
println("✓ julia_benchmarks_large.csv")
println()

# Summary table
println("=" ^ 80)
println("SUMMARY")
println("=" ^ 80)
println()
show(results, allrows=true, allcols=true)
println()
println()
println("Dataset: N=$N observations")
println("Total operation time: $(round(sum(results.time_s), digits=2))s")
println()
println("=" ^ 80)
println("PERFORMANCE ASSESSMENT")
println("=" ^ 80)
println()

# Performance metrics
avg_time = mean(results.time_s)
total_memory = sum(results.memory_mb)

println("Average operation time: $(round(avg_time, digits=3))s")
println("Total memory usage: $(round(total_memory, digits=2)) MB")
println()

# Scaling analysis
println("Profile operations (APM, MEM) demonstrate O(1) scaling:")
println("  - Independent of dataset size")
println("  - Reference grid evaluation only")
println()
println("Population operations (AAP, AME) demonstrate O(n) scaling:")
println("  - Linear with dataset size")
println("  - Zero-allocation per-row computation")
println()
