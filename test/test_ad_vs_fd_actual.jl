# Test AD vs FD with the actual complex model
using GLM, Margins, DataFrames, FormulaCompiler, Tables, BenchmarkTools

# Load data generation
include("support/generate_large_synthetic_data.jl")

# Small dataset for testing
df = generate_synthetic_dataset(1000; seed = 08540)

# Fit the complex model
m = let
    fx = @formula(response ~
        socio4 +
        (1 + socio4) & (dists_p_inv + are_related_dists_a_inv) +
        !socio4 & dists_a_inv +
        (num_common_nbs & (dists_a_inv <= inv(2))) & (1 + are_related_dists_a_inv + dists_p_inv) +
        # individual variables
        (schoolyears_p + wealth_d1_4_p + man_p + age_p + religion_c_p +
        same_building + population +
        hhi_religion + hhi_indigenous +
        coffee_cultivation + market +
        relation) & (1 + socio4 + are_related_dists_a_inv) +
        # tie variables
        (
            degree_a_mean + degree_h +
            age_a_mean + age_h * age_h_nb_1_socio +
            schoolyears_a_mean + schoolyears_h * schoolyears_h_nb_1_socio +
            man_x * man_x_mixed_nb_1 +
            wealth_d1_4_a_mean + wealth_d1_4_h * wealth_d1_4_h_nb_1_socio +
            isindigenous_x * isindigenous_homop_nb_1 +
            religion_c_x * religion_homop_nb_1
        ) & (1 + socio4 + are_related_dists_a_inv) +
        religion_c_x & hhi_religion +
        isindigenous_x & hhi_indigenous
    )
    fit(GeneralizedLinearModel, fx, df, Bernoulli(), LogitLink())
end

println("Model with $(length(coef(m))) parameters")

# Build FormulaCompiler components directly
data = Tables.columntable(df)
compiled = compile_formula(m, data)
β = coef(m)

# Get continuous variables
cont_vars = continuous_variables(compiled, data)
println("$(length(cont_vars)) continuous variables")

# Build derivative evaluator for just a few variables
test_vars = cont_vars[1:min(3, length(cont_vars))]
de = build_derivative_evaluator(compiled, data; vars=test_vars)

println("\n=== Testing single row performance ===")
row = 1
var = test_vars[1]
gβ = zeros(length(de))

# Test FD
println("FD backend (single column):")
@btime accumulate_ame_gradient!(
    $gβ, $de, $β, [$row], $var;
    link=$(LogitLink()), backend=:fd
)

# Test AD
println("AD backend (full Jacobian, extract column):")
@btime accumulate_ame_gradient!(
    $gβ, $de, $β, [$row], $var;
    link=$(LogitLink()), backend=:ad
)

println("\n=== Testing with 100 rows ===")
rows = 1:100

# FD
println("FD backend:")
@btime accumulate_ame_gradient!(
    $gβ, $de, $β, $rows, $var;
    link=$(LogitLink()), backend=:fd
)

# AD
println("AD backend:")
@btime accumulate_ame_gradient!(
    $gβ, $de, $β, $rows, $var;
    link=$(LogitLink()), backend=:ad
)

println("\n=== Key observation ===")
println("With $(length(β)) parameters:")
println("- FD computes 1 column via 2 formula evaluations")
println("- AD computes all $(length(test_vars)) columns via dual arithmetic")
println("- AD overhead grows with model complexity")
println("\nFor this complex model, FD is likely faster for single variables")
println("AD would be better if we needed all $(length(test_vars)) variables at once")