# complex_example.jl
# Evaluate performance in a very complex formula, with lots of observations

using Revise
using GLM, Margins
using DataFrames

# Use test example dataset
# import CSV
# df = CSV.read("test/test_data.csv", DataFrame)

# Load original data or generate synthetic large dataset
include("support/generate_large_synthetic_data.jl")
df = generate_synthetic_dataset(Int(620_000/2); seed = 08540);

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
        # random intercepts
        # (1|perceiver) + (1|village_code)
        # excluded from homophily interactions (by BIC):
        # dists_p_inv
        # !socio4 & dists_a_inv
        # coffee_cultivation + market
    )
    fit(
        GeneralizedLinearModel, fx, df, Bernoulli(), LogitLink();
    );
end;

# Margins
# Margins.clear_engine_cache!()

scenario = (socio4 = [false, true], are_related_dists_a_inv = [1/6],);

cg = cartesian_grid(socio4 = [false, true], are_related_dists_a_inv = [1, 1/6]);

@time apm_result = profile_margins(m, df, cg; type=:predictions);
DataFrame(apm_result)

@time mem_result = profile_margins(m, df, cg; type = :effects, contrasts = :pairwise);
DataFrame(mem_result)

@time aap_result = population_margins(m, df; type=:predictions);
DataFrame(aap_result)

@time ame_result = population_margins(m, df; type=:effects);

@time ame_result_ = population_margins(m, df; type=:effects, scenarios = scenario);

@time ame_result_1 = population_margins(m, df; type=:effects, vars = [:religion_c_x], contrasts = :pairwise);
DataFrame(ame_result_1)

@time ame_result_1 = population_margins(m, df; type=:effects, vars = [:wealth_d1_4_p, :wealth_d1_4_h], scenarios = scenario);
DataFrame(ame_result_1)

@time predict(m, df);
