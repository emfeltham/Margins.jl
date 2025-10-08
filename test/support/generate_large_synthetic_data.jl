"""
Generate synthetic dataset based on the structure of test_data.csv
Creates a 620K row dataset for performance testing.
"""

using DataFrames
using Random
using Distributions
using StatsBase
using CategoricalArrays

function generate_synthetic_dataset(n_rows::Int = 620_000; seed::Int = 08540)
    Random.seed!(seed)

    # Helper function to generate UUID-like strings
    function generate_uuid_like()
        chars = "0123456789abcdef-"
        return join([rand(chars) for _ in 1:36])
    end

    # Helper function to generate mixed categorical values like "false, true"
    function generate_mixed_categorical(options, prob_mixed=0.3)
        if rand() < prob_mixed
            selected = sample(options, min(2, length(options)), replace=false)
            return join(selected, ", ")
        else
            return rand(options)
        end
    end

    # Generate base distributions from the sample data
    village_codes = [152, 153, 154, 155, 156, 157, 158, 159, 160, 161]  # Sample village codes
    relations = ["free_time", "work", "family", "neighbor", "friend"]
    religions = ["Catholic", "Protestant", "No Religion", "Other"]

    # Create the DataFrame
    df = DataFrame(
        # Binary response
        response = rand(Bernoulli(0.35), n_rows),  # ~35% true rate from sample

        # ID columns - generate realistic UUIDs (categorical for perceiver, strings for alters)
        perceiver = categorical([generate_uuid_like() for _ in 1:n_rows]),
        alter1 = [generate_uuid_like() for _ in 1:n_rows],
        alter2 = [generate_uuid_like() for _ in 1:n_rows],

        # Categorical variables
        village_code = categorical(rand(village_codes, n_rows)),
        relation = categorical(rand(relations, n_rows)),
        kin431 = rand(Bernoulli(0.25), n_rows),  # ~25% true rate
        same_building = rand(Bernoulli(0.15), n_rows),  # ~15% true rate
        socio4 = rand(Bernoulli(0.4), n_rows),  # ~40% true rate

        # Distance variables - log-normal distribution to match observed pattern
        dists_a = rand(LogNormal(1.5, 0.8), n_rows),
        dists_p = rand(LogNormal(1.3, 0.7), n_rows),
        are_related_dists_a = rand(LogNormal(1.8, 1.2), n_rows),

        # Mixed categorical variables (can have multiple values)
        man_x = categorical([generate_mixed_categorical(["true", "false"], 0.2) for _ in 1:n_rows]),
        religion_c_x = categorical([generate_mixed_categorical(religions, 0.3) for _ in 1:n_rows]),
        isindigenous_x = categorical([generate_mixed_categorical(["true", "false"], 0.15) for _ in 1:n_rows]),

        # Individual characteristics
        isindigenous_p = rand(Bernoulli(0.2), n_rows),
        man_p = rand(Bernoulli(0.48), n_rows),
        religion_c_p = categorical(rand(religions, n_rows)),
        coffee_cultivation = rand(Bernoulli(0.7), n_rows),
        market = rand(Normal(-0.4, 0.3), n_rows),  # Continuous variable centered around -0.4
        maj_catholic = rand(Bernoulli(0.8), n_rows),
        maj_indigenous = rand(Bernoulli(0.15), n_rows),

        # Count variables
        num_common_nbs = rand(Poisson(0.5), n_rows),  # Low count variable

        # Age variables (in years)
        age_a_mean = rand(Uniform(18, 80), n_rows),
        age_h = rand(Uniform(0, 70), n_rows),  # Age homophily measure
        age_p = rand(18:65, n_rows),  # Discrete age

        # Wealth variables
        wealth_d1_4_h = rand(Beta(2, 5), n_rows),  # Beta distribution for wealth homophily
        wealth_d1_4_a_mean = rand(Beta(3, 4), n_rows),
        wealth_d1_4_p = rand(Uniform(0, 1), n_rows),

        # Education variables
        schoolyears_p = rand(0:16, n_rows),
        schoolyears_a_mean = rand(Uniform(0, 15), n_rows),
        schoolyears_h = rand(Uniform(0, 10), n_rows),
        schoolyears_h_nb_1_socio = rand(0:5, n_rows),

        # Population and demographic variables
        population = rand([50, 100, 150, 200, 244, 300, 400, 500], n_rows),  # Discrete population sizes
        pct_protestant = rand(Beta(2, 8), n_rows),  # Percentage variables
        pct_catholic = rand(Beta(8, 3), n_rows),
        pct_indigenous = rand(Beta(2, 8), n_rows),
        pct_notindigenous = rand(Beta(8, 2), n_rows),

        # Homophily measures
        homop_religion = rand(Beta(6, 4), n_rows),
        hhi_religion = rand(Beta(4, 6), n_rows),
        homop_indigenous = rand(Beta(3, 7), n_rows),
        hhi_indigenous = rand(Beta(2, 8), n_rows),

        # Network measures
        religion_homop_nb_1 = rand(Uniform(0, 1), n_rows),
        isindigenous_homop_nb_1 = rand(Uniform(0, 1), n_rows),
        wealth_d1_4_h_nb_1_socio = rand(Uniform(0, 0.1), n_rows),
        age_h_nb_1_socio = rand(Uniform(0, 80), n_rows),

        # Degree variables
        degree_a_mean = rand(Uniform(1, 25), n_rows),
        degree_h = rand(0:10, n_rows),
        degree_p = rand(1:8, n_rows),

        # Additional network measures
        man_x_mixed_nb_1 = rand(Uniform(0, 2), n_rows)
    )

    # Compute inverse distance measures
    df.are_related_dists_a_inv = 1 ./ df.are_related_dists_a
    df.dists_p_inv = 1 ./ df.dists_p
    df.dists_a_inv = 1 ./ df.dists_a

    # Handle potential Inf values by replacing with maximum finite value
    finite_are_related = filter(isfinite, df.are_related_dists_a_inv)
    finite_dists_p = filter(isfinite, df.dists_p_inv)
    finite_dists_a = filter(isfinite, df.dists_a_inv)

    if !isempty(finite_are_related)
        replace!(df.are_related_dists_a_inv, Inf => maximum(finite_are_related))
    end
    if !isempty(finite_dists_p)
        replace!(df.dists_p_inv, Inf => maximum(finite_dists_p))
    end
    if !isempty(finite_dists_a)
        replace!(df.dists_a_inv, Inf => maximum(finite_dists_a))
    end

    return df
end

# # Function to save the synthetic dataset
# function save_synthetic_dataset(filename::String = "test/synthetic_large_data.csv"; n_rows::Int = 620_000)
#     println("Generating synthetic dataset with $n_rows rows...")
#     df = generate_synthetic_dataset(n_rows)

#     println("Saving to $filename...")
#     CSV.write(filename, df)

#     println("Dataset saved successfully!")
#     println("Dimensions: $(size(df))")
#     println("First few rows:")
#     println(first(df, 3))

#     return df
# end