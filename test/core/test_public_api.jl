# test/core/test_public_api.jl
#
# The public API of Margins 3.0, frozen at the 3.0.0 release (plus, since: the random_effects,
# re_new and re_terms keywords and metadata keys, Unreleased in CHANGELOG). Each pin below fails when the
# public surface changes: an exported name added or removed, a keyword added to or removed
# from an exported function, a positional form added or removed, a field of an exported type
# renamed, or a column of a representative result renamed, added, removed or moved.
#
# A deliberate API change after 3.0.0 must update the pin here AND have a CHANGELOG entry
# (with a migration-table row for anything removed or renamed). Additive changes (a new
# keyword, a new exported name) also update the pin, so that every change is reviewed.

using Test, Margins, GLM, DataFrames, CategoricalArrays

@testset "Public API freeze (3.0.0)" begin

    @testset "exported names" begin
        expected = Set([
            :CategoricalMixture, :ContrastResult, :EffectsResult, :Inference, :MarginsResult,
            :PredictionsResult, :ResultContext,
            :population_margins, :profile_margins, :second_differences, :contrast,
            :balanced_grid, :cartesian_grid, :complete_reference_grid, :hierarchical_grid,
            :means_grid, :quantile_grid,
            :create_balanced_mixture, :create_mixture_column, :expand_mixture_grid, :mix,
            :validate_mixture_levels, :validate_mixture_weights,
            :get_display_digits, :get_profile_digits, :set_display_digits, :set_profile_digits,
        ])
        exported = Set(filter(!=(:Margins), names(Margins)))
        @test isempty(setdiff(exported, expected))   # added without updating this pin
        @test isempty(setdiff(expected, exported))   # removed
    end

    # Per exported function: the set of (positional arity, keyword names) over its methods.
    # `kwargs...` methods are the ones that reject removed forms with a migration message.
    @testset "positional forms and keywords" begin
        S(x...) = Symbol(x...)
        expected = Dict(
            :population_margins => Set([(2, [:alpha, :backend, :ci_alpha, :contrasts, :df, :groups, :measure, :random_effects, :re_new, :re_nodes, :re_se, :re_terms, :scale, :scenarios, :type, :vars, :vcov, :warn, :weights])]),
            :profile_margins => Set([(3, [:alpha, :backend, :ci_alpha, :contrasts, :df, :interval, :measure, :offset, :random_effects, :re_new, :re_nodes, :re_se, :re_terms, :scale, :type, :vars, :vcov, :warn])]),
            :second_differences => Set([
                (2, [S("kwargs...")]),
                (3, [:all_contrasts, :alpha, :contrast, :include_gradients, :modifier_type, :p_adjust, :vcov, :warn]),
                (4, [:alpha, :at, :backend, :contrast, :contrasts, :delta, :df, :groups, :include_gradients, :levels, :measure, :modifier_type, :p_adjust, :profile, :random_effects, :re_new, :re_nodes, :re_se, :re_terms, :scale, :vcov, :warn, :weights]),
                (4, [S("kwargs...")]),
            ]),
            :contrast => Set([(2, [S("kwargs...")]), (3, [:alpha]), (4, [S("kwargs...")])]),
            :balanced_grid => Set([(1, [:typical, S("vars..."), :weights])]),
            :cartesian_grid => Set([(0, [S("vars...")]), (1, Symbol[])]),
            :complete_reference_grid => Set([(3, [:typical, :weights])]),
            :hierarchical_grid => Set([(2, [:max_depth, :warn_large, :weights])]),
            :means_grid => Set([(1, [:typical, :weights]), (2, [:typical, :weights])]),   # (data), (model, data)
            :quantile_grid => Set([(1, [S("vars..."), :weights])]),
            :create_balanced_mixture => Set([(1, Symbol[])]),
            :create_mixture_column => Set([(2, Symbol[])]),
            :expand_mixture_grid => Set([(2, Symbol[])]),
            :get_display_digits => Set([(0, Symbol[])]),
            :get_profile_digits => Set([(0, Symbol[])]),
            :set_display_digits => Set([(1, Symbol[])]),
            :set_profile_digits => Set([(1, Symbol[])]),
        )
        for n in names(Margins)
            f = getfield(Margins, n)
            f isa Function || continue
            ms = [m for m in methods(f) if m.module === Margins]
            isempty(ms) && continue   # re-exported from FormulaCompiler (mix, validate_mixture_*)
            got = Set((m.nargs - 1, sort(Base.kwarg_decl(m))) for m in ms)
            @test haskey(expected, n)
            haskey(expected, n) || continue
            @test got == expected[n]
            got == expected[n] || @info "API pin differs" n got expected = expected[n]
        end
    end

    @testset "fields of exported types" begin
        @test fieldnames(EffectsResult) == (:estimates, :standard_errors, :variables, :terms, :context, :gradients, :inference, :metadata)
        @test fieldnames(PredictionsResult) == (:estimates, :standard_errors, :context, :gradients, :inference, :metadata)
        @test fieldnames(ContrastResult) == (:contrast, :se, :statistic, :df, :p_value, :ci_lower, :ci_upper, :estimate1, :estimate2, :gradient, :row1, :row2, :metadata)
        @test fieldnames(Inference) == (:vcov, :df, :alpha, :dropped)
        @test fieldnames(ResultContext) == (:columns, :kinds)
        @test fieldnames(CategoricalMixture) == (:levels, :weights)
        @test first.(Margins.result_metadata_keys()) == [:analysis_type, :type, :vars, :n_vars, :scale,
            :measure, :backend, :n_obs, :weighted, :weights_source, :random_effects, :re_terms, :has_contexts, :groups_vars,
            :scenarios_vars, :model_type, :timestamp]
        @test Margins.SECOND_DIFFERENCE_COLUMNS == (:variable, :contrast, :modifier, :modifier_type,
            :modifier_level1, :modifier_level2, :ame_at_level1, :ame_at_level2, :second_diff, :se,
            :statistic, :df, :p_value, :ci_lower, :ci_upper, :significant)
    end

    @testset "result columns" begin
    api_n = 120
    api_fr(t) = t - floor(t)
    api_data = let i = 1:api_n
        d = DataFrame(x = api_fr.(i .* 0.6180339887) .* 2, z = api_fr.(i .* 0.7548776662),
                      t = api_fr.(i .* 0.4142135624) .< 0.5,
                      g = categorical(["a", "b", "c"][1 .+ mod.(i, 3)]))
        d.y = 1 .+ d.x .+ 0.5 .* d.t .+ 0.3 .* d.x .* d.t .+ api_fr.(i .* i .* 0.1234567)
        d.yb = api_fr.(i .* i .* 0.2718281828) .< 1 ./ (1 .+ exp.(-(d.x .- 1 .+ d.t .- 0.5 .* d.z)))
        d
    end
    api_lm = lm(@formula(y ~ x * t + g + z), api_data)
    api_logit = glm(@formula(yb ~ x * t + g + z), api_data, Binomial(), LogitLink())
    api_ame = population_margins(api_lm, api_data; type=:effects)
    api_sc = population_margins(api_logit, api_data; type=:effects, vars=[:x], scenarios=(t=[false, true],))
    api_calls = [
        "AME"                   => () -> DataFrame(api_ame),
        "AME compact"           => () -> DataFrame(api_ame; format=:compact),
        "AME confidence"        => () -> DataFrame(api_ame; format=:confidence),
        "AME stata lm"          => () -> DataFrame(api_ame; format=:stata),
        "AME stata logit"       => () -> DataFrame(population_margins(api_logit, api_data; vars=[:x]); format=:stata),
        "AME gradients"         => () -> DataFrame(api_ame; include_gradients=true),
        "AAP"                   => () -> DataFrame(population_margins(api_lm, api_data; type=:predictions)),
        "AME groups"            => () -> DataFrame(population_margins(api_lm, api_data; vars=[:x], groups=:g)),
        "AME scenarios"         => () -> DataFrame(api_sc),
        "AAP scenarios+groups"  => () -> DataFrame(population_margins(api_lm, api_data; type=:predictions, scenarios=(z=[0.2, 0.8],), groups=:g)),
        "MEM"                   => () -> DataFrame(profile_margins(api_lm, api_data, cartesian_grid(x=[0.5, 1.5]); type=:effects)),
        "MEM standard"          => () -> DataFrame(profile_margins(api_lm, api_data, cartesian_grid(x=[0.5, 1.5]); type=:effects); format=:standard),
        "APM"                   => () -> DataFrame(profile_margins(api_lm, api_data, cartesian_grid(x=[0.5, 1.5]); type=:predictions)),
        "APM prediction int."   => () -> DataFrame(profile_margins(api_lm, api_data, cartesian_grid(x=[0.5, 1.5]); type=:predictions, interval=:prediction)),
        "SD result"             => () -> second_differences(api_sc, :x, :t),
        "SD result p_adjust"    => () -> second_differences(api_sc, :x, :t; p_adjust=:holm),
        "SD model discrete"     => () -> second_differences(api_logit, api_data, :x, :t),
        "SD model at=:mean"     => () -> second_differences(api_logit, api_data, :x, :z; at=:mean),
        "SD model groups"       => () -> second_differences(api_logit, api_data, :x, :t; groups=:g),
        "SD model profile"      => () -> second_differences(api_logit, api_data, :x, :t; profile=(z=[0.25, 0.75],)),
        "MEM means_grid(model)" => () -> DataFrame(profile_margins(api_lm, api_data, means_grid(api_lm, api_data); type=:effects)),
    ]
        stat = ["statistic", "df", "p_value", "ci_lower", "ci_upper"]
        sd = ["variable", "contrast", "modifier", "modifier_type", "modifier_level1", "modifier_level2",
              "ame_at_level1", "ame_at_level2", "second_diff", "se", "statistic", "df", "p_value",
              "ci_lower", "ci_upper", "significant"]
        expected = Dict(
            "AME"                  => ["type", "variable", "contrast", "estimate", "se", stat..., "n"],
            "AME compact"          => ["type", "variable", "contrast", "estimate", "se"],
            "AME confidence"       => ["type", "variable", "contrast", "estimate", "ci_lower", "ci_upper"],
            "AME stata lm"         => ["type", "margin", "std_err", "t", "P_t", "N"],
            "AME stata logit"      => ["type", "margin", "std_err", "z", "P_z", "N"],
            "AME gradients"        => ["type", "variable", "contrast", "estimate", "se", stat..., "n", "gradient"],
            "AAP"                  => ["type", "estimate", "se", stat..., "n"],
            "AME groups"           => ["over_g", "type", "variable", "contrast", "estimate", "se", stat..., "n"],
            "AME scenarios"        => ["at_t", "type", "variable", "contrast", "estimate", "se", stat..., "n"],
            "AAP scenarios+groups" => ["over_g", "at_z", "type", "estimate", "se", stat..., "n"],
            "MEM"                  => ["x", "type", "variable", "contrast", "estimate", "se", stat..., "n"],
            "MEM standard"         => ["x", "type", "variable", "contrast", "estimate", "se", stat..., "n"],
            "APM"                  => ["x", "type", "estimate", "se", stat..., "n"],
            "APM prediction int."  => ["x", "type", "estimate", "se", stat..., "n"],
            "SD result"            => sd,
            "SD result p_adjust"   => [sd[1:13]..., "p_adjusted", sd[14:end]...],
            "SD model discrete"    => sd,
            "SD model at=:mean"    => [sd..., "eval_point", "delta_used"],
            "SD model groups"      => [sd..., "over_g"],
            "SD model profile"     => [sd..., "z"],
            "MEM means_grid(model)" => ["x", "z", "t", "g", "type", "variable", "contrast", "estimate", "se", stat..., "n"],
        )
        @test Set(first.(api_calls)) == Set(keys(expected))
        Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            for (label, f) in api_calls
                got = names(f())
                @test got == expected[label]
                got == expected[label] || @info "column pin differs" label got expected = expected[label]
            end
        end
        # behaviour defaults: profile fill-ins and complete_reference_grid use the model's weights
        wfit = lm(@formula(y ~ x * t + g + z), api_data; wts = 1 .+ api_data.z)
        wres = profile_margins(wfit, api_data, cartesian_grid(x=[0.5]); type=:predictions)
        @test wres.metadata[:weights_source] === :model
        @test complete_reference_grid(cartesian_grid(x=[0.5]), wfit, api_data).z[1] ≈
              sum((1 .+ api_data.z) .* api_data.z) / sum(1 .+ api_data.z)
        @test api_sc.context.kinds == [:scenario]
        @test keys(api_sc.context.columns) == (:t,)
        @test api_sc.inference isa Inference
    end
end
