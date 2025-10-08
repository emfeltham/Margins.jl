# Test suite for categorical mixture workflow helper functions
# These utilities support reference grid construction for marginal effects

using Test
using Margins: create_mixture_column, expand_mixture_grid, create_balanced_mixture
using FormulaCompiler: mix, is_mixture_column, validate_mixture_consistency!,
    extract_mixture_spec, validate_mixture_weights, validate_mixture_levels

@testset "Mixture Workflow Utilities" begin

    @testset "create_mixture_column Integration" begin
        # Test that helper-created columns work with mixture operations
        mixture = mix("P" => 0.7, "Q" => 0.3)
        mixture_col = create_mixture_column(mixture, 4)

        test_data = (
            x = [1.0, 2.0, 3.0, 4.0],
            group = mixture_col
        )

        # Should pass validation
        @test_nowarn validate_mixture_consistency!(test_data)
        @test is_mixture_column(test_data.group)

        # Should extract correct spec
        spec = extract_mixture_spec(test_data.group[1])
        @test spec.levels == ["P", "Q"]
        @test spec.weights == [0.7, 0.3]
    end

    @testset "expand_mixture_grid Integration" begin
        base_data = (x = [1.0, 2.0],)
        mixture_specs = Dict(:group => mix("Alpha" => 0.4, "Beta" => 0.6))

        expanded = expand_mixture_grid(base_data, mixture_specs)
        result_data = expanded[1]

        # Should create valid mixture data
        @test_nowarn validate_mixture_consistency!(result_data)
        @test is_mixture_column(result_data.group)

        # Should have correct structure
        @test length(result_data.x) == 2
        @test length(result_data.group) == 2

        spec = extract_mixture_spec(result_data.group[1])
        @test spec.levels == ["Alpha", "Beta"]
        @test spec.weights == [0.4, 0.6]
    end

    @testset "create_balanced_mixture Integration" begin
        balanced_dict = create_balanced_mixture(["Red", "Green", "Blue"])

        # Should create valid balanced mixture
        @test length(balanced_dict) == 3
        @test all(w ≈ 1/3 for w in values(balanced_dict))

        # Should pass validation
        validate_mixture_weights(collect(values(balanced_dict)))
        validate_mixture_levels(collect(keys(balanced_dict)))
    end

end
