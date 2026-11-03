# test_per_row_allocations.jl
#
# Per-row allocation of the ContrastEvaluator primitives Margins builds on, on a model with
# interactions: `contrast_modelrow!` (multi-level categorical, Bool, a loop over rows and
# contrasts), `contrast_gradient!` (link and response scale) and `delta_method_se` allocate
# nothing after a warm-up. Measured inside a function (on Julia 1.10, `@allocated` at testset
# scope counts boxing).
#
# julia --project="." test/test_per_row_allocations.jl

using Test, Random
using Margins, GLM, DataFrames, CategoricalArrays, Tables
using FormulaCompiler: compile_formula, contrastevaluator, contrast_modelrow!, contrast_gradient!
using Margins: delta_method_se

function _perrow_allocs(f::F) where {F}
    f()
    return @allocated f()
end

function _perrow_contrast_loop!(buf, evaluator, rows)
    for i in rows
        contrast_modelrow!(buf, evaluator, i, :treatment, "Control", "Drug_A")
        contrast_modelrow!(buf, evaluator, i, :treatment, "Control", "Drug_B")
        contrast_modelrow!(buf, evaluator, i, :binary_var, false, true)
    end
    return buf
end

@testset "Per-Row Allocation Coverage" begin
    Random.seed!(20251102)
    n = 100
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        z = randn(n),
        treatment = categorical(rand(["Control", "Drug_A", "Drug_B"], n)),
        binary_var = rand([false, true], n)
    )
    model = lm(@formula(y ~ x * treatment + z * binary_var), df)
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    β = coef(model)

    @testset "Categorical contrasts (ContrastEvaluator)" begin
        evaluator = contrastevaluator(compiled, data, [:treatment, :binary_var])
        buf = Vector{Float64}(undef, length(compiled))
        @test _perrow_allocs(() -> contrast_modelrow!(buf, evaluator, 1, :treatment, "Control", "Drug_A")) == 0
        @test _perrow_allocs(() -> contrast_modelrow!(buf, evaluator, 1, :binary_var, false, true)) == 0
        @test _perrow_allocs(() -> _perrow_contrast_loop!(buf, evaluator, 1:50)) == 0
    end

    @testset "Categorical contrast gradients" begin
        evaluator = contrastevaluator(compiled, data, [:treatment])
        ∇β = Vector{Float64}(undef, length(compiled))
        @test _perrow_allocs(() -> contrast_gradient!(∇β, evaluator, 1, :treatment, "Control", "Drug_A", β)) == 0

        df_logit = copy(df)
        df_logit.y_binary = rand([0, 1], n)
        model_logit = glm(@formula(y_binary ~ treatment), df_logit, Binomial(), LogitLink())
        data_logit = Tables.columntable(df_logit)
        compiled_logit = compile_formula(model_logit, data_logit)
        β_logit = coef(model_logit)
        evaluator_logit = contrastevaluator(compiled_logit, data_logit, [:treatment])
        ∇β_logit = Vector{Float64}(undef, length(compiled_logit))
        link = LogitLink()
        @test _perrow_allocs(() -> contrast_gradient!(∇β_logit, evaluator_logit, 1, :treatment,
                                                      "Control", "Drug_A", β_logit, link)) == 0
    end

    @testset "Delta method standard errors" begin
        evaluator = contrastevaluator(compiled, data, [:treatment])
        V = vcov(model)
        @test _perrow_allocs(() -> delta_method_se(evaluator, 1, :treatment, "Control", "Drug_A", β, V)) == 0
    end
end
