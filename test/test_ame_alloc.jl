# julia --project="." test/test_ame_alloc.jl > test/test_ame_alloc.txt 2>&1


## ASBOLTELY DO NOT USE :fd
## DO NOT CHEAT BY RAISING TESTS ABOVE 0 ALLOCS

using Test
using DataFrames, CategoricalArrays
using Tables, StatsModels, GLM
using BenchmarkTools
using Margins
using FormulaCompiler
using LinearAlgebra: dot

#=
Assessment:
- This file tests O(1) per-row allocations after setup.
    - Compiles once and preallocates buffers/scenarios outside measurement.
    - Benchmarks a single-row kernel on both scales via named functions with full interpolation.
    - Asserts minimum(memory) == 0 for that per-row body across n ∈ {1e2, 1e4, 1e5, 1e6}.
    - This demonstrates constant (zero) additional allocations per row once compiled and preallocated.
    - This demonstrates constant (zero) additional allocations per row once compiled and preallocated. 
Scope clarification:
    - It proves the variable-cost kernel is allocation-free (the O(1) you care about).
    - It doesn’t claim the entire end-to-end AME over all rows is 0 bytes; total memory for the full call may include fixed/setup costs outside the
per-row loop.
=#

"""
    _one_row_response!(grad_sum, g_false, g_true, row_buf, compiled, scen_f, scen_t, r, β, link)

Compute one-row boolean scenario gradient delta on response scale in-place with no allocations.
Returns one gradient component delta to anchor the benchmark (avoids linkinv in body).
"""
function _one_row_response!(grad_sum::AbstractVector{Float64},
                            g_false::AbstractVector{Float64},
                            g_true::AbstractVector{Float64},
                            row_buf::AbstractVector{Float64},
                            compiled,
                            scen_f,
                            scen_t,
                            r::Int,
                            β::AbstractVector{<:Real},
                            link::L) where {L<:GLM.Link}
    # false scenario
    # Compute gradients in-place (internally builds model row)
    Margins._gradient_with_scenario!(g_false, compiled, scen_f, r, :response, β, link, row_buf)
    Margins._gradient_with_scenario!(g_true,  compiled, scen_t, r, :response, β, link, row_buf)
    @inbounds @fastmath for i in eachindex(grad_sum)
        grad_sum[i] += (g_true[i] - g_false[i])
    end
    return g_true[1] - g_false[1]
end

"""
    _one_row_link!(grad_sum, g_false, g_true, row_buf, compiled, scen_f, scen_t, r, β, link)

Compute one-row boolean scenario contrast on link scale in-place with no allocations.
Returns the discrete change in η (η_true − η_false).
"""
function _one_row_link!(grad_sum::AbstractVector{Float64},
                        g_false::AbstractVector{Float64},
                        g_true::AbstractVector{Float64},
                        row_buf::AbstractVector{Float64},
                        compiled,
                        scen_f,
                        scen_t,
                        r::Int,
                        β::AbstractVector{<:Real},
                        link::L) where {L<:GLM.Link}
    # false scenario
    # Compute gradients in-place (internally builds model row)
    Margins._gradient_with_scenario!(g_false, compiled, scen_f, r, :link, β, link, row_buf)
    Margins._gradient_with_scenario!(g_true,  compiled, scen_t, r, :link, β, link, row_buf)
    @inbounds @fastmath for i in eachindex(grad_sum)
        grad_sum[i] += (g_true[i] - g_false[i])
    end
    return g_true[1] - g_false[1]
end

@testset "Alloc tests" begin

@testset "Boolean AME via scenarios (population)" begin
    # Synthetic data with boolean and continuous predictors
    n = 200
    df = DataFrame(
        y = rand(Bool, n),
        b = rand(Bool, n),
        x = randn(n),
    )

    # Fit a logistic regression
    model = glm(@formula(y ~ b + x), df, Binomial(), LogitLink())
    data_nt = Tables.columntable(df)

    # Build engine for boolean variable
    engine = Margins.build_engine(model, data_nt, [:b], GLM.vcov)
    rows = collect(1:n)

    # Compute AME with new in-place scenario-based implementation (response scale)
    ame_resp, gβ_resp = Margins._compute_boolean_ame(engine, :b, rows, :response, :ad)
    @debug "AME (response)" ame_resp

    # Reference implementation (response scale): scenario loop using FormulaCompiler primitives
    compiled = engine.compiled
    β = engine.β
    link = engine.link
    row_buf = engine.row_buf
    scen_f = FormulaCompiler.create_scenario("b_false", data_nt, Dict(:b => false))
    scen_t = FormulaCompiler.create_scenario("b_true",  data_nt, Dict(:b => true))

    # Compute average discrete difference and average gradient difference
    ref_sum = 0.0
    ref_gsum = zeros(length(β))
    for r in rows
        # false scenario
        FormulaCompiler.modelrow!(row_buf, compiled, scen_f.data, r)
        ηf = dot(row_buf, β)
        μf = GLM.linkinv(link, ηf)
        gβf = GLM.mueta(link, ηf) .* row_buf
        # true scenario
        FormulaCompiler.modelrow!(row_buf, compiled, scen_t.data, r)
        ηt = dot(row_buf, β)
        μt = GLM.linkinv(link, ηt)
        gβt = GLM.mueta(link, ηt) .* row_buf

        ref_sum += (μt - μf)
        ref_gsum .+= (gβt .- gβf)
    end
    ref_ame = ref_sum / n
    ref_gβ = ref_gsum ./ n
    @debug "Ref AME (response)" ref_ame

    @test isapprox(ame_resp, ref_ame; rtol=1e-8, atol=1e-10)
    @test length(gβ_resp) == length(ref_gβ)
    @test all(isapprox.(gβ_resp, ref_gβ; rtol=1e-8, atol=1e-10))

    # Link scale check
    ame_link, gβ_link = Margins._compute_boolean_ame(engine, :b, rows, :link, :ad)
    @debug "AME (link)" ame_link
    # Reference (link): gradient is the design row; effect is ηt-ηf
    ref_sum = 0.0
    ref_gsum .= 0.0
    for r in rows
        FormulaCompiler.modelrow!(row_buf, compiled, scen_f.data, r)
        ηf = dot(row_buf, β)
        gβf = copy(row_buf)
        FormulaCompiler.modelrow!(row_buf, compiled, scen_t.data, r)
        ηt = dot(row_buf, β)
        gβt = copy(row_buf)
        ref_sum += (ηt - ηf)
        ref_gsum .+= (gβt .- gβf)
    end
    ref_ame = ref_sum / n
    ref_gβ = ref_gsum ./ n
    @debug "Ref AME (link)" ref_ame
    @test isapprox(ame_link, ref_ame; rtol=1e-8, atol=1e-10)
    @test all(isapprox.(gβ_link, ref_gβ; rtol=1e-8, atol=1e-10))
end

@testset "Boolean AME per-row allocations ~O(1) (link scale)" begin
    ns = [100, 10_000, 100_000, 1_000_000]
    for n in ns
        # Generate data and fit model
        df = DataFrame(
            y = rand(Bool, n),
            b = rand(Bool, n),
            x = randn(n),
        )
        model = glm(@formula(y ~ b + x), df, Binomial(), LogitLink())
        data_nt = Tables.columntable(df)
        engine = Margins.build_engine(model, data_nt, [:b], GLM.vcov)
        rows = collect(1:n)

        # Set up scenarios and buffers once
        compiled = engine.compiled
        β = engine.β
        row_buf = engine.row_buf
        scen_f = FormulaCompiler.create_scenario("b_false", data_nt, Dict(:b => false))
        scen_t = FormulaCompiler.create_scenario("b_true",  data_nt, Dict(:b => true))
        g_false = Vector{Float64}(undef, length(β))
        g_true  = Vector{Float64}(undef, length(β))
        grad_sum = similar(g_false); fill!(grad_sum, 0.0)
        r = 1

        # Warmup
        _one_row_link!(grad_sum, g_false, g_true, row_buf, compiled, scen_f, scen_t, r, β, engine.link)

        # Benchmark per-row body
        b = @benchmark $(_one_row_link!)($grad_sum, $g_false, $g_true, $row_buf, $compiled, $scen_f, $scen_t, $r, $β, $(engine.link)) samples=10 evals=1
        mem = minimum(b).memory
        @debug "Per-row allocations (link) for n=$n" mem
        # Assert zero per-row memory after warmup
        @test mem == 0
    end
end

@testset "Boolean AME per-row allocations ~O(1) (response scale)" begin
    ns = [100, 10_000, 100_000, 1_000_000]
    for n in ns
        # Generate data and fit model
        df = DataFrame(
            y = rand(Bool, n),
            b = rand(Bool, n),
            x = randn(n),
        )
        model = glm(@formula(y ~ b + x), df, Binomial(), LogitLink())
        data_nt = Tables.columntable(df)
        engine = Margins.build_engine(model, data_nt, [:b], GLM.vcov)
        rows = collect(1:n)

        # Set up scenarios and buffers once
        compiled = engine.compiled
        β = engine.β
        link = engine.link
        row_buf = engine.row_buf
        scen_f = FormulaCompiler.create_scenario("b_false", data_nt, Dict(:b => false))
        scen_t = FormulaCompiler.create_scenario("b_true",  data_nt, Dict(:b => true))
        g_false = Vector{Float64}(undef, length(β))
        g_true  = Vector{Float64}(undef, length(β))
        grad_sum = similar(g_false); fill!(grad_sum, 0.0)
        r = 1

        # Warmup
        _one_row_response!(grad_sum, g_false, g_true, row_buf, compiled, scen_f, scen_t, r, β, link)

        # Benchmark per-row body
        b = @benchmark $(_one_row_response!)($grad_sum, $g_false, $g_true, $row_buf, $compiled, $scen_f, $scen_t, $r, $β, $link) samples=10 evals=1
        mem = minimum(b).memory
        @debug "Per-row allocations for n=$n" mem
        # Assert zero per-row memory after warmup
        @test mem == 0
    end
end

# Continuous AME (AD backend default): small constant O(1) per-row allocations
"""
    _one_row_me_eta_ad!(g, de, β, row)
"""
function _one_row_me_eta_ad!(g::AbstractVector{Float64}, de, β::AbstractVector{<:Real}, row::Int)
    FormulaCompiler.marginal_effects_eta!(g, de, β, row; backend=:ad)
    return g[1]
end

"""
    _one_row_me_mu_ad!(g, de, β, row, link)
"""
function _one_row_me_mu_ad!(g::AbstractVector{Float64}, de, β::AbstractVector{<:Real}, row::Int, link)
    FormulaCompiler.marginal_effects_mu!(g, de, β, row; link=link, backend=:ad)
    return g[1]
end

@testset "Continuous AME per-row allocations ~O(1) with AD" begin
    ns = [100, 10_000, 100_000, 1_000_000]
    mems_eta = Int[]; mems_mu = Int[]
    for n in ns
        df = DataFrame(y = randn(n), x = randn(n), z = randn(n))
        model = lm(@formula(y ~ x + z), df)
        data_nt = Tables.columntable(df)
        engine = Margins.build_engine(model, data_nt, [:x, :z], GLM.vcov)
        @test engine.de !== nothing
        de = engine.de
        β = engine.β
        link = engine.link
        g = Vector{Float64}(undef, length(de.vars))
        row = 1

        # Warmup
        _one_row_me_eta_ad!(g, de, β, row)
        _one_row_me_mu_ad!(g, de, β, row, link)

        # Benchmarks
        b1 = @benchmark $(_one_row_me_eta_ad!)($g, $de, $β, $row) samples=10 evals=1
        b2 = @benchmark $(_one_row_me_mu_ad!)($g, $de, $β, $row, $link) samples=10 evals=1
        push!(mems_eta, minimum(b1).memory)
        push!(mems_mu,  minimum(b2).memory)
        @debug "Continuous AD per-row bytes (eta, mu) for n=$n" mems_eta[end] mems_mu[end]
    end
    # Assert zero per-row memory across sizes
    @test all(m == 0 for m in mems_eta)
    @test all(m == 0 for m in mems_mu)
end

# Categorical AME (discrete contrasts): small constant O(1) per-row allocations
"""
    _one_row_cat_response!(grad_sum, g1, g2, row_buf, compiled, scen_a, scen_b, row, β, link)
"""
function _one_row_cat_response!(grad_sum::AbstractVector{Float64},
                                g1::AbstractVector{Float64},
                                g2::AbstractVector{Float64},
                                row_buf::AbstractVector{Float64},
                                compiled,
                                scen_a,
                                scen_b,
                                row::Int,
                                β::AbstractVector{<:Real},
                                link::L) where {L<:GLM.Link}
    Margins._gradient_with_scenario!(g1, compiled, scen_a, row, :response, β, link, row_buf)
    Margins._gradient_with_scenario!(g2, compiled, scen_b, row, :response, β, link, row_buf)
    @inbounds @fastmath for i in eachindex(grad_sum)
        grad_sum[i] += (g2[i] - g1[i])
    end
    return g2[1] - g1[1]
end

"""
    _one_row_cat_link!(grad_sum, g1, g2, row_buf, compiled, scen_a, scen_b, row, β, link)
"""
function _one_row_cat_link!(grad_sum::AbstractVector{Float64},
                            g1::AbstractVector{Float64},
                            g2::AbstractVector{Float64},
                            row_buf::AbstractVector{Float64},
                            compiled,
                            scen_a,
                            scen_b,
                            row::Int,
                            β::AbstractVector{<:Real},
                            link::L) where {L<:GLM.Link}
    Margins._gradient_with_scenario!(g1, compiled, scen_a, row, :link, β, link, row_buf)
    Margins._gradient_with_scenario!(g2, compiled, scen_b, row, :link, β, link, row_buf)
    @inbounds @fastmath for i in eachindex(grad_sum)
        grad_sum[i] += (g2[i] - g1[i])
    end
    return g2[1] - g1[1]
end

@testset "Categorical AME per-row allocations ~O(1)" begin
    ns = [100, 10_000, 100_000, 1_000_000]
    for n in ns
        df = DataFrame(y = rand(Bool, n), x = randn(n), g = categorical(rand(["A", "B"], n)))
        model = glm(@formula(y ~ x + g), df, Binomial(), LogitLink())
        data_nt = Tables.columntable(df)
        engine = Margins.build_engine(model, data_nt, [:x], GLM.vcov)

        compiled = engine.compiled
        β = engine.β
        link = engine.link
        row_buf = engine.row_buf
        scen_a = FormulaCompiler.create_scenario("g_A", data_nt, Dict(:g => "A"))
        scen_b = FormulaCompiler.create_scenario("g_B", data_nt, Dict(:g => "B"))
        g1 = Vector{Float64}(undef, length(β))
        g2 = Vector{Float64}(undef, length(β))
        grad_sum = similar(g1); fill!(grad_sum, 0.0)
        r = 1

        # Pure modelrow! on categorical scenario should be zero-alloc
        FormulaCompiler.modelrow!(row_buf, compiled, scen_a.data, r) # warmup
        b0 = @benchmark FormulaCompiler.modelrow!($row_buf, $compiled, $(scen_a.data), $r) samples=10 evals=1
        @test minimum(b0).memory == 0

        # Pure single-scenario gradient (link scale) should be zero-alloc
        Margins._gradient_with_scenario!(g1, compiled, scen_a, r, :link, β, link, row_buf) # warmup
        b_link_single = @benchmark $(Margins._gradient_with_scenario!)($g1, $compiled, $scen_a, $r, :link, $β, $link, $row_buf) samples=10 evals=1
        @test minimum(b_link_single).memory == 0

        # Pure single-scenario gradient (response scale) should be zero-alloc
        Margins._gradient_with_scenario!(g1, compiled, scen_a, r, :response, β, link, row_buf) # warmup
        b_resp_single = @benchmark $(Margins._gradient_with_scenario!)($g1, $compiled, $scen_a, $r, :response, $β, $link, $row_buf) samples=10 evals=1
        @test minimum(b_resp_single).memory == 0

        # Warmup and benchmark response scale
        _one_row_cat_response!(grad_sum, g1, g2, row_buf, compiled, scen_a, scen_b, r, β, link)
        b1 = @benchmark $(_one_row_cat_response!)($grad_sum, $g1, $g2, $row_buf, $compiled, $scen_a, $scen_b, $r, $β, $link) samples=10 evals=1
        @test minimum(b1).memory == 0

        # Warmup and benchmark link scale
        fill!(grad_sum, 0.0)
        _one_row_cat_link!(grad_sum, g1, g2, row_buf, compiled, scen_a, scen_b, r, β, link)
        b2 = @benchmark $(_one_row_cat_link!)($grad_sum, $g1, $g2, $row_buf, $compiled, $scen_a, $scen_b, $r, $β, $link) samples=10 evals=1
        @test minimum(b2).memory == 0
    end
end

# Mixed model: continuous (AD) + categorical (discrete). Test both per-row paths separately.
@testset "Mixed AME per-row allocations (continuous AD + categorical)" begin
    ns = [100, 10_000, 100_000]
    for n in ns
        df = DataFrame(y = rand(Bool, n), x = randn(n), g = categorical(rand(["A", "B"], n)))
        model = glm(@formula(y ~ x + g), df, Binomial(), LogitLink())
        data_nt = Tables.columntable(df)
        engine = Margins.build_engine(model, data_nt, [:x], GLM.vcov)
        de = engine.de
        β = engine.β
        link = engine.link
        compiled = engine.compiled
        row_buf = engine.row_buf
        scen_a = FormulaCompiler.create_scenario("g_A", data_nt, Dict(:g => "A"))
        scen_b = FormulaCompiler.create_scenario("g_B", data_nt, Dict(:g => "B"))
        g_cont = Vector{Float64}(undef, length(de.vars))
        g1 = Vector{Float64}(undef, length(β))
        g2 = Vector{Float64}(undef, length(β))
        grad_sum = similar(g1); fill!(grad_sum, 0.0)
        r = 1

        # Continuous per-row (AD)
        _one_row_me_eta_ad!(g_cont, de, β, r)
        b_cont = @benchmark $(_one_row_me_eta_ad!)($g_cont, $de, $β, $r) samples=10 evals=1
        @test minimum(b_cont).memory == 0

        # Categorical per-row (in-place gradient delta)
        _one_row_cat_link!(grad_sum, g1, g2, row_buf, compiled, scen_a, scen_b, r, β, link)
        b_cat = @benchmark $(_one_row_cat_link!)($grad_sum, $g1, $g2, $row_buf, $compiled, $scen_a, $scen_b, $r, $β, $link) samples=10 evals=1
        @test minimum(b_cat).memory == 0
    end
end

end
