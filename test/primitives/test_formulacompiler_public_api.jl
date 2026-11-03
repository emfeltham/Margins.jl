# test_formulacompiler_public_api.jl
# Margins uses only FormulaCompiler's exported API. Every name Margins imports from
# FormulaCompiler (`using/import FormulaCompiler: …`) or accesses qualified
# (`FormulaCompiler.name`, including method extensions such as the
# `fixed_effects_form` hook) must be exported by FormulaCompiler. Field access on FC objects
# cannot be checked this way; the FC accessors (`derivative_variables`, `contrast_endpoints`,
# …) replace it.

using Test
using DataFrames, CategoricalArrays, GLM, StatsModels
using FormulaCompiler
using Margins

function _fc_names_used(dir)
    used = Dict{Symbol, Vector{String}}()
    record(s, loc) = push!(get!(used, s, String[]), loc)
    function walk(ex, loc)
        ex isa Expr || return
        if ex.head in (:using, :import)
            for a in ex.args
                # `using FormulaCompiler: a, b` parses as (:(:), FormulaCompiler, a, b)
                if a isa Expr && a.head == :(:) && a.args[1] == Expr(:., :FormulaCompiler)
                    for b in a.args[2:end]
                        record(b.args[end], loc)
                    end
                end
            end
            return
        end
        if ex.head == :. && ex.args[1] == :FormulaCompiler && ex.args[2] isa QuoteNode
            record(ex.args[2].value, loc)
        end
        for a in ex.args
            if a isa LineNumberNode
                loc = string(basename(String(a.file)), ":", a.line)
            else
                walk(a, loc)
            end
        end
    end
    for (root, _, files) in walkdir(dir), f in files
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        walk(Meta.parseall(read(path, String); filename = path), f)
    end
    return used
end

@testset "Margins uses only FormulaCompiler's public API" begin
    pkg = pkgdir(Margins)
    used = merge(_fc_names_used(joinpath(pkg, "src")), _fc_names_used(joinpath(pkg, "ext")))
    @test !isempty(used)
    private = Dict(s => l for (s, l) in used if !Base.isexported(FormulaCompiler, s))
    @test isempty(private)

    # Margins' own create_balanced_mixture (FC has an unexported function of that name)
    @test parentmodule(Margins.create_balanced_mixture) === Margins
    @test create_balanced_mixture(["Red", "Green", "Blue"]) ==
          Dict("Red" => 1/3, "Green" => 1/3, "Blue" => 1/3)
end

@testset "Baseline level from the model's contrasts" begin
    # The default baseline contrasts are against FC's `baseline_level`, which respects
    # DummyCoding(base = …); the first contrast level ("A") is not the baseline here.
    df = DataFrame(y = repeat([0.1, 0.5, 0.9, 1.4], 25), x = range(0, 1; length = 100),
                   g = categorical(repeat(["A", "B", "C", "A"], 25)))
    m = lm(@formula(y ~ x + g), df; contrasts = Dict(:g => DummyCoding(base = "B")))
    res = DataFrame(population_margins(m, df; type = :effects, vars = [:g]))
    @test sort(res.contrast) == ["A - B", "C - B"]
    β = Dict(zip(coefnames(m), coef(m)))
    @test res.estimate[res.contrast .== "A - B"][1] ≈ β["g: A"]
    @test res.estimate[res.contrast .== "C - B"][1] ≈ β["g: C"]
end
