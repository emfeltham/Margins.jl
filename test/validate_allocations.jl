#!/usr/bin/env julia
# validate_allocations.jl — Reproducible allocation checks for Margins.jl
#
# Scenario categories:
# - [TEST]: Micro/diagnostic loops (not used in production); help isolate causes
# - [PROD]: Actual production paths used by Margins.jl API
# julia --project="test" validate_allocations.jl > validate_allocations.txt 2>&1

using BenchmarkTools
using GLM
using DataFrames
using Tables
using Margins
import FormulaCompiler

function print_allocs(label, allocs_small, n_small, allocs_large, n_large)
    per_row_small = allocs_small / n_small
    per_row_large = allocs_large / n_large
    ratio = allocs_large / max(allocs_small, 1)
    println("$label")
    println("  Small n=$n_small: allocs=$(allocs_small), per-row=$(round(per_row_small, digits=4))")
    println("  Large n=$n_large: allocs=$(allocs_large), per-row=$(round(per_row_large, digits=4))")
    println("  Scaling ratio: $(round(ratio, digits=3)) (dataset $(round(n_large/n_small, digits=1))x larger)")
    println()
end

function build_simple(n)
    df = DataFrame(x = randn(n), y = randn(n))
    m = lm(@formula(y ~ x), df)
    nt = Tables.columntable(df)
    eng = Margins.build_engine(m, nt, [:x])
    return (df, m, nt, eng)
end

function fc_modelrow_loop(engine, data_nt, rows)
    for r in rows
        FormulaCompiler.modelrow!(engine.row_buf, engine.compiled, data_nt, r)
    end
end

function fc_modelrow_loop_optimized(engine, data_nt, rows)
    # Extract to locals once to avoid per-iteration field access
    row_buf = engine.row_buf
    compiled = engine.compiled
    for r in rows
        FormulaCompiler.modelrow!(row_buf, compiled, data_nt, r)
    end
end

function fc_modelrow_loop_pure(row_buf, compiled, data_nt, rows)
    for r in rows
        FormulaCompiler.modelrow!(row_buf, compiled, data_nt, r)
    end
end

function fc_me_eta_loop(engine, rows)
    for r in rows
        FormulaCompiler.marginal_effects_eta!(engine.g_buf, engine.de, engine.β, r; backend=:fd)
    end
end

function fc_me_eta_loop_optimized(engine, rows)
    # Extract to locals once to avoid per-iteration field access
    g_buf = engine.g_buf
    de = engine.de
    β = engine.β
    for r in rows
        FormulaCompiler.marginal_effects_eta!(g_buf, de, β, r; backend=:fd)
    end
end

function main()
    println("🔎 Validating allocations with clean, warmed benchmarks…")
    n_small, n_large = 2000, 10000
    df_s, m_s, nt_s, eng_s = build_simple(n_small)
    df_l, m_l, nt_l, eng_l = build_simple(n_large)
    rows_s = 1:n_small
    rows_l = 1:n_large

    # Warmup
    fc_modelrow_loop(eng_s, nt_s, 1:10)
    fc_modelrow_loop(eng_l, nt_l, 1:10)
    fc_me_eta_loop(eng_s, 1:10)
    fc_me_eta_loop(eng_l, 1:10)
    Margins._population_predictions(eng_s, nt_s; target=:mu)
    Margins._population_predictions(eng_l, nt_l; target=:mu)

    # 1) FormulaCompiler primitives in loops
    b = @benchmark fc_modelrow_loop($eng_s, $nt_s, $rows_s) samples=8 evals=1
    a_s = minimum(b).allocs
    b = @benchmark fc_modelrow_loop($eng_l, $nt_l, $rows_l) samples=8 evals=1
    a_l = minimum(b).allocs
    print_allocs("[TEST] modelrow! loop", a_s, n_small, a_l, n_large)
    
    b = @benchmark fc_modelrow_loop_optimized($eng_s, $nt_s, $rows_s) samples=8 evals=1
    a_s = minimum(b).allocs
    b = @benchmark fc_modelrow_loop_optimized($eng_l, $nt_l, $rows_l) samples=8 evals=1
    a_l = minimum(b).allocs
    print_allocs("[TEST] modelrow! loop (optimized locals)", a_s, n_small, a_l, n_large)

    # Pure locals: pass compiled + buffers directly (no engine field access inside or outside)
    b = @benchmark fc_modelrow_loop_pure($(eng_s.row_buf), $(eng_s.compiled), $nt_s, $rows_s) samples=8 evals=1
    a_s = minimum(b).allocs
    b = @benchmark fc_modelrow_loop_pure($(eng_l.row_buf), $(eng_l.compiled), $nt_l, $rows_l) samples=8 evals=1
    a_l = minimum(b).allocs
    print_allocs("[TEST] modelrow! loop (pure locals)", a_s, n_small, a_l, n_large)

    b = @benchmark fc_me_eta_loop($eng_s, $rows_s) samples=8 evals=1
    a_s = minimum(b).allocs
    b = @benchmark fc_me_eta_loop($eng_l, $rows_l) samples=8 evals=1
    a_l = minimum(b).allocs
    print_allocs("[TEST] marginal_effects_eta! loop", a_s, n_small, a_l, n_large)
    
    b = @benchmark fc_me_eta_loop_optimized($eng_s, $rows_s) samples=8 evals=1
    a_s = minimum(b).allocs
    b = @benchmark fc_me_eta_loop_optimized($eng_l, $rows_l) samples=8 evals=1
    a_l = minimum(b).allocs
    print_allocs("[TEST] marginal_effects_eta! loop (optimized locals)", a_s, n_small, a_l, n_large)

    # 2) Population predictions (end-to-end internal primitive)
    b = @benchmark Margins._population_predictions($eng_s, $nt_s; target=:mu) samples=8 evals=1
    a_s = minimum(b).allocs
    b = @benchmark Margins._population_predictions($eng_l, $nt_l; target=:mu) samples=8 evals=1
    a_l = minimum(b).allocs
    print_allocs("[PROD] _population_predictions (μ)", a_s, n_small, a_l, n_large)

    # 3) Public API end-to-end
    b = @benchmark population_margins($m_s, $df_s; type=:predictions, backend=:fd) samples=8 evals=1
    a_s = minimum(b).allocs
    b = @benchmark population_margins($m_l, $df_l; type=:predictions, backend=:fd) samples=8 evals=1
    a_l = minimum(b).allocs
    print_allocs("[PROD] population_margins (predictions)", a_s, n_small, a_l, n_large)

    b = @benchmark population_margins($m_s, $df_s; type=:effects, vars=[:x], backend=:fd) samples=8 evals=1
    a_s = minimum(b).allocs
    b = @benchmark population_margins($m_l, $df_l; type=:effects, vars=[:x], backend=:fd) samples=8 evals=1
    a_l = minimum(b).allocs
    print_allocs("[PROD] population_margins (effects)", a_s, n_small, a_l, n_large)
end

main()
