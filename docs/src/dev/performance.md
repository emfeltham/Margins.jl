# Developer Performance Guide

This page summarizes best practices to keep the per-row loops allocation-free, and per-call allocations from growing with n, without compromising statistical correctness.

## Core Principles

- Preserve statistical validity (delta-method, full Σ) at all times
- Aim for constant allocations w.r.t. sample size in production code (the per-call Float64 copies of integer columns and of columns a scenario replaces are the accepted O(n) exceptions)
- Preallocate once; reuse buffers and result tables
- Avoid dynamic growth in hot paths

## Patterns That Work

- Compile/cache FormulaCompiler artifacts outside loops
- Move hot loops into helpers that take concrete arguments (`compiled`, `row_buf`, `β`, `link`, `de`)
- Reuse `row_buf`, `g_buf`, `gβ_accumulator`, the `MEWorkspace`; avoid temporary vectors
- Prefer scalar loops over broadcasts that allocate
- Preallocate DataFrame columns and assign by index

## Validation

- The allocation tests are in `test/performance/` (`test_allocation_scaling.jl`, `test_ame_fd_loop_zero_alloc.jl`, `test_population_aap_loop_zero_alloc.jl`).

Expected (tested): the row loops allocate 0 bytes; warm `population_margins` calls (predictions and effects) on Float64 data allocate about the same amount at n = 1,000 and 50,000 (well below one 8n-byte column).

