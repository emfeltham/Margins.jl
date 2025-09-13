# Developer Performance Guide

This page summarizes best practices to ensure O(1) allocations in production paths without compromising statistical correctness. It mirrors PERFORMANCE_BEST_PRACTICES.md at the repo root for convenient browsing.

See also: ../../PERFORMANCE_BEST_PRACTICES.md (same content).

## Core Principles

- Preserve statistical validity (delta-method, full Σ) at all times
- Aim for constant allocations w.r.t. sample size in production code
- Preallocate once; reuse buffers and result tables
- Avoid dynamic growth in hot paths

## Patterns That Work

- Compile/cache FormulaCompiler artifacts outside loops
- Move hot loops into helpers that take concrete arguments (`compiled`, `row_buf`, `β`, `link`, `de`)
- Reuse `row_buf`, `η_buf`, `g_buf`, `gβ_accumulator`; avoid temporary vectors
- Prefer scalar loops over broadcasts that allocate
- Preallocate DataFrame columns and assign by index

## Validation

- Use `validate_allocations.jl`; rely on tags:
  - [PROD]: production code paths — must be O(1)
  - [TEST]: diagnostic loops — may allocate by design for isolation

Expected (validated): `_population_predictions`, `population_margins` (pred/effects) show constant allocations.

