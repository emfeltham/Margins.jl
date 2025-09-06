Margins Compute Rewrite Plan — Correct AME + Scenarios

Overview
- Goal: One coherent plan that (1) implements Average Marginal Effects (AME) the mathematically correct way without overrides and (2) supports counterfactual “scenario/at” workflows via overrides as a separate, opt‑in layer.
- Performance: Zero allocations per row after warmup for all per‑row kernels (continuous AD evaluator, categorical/binary contrasts), compute‑only vs presentation separation.

Part I — Correct AME (No Overrides in Core)

1) Mathematical Targets
- Binary d ∈ {0,1}:
  AME = (1/n) Σ_i [ m(x_i, d=1) − m(x_i, d=0) ]
- Categorical c with baseline b and level k:
  AME(b→k) = (1/n) Σ_i [ m(x_i, c=k) − m(x_i, c=b) ]
- Continuous x:
  AME = (1/n) Σ_i ∂m(x_i)/∂x (derivative-based)

Key: “Over the sample distribution” means all non‑target covariates are held at each row’s observed values x_i. For categoricals/binary this is a discrete contrast; for continuous this is a derivative.

2) Core Compute (No Overrides)
- Categorical/Binary AME (stratified averaging via contrasts):
  - Use FormulaCompiler’s compiled formula and contrasts to evaluate per‑row changes with no data mutation.
  - Use FC’s contrast_modelrow! to compute ΔX_row = X_to − X_from at row i. Then:
    η_from = dot(X_from, β)
    η_to   = η_from + dot(ΔX_row, β)
    - Link scale AME contribution: Δη_i = dot(ΔX_row, β)
    - Response scale AME contribution: μ_to − μ_from = linkinv(η_to) − linkinv(η_from)
  - Accumulate per‑row contributions and average at the end (either contrast‑then‑average or average‑then‑contrast; both are identical by linearity).
  - Gradients:
    - Link scale: ∂(Δη)/∂β = ΔX_row (in‑place copy to gradient buffer)
    - Response scale: ∂(μ_to − μ_from)/∂β = (dμ/dη|η_to) X_to − (dμ/dη|η_from) X_from
      • Obtain X_from and X_to without overrides: X_to = X_from + ΔX_row.
      • Use FC’s non‑allocating link derivative utilities (e.g., _dmu_deta) and in‑place loops.

- Continuous AME (derivative):
  - Use FormulaCompiler’s AD derivative evaluator:
    • marginal_effects_eta!/mu!(g, de, β, row; backend=:ad)
    • Per‑row AD is zero‑allocation after warmup (derivative_allocations.csv)
  - Gradients:
    • Analytical, in‑place: me_eta_grad_beta! / me_mu_grad_beta! (zero‑alloc)
    • Or use an AD‑zero accumulate path if needed.

3) API & Structure
- Compute‑only, function-level API (internal):
  - ame_categorical!(out, compiled, β, link, rows, var, level_from, level_to; scale=:response|:link)
    • Uses contrast_modelrow! per row with in‑place buffers.
  - ame_continuous!(out, de, β, link, rows; scale)
    • Uses marginal_effects_* with AD evaluator.
  - Gradients: ame_categorical_grad_beta!, ame_continuous_grad_beta! using the formulas above.
- Public wrapper (population_margins): Prepare → compute → finalize.
  - Prepare: compile once, build AD evaluator (continuous only), allocate buffers.
  - Compute: call the no‑override AME functions; zero‑allocation per row after warmup.
  - Finalize: build DataFrame (presentation only; not part of compute allocations).

4) Performance & Allocations (Requirements)
- Per‑row kernels (continuous AD, categorical/binary contrasts): minimum(memory) == 0 after warmup.
- No overrides, no broadcast, no copy(row) in hot loops; only in‑place loops and BLAS dot.
- Compute‑only normalized totals: bytes‑per‑row == 0 across sizes; presentation layer excluded.

5) Tests (Unified in test/test_ame_alloc.jl)
- Continuous (AD): per‑row η/μ marginal effects and per‑row gradients; assert zero alloc.
- Categorical/Binary: per‑row contrast contributions and gradients (using contrast_modelrow!), assert zero alloc.
- Mixed: continuous + categorical per‑row kernels tested separately in the same model.
- Isolation microbenches:
  • Pure modelrow! (categorical case) == 0 bytes
  • Pure per‑row helper (link/response) == 0 bytes

Part II — Scenarios / “at” (Overrides as a Separate Layer)

1) Purpose
- “Scenario/at” workflows answer counterfactual questions (“if everyone were at x=c”, “if everyone were in group A”) separate from AME.
- Implemented via overrides (OverrideVector) with O(1) memory, evaluated with modelrow! per row.

2) DataScenario Design
- Parameterize to avoid dynamic dispatch in hot paths:
  mutable struct DataScenario{NT<:NamedTuple}
    name::String
    overrides::Dict{Symbol,Any}
    data::NT
    original_data::NT
  end
- Constructors must return DataScenario{typeof(modified_data)} so `scenario.data` has a concrete type.
- Categorical overrides:
  • create_categorical_override pre‑normalizes pooled CategoricalValue at construction (no per‑row normalization).

3) Scenario Compute
- Evaluate per‑row with modelrow!(row_buf, compiled, scenario.data, row) — zero‑allocation path.
- For scenario differences (profile contrasts), reuse the in‑place per‑row gradient/prediction helpers from Part I.

4) API
- Public keyword `scenario`/`at` in population_margins/profile_margins uses DataScenario internally.
- AME code paths in Part I do not use overrides; scenarios are only for explicit “at(...)” workflows.

5) Tests
- Scenario-only correctness and allocation tests:
  • Pure modelrow! with DataScenario.data == 0 bytes
  • Profile contrasts and scenario differences: zero‑allocation per row after warmup.

Implementation Roadmap
1) Refactor AME (no overrides):
   - [ ] Add contrast-based categorical/binary AME functions using contrast_modelrow! and in‑place loops; add analytical gradient functions.
   - [ ] Use AD evaluator for continuous per‑row AME and analytical gradients; ensure zero alloc per row.
2) [ ] Split population_margins into prepare → compute → finalize; route AME to no‑override compute.
3) [ ] Parameterize DataScenario; update create_scenario constructors to preserve concrete NamedTuple types; keep categorical overrides pooled.
4) Update tests:
   - [ ] Consolidate per‑row alloc tests (continuous AD, categorical/binary contrasts, mixed); add explicit modelrow! isolation checks.
   - [ ] Add scenario/at tests separately (zero per row).
5) [ ] Docs: replace older plans; keep one source of truth.

Acceptance Criteria
- [ ] Per‑row kernels (continuous AD, categorical/binary contrasts) == 0 bytes after warmup across sizes.
- [ ] Pure modelrow! with DataScenario.data == 0 bytes.
- [ ] AME results match contrast‑then‑average and average‑then‑contrast to machine precision.
- [ ] Scenario/at workflows correct and zero‑alloc per row.
