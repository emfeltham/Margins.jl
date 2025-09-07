Margins Compute Rewrite Plan — Correct AME + Scenarios

- Overwrite existing functions, do not create new names unless it is a genuinely new function
- Both statistical correctness and efficiency are _paramount_.

Overview
- Goal: Implement Average Marginal Effects (AME) and scenario analysis with unified counterfactual computation infrastructure.
- Key insight: **AME IS a counterfactual scenario** - both require override system for statistical correctness.
- Performance: Zero allocations per row after warmup for all per‑row kernels, compute‑only vs presentation separation.

Conventions & Scope
- Mixed models: Fixed-effects only (random effects set to 0). We do not marginalize over random effects.
- Scale handling: Link vs response computation and chain rules are delegated to FormulaCompiler.
- Predictor scaling: Mapping between standardized/original scales is handled in FormulaCompiler evaluators.
- Categorical/Boolean contrasts: Always use baseline→level contrasts. For booleans, baseline is 0/false and contrast is false→true.

Part 0 — Override System Foundation (Priority First)

(Improve the existing override system in FormulaCompiler)

Goals
- Establish a unified, type-stable override infrastructure used by AME analysis (and scenarios).
- Guarantee 0 allocations per row after warmup when evaluating `modelrow!` with overrides.
- Enforce baseline determination for categorical/boolean variables and rebuild full design rows (including interactions) under overrides.

API & Invariants
- Scenarios: thin wrappers over FormulaCompiler’s `OverrideVector` for per-row overrides.
- Baselines: infer categorical baselines from model coding; booleans use `false→true` (0→1).
- Row rebuild: overrides must trigger recomputation of the design row (and η) consistent with interactions and contrasts.
- Mixed models: fixed-effects only; overrides do not alter random-effects contributions (treated as 0).
- Scale/predictor handling: link/response and standardized/original mapping delegated to FormulaCompiler.

Tests (Micro + Integration)
- `modelrow!` with overrides: minimum(memory) == 0 after warmup.
- Contrast coding invariance: baseline→level contrasts identical across dummy/effects/helmert.
- Interactions: correctness for terms like `x + x^2 + x*z` under overrides.
- Boolean consistency: `false→true` equals `0→1`.
- Edge stability: large |η| cases with stable `dμ/dη` evaluation (delegated to FormulaCompiler, verified here).


Part I — Correct AME (Counterfactual Computation)

1) Mathematical Targets
- Binary d ∈ {0,1}:
  AME = (1/n) Σ_i [ m(x_i, d=1) − m(x_i, d=0) ]
- Categorical c with baseline b and level k:
  AME(b→k) = (1/n) Σ_i [ m(x_i, c=k) − m(x_i, c=b) ]
- Continuous x:
  AME = (1/n) Σ_i ∂m(x_i)/∂x (derivative-based)

Weighted averages (optional weights w_i ≥ 0):
- Replace unweighted means by weighted means with normalization by Σ_i w_i:
  • AME_w = (1/Σ_i w_i) Σ_i w_i [ m_to(x_i) − m_from(x_i) ]
  • AME_w(cont) = (1/Σ_i w_i) Σ_i w_i ∂m(x_i)/∂x
  • Gradients: g_AME_w = (1/Σ_i w_i) Σ_i w_i g_i (same structure for link/response; response chain handled in FC)

**CRITICAL**: For categorical/binary, this is a **counterfactual contrast** requiring overrides:
1. Set focal variable to level A for ALL observations → predict ŷₐ
2. Set focal variable to level B for ALL observations → predict ŷᵦ  
3. AME = mean(ŷₐ - ŷᵦ)

"Over the sample distribution" means non-target covariates remain at observed values x_i while focal variable is artificially manipulated for everyone.

2) Core Compute (Unified Override System)
- Categorical/Binary AME (counterfactual contrasts via overrides):
  - Create two override scenarios: focal variable = level A, focal variable = level B
  - For each observation i, compute predictions under both scenarios:
    • ŷₐᵢ = model_predict(xᵢ with focal variable overridden to A)  
    • ŷᵦᵢ = model_predict(xᵢ with focal variable overridden to B)
  - Per-row contribution: Δᵢ = ŷₐᵢ - ŷᵦᵢ
  - AME = mean(Δᵢ) = (1/n) Σᵢ Δᵢ
  - Implementation: Use FormulaCompiler's override system with modelrow! for zero-allocation prediction
  - **Statistical correctness**: Only contrast-then-average is valid for response scale (nonlinear link)
  - Gradients:
    - Link scale: ∂(Δη_i)/∂β = X_to_i - X_from_i (difference in design matrices)
    - Response scale: handled via FormulaCompiler chain rules using per-scenario η values
      • Use override system to generate X_to_i and X_from_i for each observation
      • Accumulate gradients across all observations for delta-method SE computation

- Continuous AME (derivative):
  - Use FormulaCompiler’s derivative evaluators:
    • marginal_effects_eta!/mu!(g, de, β, row; backend=:ad)
    • Per‑row AD is zero‑allocation after warmup (derivative_allocations.csv)
  - Gradients:
    • Analytical, in‑place: me_eta_grad_beta! / me_mu_grad_beta! (zero‑alloc)
    • Or use an AD‑zero accumulate path if needed.

3) API & Structure
- Compute‑only, function-level API (internal):
  - ame_categorical!(out, engine, rows, var, level_to; scale=:response|:link, weights=nothing)
    • Baseline→level contrast: level_from is inferred as the baseline (reference) level
    • Uses override system to create counterfactual scenarios per observation
    • Calls modelrow! with overridden data for zero-allocation prediction
  - ame_continuous!(out, engine, rows, var; scale, weights=nothing)
    • Uses AD evaluator for derivative-based marginal effects
  - Gradients: ame_categorical_grad_beta!, ame_continuous_grad_beta! using override-based design matrices
- Public wrapper (population_margins): Prepare → compute → finalize.
  - Prepare: compile formula, build override scenarios, allocate buffers
  - Compute: call unified AME functions using override system; zero‑allocation per row after warmup
    • If `weights` provided (or discoverable from model), use weighted aggregation with normalization by Σ_i w_i
  - Finalize: build DataFrame (presentation only; not part of compute allocations)

Scenario kwarg (Stata `at()` equivalent; population analysis)
- `population_margins(model, data; scenario=nothing, ...)` accepts `scenario` specifying representative covariate values/mixtures:
  • Scalars for continuous (e.g., `x=5.0`), booleans (`t=true`), categorical levels (`g="B"`), or mixtures (`g=mix("A"=>0.3,"B"=>0.7)`).
  • Special tokens may be supported by helpers (e.g., `:mean`, `:median`) via a precompute step that maps to scalars before compute.
- Semantics: scenario fixes only the user‑specified covariates to the representative values for EVERY observation (counterfactual); all other covariates remain at their observed values. Compute effects/predictions under this partial override and average across the sample (AME at representative values).
- **With `vars` parameter**: scenario provides conditioning values while `vars` specifies focal variables for AME computation (equivalent to Stata's `margins, dydx(vars) at(scenario)`).
- `profile_margins` does not take `scenario`; it accepts explicit reference grids.

4) Performance & Allocations (Requirements)
- Per‑row kernels (continuous AD, categorical/binary overrides): minimum(memory) == 0 after warmup.
- Override system uses FormulaCompiler's efficient OverrideVector with O(1) memory per scenario.
- No broadcast, no copy(row) in hot loops; only in‑place modelrow! calls and BLAS operations.
- Compute‑only normalized totals: bytes‑per‑row == 0 across sizes; presentation layer excluded.

Performance Notes
- Default override path: use FormulaCompiler `OverrideVector`, reused per row/per thread, mutating values in place; zero allocations after warmup.
- Avoid per-call NamedTuple/keyword overrides (e.g., `overrides=(x=...)`) due to allocation/dispatch overhead; reserve only for debugging.
- Temporary mutation of data columns is NOT a public path; if used internally, restrict to private, single-thread micro-optimizations with immediate restore and dedicated tests. Beware categorical levels/interactions.
- **Threading scope**: Single-threaded implementation for initial version. Design APIs to avoid shared mutable state that would preclude future multithreading, but don't implement threading infrastructure yet.
- Benchmarking: include microbenches comparing OverrideVector reuse vs temporary mutation vs per-call NamedTuple to document the chosen default.

5) Tests (Unified in test/test_ame_alloc.jl)
- Continuous (AD): per‑row η/μ marginal effects and per‑row gradients; assert zero alloc.
- Categorical/Binary: per‑row counterfactual contrasts via override system; assert zero alloc.
- Mixed: continuous + categorical per‑row kernels tested separately in the same model.
- Statistical correctness: AME results match manual counterfactual computation to machine precision.
- Weighted correctness: weighted AME equals manual weighted contrast/derivative with normalization; weighted gradient equals weighted average of per-row gradients.
- Isolation microbenches:
  • Pure modelrow! with override scenarios == 0 bytes
  • Pure per‑row helper (link/response) == 0 bytes

Additional Statistical Tests
- Baseline invariance across contrast codings (dummy/effects/helmert): b→k contrasts identical.
- Boolean baseline contrast consistency: false→true equals 0→1 contrast.
- MixedModels fixed-effects-only: results match predictions with RE set to 0.

Part II — Profile Analysis (Extended Override Applications)

CHECK DO WE STILL WANT THIS??

1) Purpose
- Profile/scenario workflows extend AME's counterfactual computation to custom reference points.
- Same override system as AME, but evaluated at specific covariate combinations rather than sample average.
- Examples: "marginal effects at the mean", "predictions when x=5, z=10"

2) Unified Override Infrastructure
- Same FormulaCompiler OverrideVector system used for both AME and profile analysis.
- DataScenario design (if retained) must be parameterized for type stability:
  mutable struct DataScenario{NT<:NamedTuple}
    name::String
    overrides::Dict{Symbol,Any}
    data::NT
    original_data::NT
  end
- **Key insight**: AME is just a special case of profile analysis where the "profile" is the entire sample.

3) Profile Compute
- Same counterfactual computation as AME, but at specified reference points rather than sample average.
- For continuous variables: evaluate derivatives at reference point rather than averaging across sample.
- For categorical variables: same counterfactual contrast approach as AME.
- Reuse all AME computational kernels with different aggregation/averaging strategy.

4) API Unification
- `population_margins(; scenario=...)`: AME using override system across entire sample, with optional representative‑value scenarios
- `profile_margins()`: Same override system, evaluated at explicitly provided reference grids
 - **No conceptual separation**: Both use identical counterfactual computation infrastructure

5) Tests
- Profile analysis inherits all AME allocation and correctness tests.
- Additional tests for reference grid specification and custom override scenarios.
- Same zero-allocation requirements: modelrow! with overrides == 0 bytes per row.
 - Scenario kwarg (population): scalar overrides, boolean, categorical level, and mixture cases; equivalence to manual counterfactual modification and averaging.

Implementation Roadmap
1) Override System Foundation (priority)
   - [ ] Implement/confirm thin scenario wrappers over FormulaCompiler `OverrideVector`
   - [ ] Guarantee zero-allocation `modelrow!` with overrides (microbench + tests)
   - [ ] Baseline inference for categoricals; boolean convention `false→true (0→1)`
   - [ ] Full row rebuild under overrides (interactions, transforms, contrasts)
   - [ ] Contrast-coding invariance tests (dummy/effects/helmert)
2) Unified AME with override system
   - [ ] Implement `ame_categorical!` using baseline→level counterfactual contrasts
   - [ ] Use FormulaCompiler derivative evaluators for continuous AME
   - [ ] Implement analytical gradients using override-generated design matrices
3) [ ] Update `population_margins()` to use override-based compute for categorical and continuous
4) [ ] Add `scenario` kwarg parsing to `population_margins()` → build representative-value overrides; keep `profile_margins()` grid-based
5) Tests
   - [ ] Consolidate allocation tests (continuous AD, categorical overrides, mixed)
   - [ ] Statistical correctness vs manual counterfactuals (including baseline contrasts)
   - [ ] Explicit `modelrow!` override isolation checks (zero allocation)
6) [ ] Documentation: reflect unified override-first computation approach

Acceptance Criteria
- [ ] Pure `modelrow!` with overrides: 0 bytes after warmup
- [ ] Override invariants: baseline→level contrasts, boolean `false→true (0→1)`
- [ ] Contrast-coding invariance for categoricals (dummy/effects/helmert)
- [ ] Interactions/transforms correct under overrides (row rebuild verified)
- [ ] Mixed models: fixed-effects-only estimand (RE=0)
- [ ] Per‑row kernels (continuous AD, categorical/binary overrides) == 0 bytes after warmup across sizes
- [ ] AME matches manual counterfactual computation (contrast-then-average on response scale)
- [ ] Profile analysis reuses AME kernels with zero additional allocation
 - [ ] `scenario` kwarg for `population_margins` fixes covariates for all rows and averages; matches manual counterfactuals; zero allocations preserved
