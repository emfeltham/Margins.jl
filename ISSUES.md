# KNOWN ISSUES

Style note:
- markdown checklist
- no icons

Confirmed bugs that are not yet fixed. Each entry records a reproducer and the code path
responsible, so the next person does not have to rediscover it. Fixed issues move out of
this file (into `CHANGELOG.md`).

## Profile categorical contrasts are unreachable

- [ ] Make row-specific categorical contrasts requestable from `profile_margins`

The filter in `_mem_continuous_and_categorical` (`src/profile/core.jl`, the
`categorical_requested` loop) drops a categorical variable from the contrast list if it is
a `CategoricalMixture` **or** if it appears in the original (pre-completion) reference grid.
Reference-grid completion fills any absent categorical with a mixture. Between the two
rules there is no way to request a categorical contrast: naming the variable in the grid
makes it "scenario-defining", and leaving it out makes it a mixture.

Every route tried returns 0 rows plus an `@info "Skipping contrasts for variable …"`:
`means_grid`, `balanced_grid(df; g=:all)`, an explicit `cartesian_grid` with discrete
levels, a hand-built `DataFrame` with a `CategoricalVector` column, and `Bool` variables
(whose typical value completes to a `Float64`, tripping the mixture check).

Consequences:
- `_compute_row_specific_baseline_contrast`, `_row_specific_contrast_grad_beta!` and
  `_compute_profile_pairwise_contrasts` (`src/profile/categorical_effects.jl`) are dead
  code today.
- `test/test_profile_pairwise.jl` asserts `nrow(df_result) == 3` for exactly this case and
  is **not** included in `test/runtests.jl` — it would fail.
- The row-specific baseline contrast behaviour documented in `CLAUDE.md` ("Profile margins
  use novel row-specific baseline contrasts") is not currently available to users.
- `notes/FUTURE.md` describes `_compute_profile_pairwise_contrasts()` as "operational".

Fix sketch: separate "this categorical is the analysis target" from "this categorical
defines the evaluation context". A variable named in `vars=` should be contrasted at the
grid rows even when it also appears in the grid — that is precisely the row-specific
contrast semantics (contrast the row's level against baseline at the row's covariates).
Once reachable, re-enable `test/test_profile_pairwise.jl` in `test/runtests.jl` and add a
non-identity-link test asserting the link-scale contrast equals the coefficient (the
gradient side of this was fixed alongside the `scale=:link` work).

## Grouping by the variable being contrasted crashes

- [ ] Return an empty result or a clear error instead of an `ArgumentError` from DataFrames

`groups=:g` combined with `vars=[:g]` produces zero contrast pairs per group (each group
holds a single level), and the empty per-context DataFrame then fails column lookup:

```julia
population_margins(m, df; type=:effects, vars=[:g], groups=:g)
# ERROR: ArgumentError: column name "estimate" not found in the data frame
#        since it has no columns
# at src/population/contexts.jl:153
```

Reproduces on both `scale=:response` and `scale=:link`, and on `lm` as well as `glm`, so it
is independent of the scale handling. The 2×2 framework docs say a variable appearing in
both `vars` and `groups` should have the specification ignored for its own effect — the
implementation instead produces an empty frame and then trips over it.
