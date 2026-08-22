# KNOWN ISSUES

Style note:
- markdown checklist
- no icons

Confirmed bugs that are not yet fixed. Each entry records a reproducer and the code path
responsible, so the next person does not have to rediscover it. Fixed issues move out of
this file (into `CHANGELOG.md`).

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

## Orphaned profile helper family

- [ ] Remove or wire up `ProfileContext` and friends

`src/profile/continuous_effects.jl` defines `ProfileContext`, `create_profile_context` and
`apply_profile_measure_transformations`. Nothing calls them: `profile_margins` computes
effects in `_mem_continuous_and_categorical` (`src/profile/core.jl`) and applies measures
with `apply_measure_transformation`. They are also missing the quotient-rule correction that
the live path applies, so they would produce wrong elasticity SEs if revived. Left in place
(with a note in their docstrings) rather than deleted alongside the profile-contrast fix, to
keep that change reviewable.
