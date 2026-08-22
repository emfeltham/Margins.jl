# KNOWN ISSUES

Style note:
- markdown checklist
- no icons

Confirmed bugs that are not yet fixed. Each entry records a reproducer and the code path
responsible, so the next person does not have to rediscover it. Fixed issues move out of
this file (into `CHANGELOG.md`).

## Effects within bins of the variable being differentiated are rejected

- [ ] Let `groups=(:x, 4)` coexist with `vars=[:x]` by distinguishing binning from pinning

`population_margins(m, df; type=:effects, vars=[:x], groups=(:x, 4))` raises the vars/groups
conflict error, but the quantity it asks for is well defined: quantile grouping partitions
the sample by `x` without pinning it, so `x` still varies inside every bin and `∂y/∂x`
averaged within a quartile is exactly Stata's `margins, dydx(x) over(xbin)`.

The skip test at `src/population/contexts.jl:52` asks only `haskey(group_spec, var)`, which
cannot tell a group that *pins* a variable (a categorical level, an explicit value list) from
one that merely *bins* it. Supporting the binned case means threading that distinction from
`_create_quantile_groups` / `_create_threshold_groups` (`src/population/contexts.jl`) through
to the skip test, and relaxing `_validate_vars_groups_overlap` (`src/population/core.jl`)
to match. Correctness tests would need to confirm the within-bin average matches a manual
computation over the bin's rows.

Workaround, documented in `docs/src/stata_migration.md` and `docs/src/examples.md`: bin `x`
into a derived column and group by that, which the conflict rule allows because the grouping
variable is no longer `x`.

```julia
df.x_bin = cut(df.x, 4)
population_margins(m, df; type=:effects, vars=[:x], groups=:x_bin)
```

## Orphaned profile helper family

- [ ] Remove or wire up `ProfileContext` and friends

`src/profile/continuous_effects.jl` defines `ProfileContext`, `create_profile_context` and
`apply_profile_measure_transformations`. Nothing calls them: `profile_margins` computes
effects in `_mem_continuous_and_categorical` (`src/profile/core.jl`) and applies measures
with `apply_measure_transformation`. They are also missing the quotient-rule correction that
the live path applies, so they would produce wrong elasticity SEs if revived. Left in place
(with a note in their docstrings) rather than deleted alongside the profile-contrast fix, to
keep that change reviewable.
