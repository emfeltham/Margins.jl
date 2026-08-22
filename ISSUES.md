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

The bin/pin distinction already exists structurally; nothing asks the question. Binning
groups store a NamedTuple, pinning groups store a raw level:

- `_create_quantile_groups` and `_create_threshold_groups` store
  `var => (lower=, upper=, label=, bin=)` plus a companion `Symbol("$(var)_bin") => i` key.
- `_parse_over_specification` stores a raw level from `unique(data_nt[var])` for a categorical.
- `_create_context_data_with_weights` already branches on exactly that shape, and **groups
  only ever subset rows** — they never override column values, unlike `scenarios`. So within a
  bin the context rows still carry their real `x`, and `_compute_all_continuous_context_effects_batch`
  would average the derivative over precisely the bin's rows. Nothing downstream needs changing.

The two places that only test `haskey` are the skip rule (`src/population/contexts.jl:54`) and
`_validate_vars_groups_overlap` (`src/population/core.jl`). The skip rule is the easy half —
add a `_group_spec_is_bin` predicate (better: an explicit `kind=:bin` field on the NamedTuples
rather than shape-sniffing) and consult it there.

The validator is the harder half. It runs on the raw user syntax via `_extract_group_variables`,
which flattens everything to bare Symbols with no spec retained, and it is called before
`data_nt` is built (`src/population/core.jl:136` vs `:148`). It needs a parallel walker that
preserves per-variable kind through `Pair` and `Vector` nesting — and that walker cannot decide
from syntax alone, because `_parse_over_specification` turns a bare `groups=:x` on a
*continuous* `x` into auto-quartiles, making that a binned request too. So either move the call
after the data conversion or pass it the raw data.

Test fallout: `test/core/test_errors.jl` currently asserts `ArgumentError` for exactly the
combinations this would allow, and the empty-result guard's `MarginsError` case (a model whose
only variable is `x`, called with `groups=(:x, 4)`) would start succeeding. A correctness test
must confirm the within-bin average matches a manual `mean(∂y/∂x)` over that bin's rows, that
the reported `n` matches, and that the half-open convention (bin 1 is `[lo, hi)`, bins >= 2 are
`(lo, hi]`) partitions the sample exactly.

Related dead code found while investigating: `_create_continuous_subgroups` and its
`(center=, indices=, label=)` value shape are unreachable — the only route to them is
`_parse_over_specification`'s `NamedTuple` branch, which `_parse_groups_specification` never
takes and `_validate_groups_parameter` rejects.

Workaround, documented in `docs/src/stata_migration.md` and `docs/src/examples.md`: bin `x`
into a derived column and group by that, which the conflict rule allows because the grouping
variable is no longer `x`.

```julia
df.x_bin = cut(df.x, 4)
population_margins(m, df; type=:effects, vars=[:x], groups=:x_bin)
```
