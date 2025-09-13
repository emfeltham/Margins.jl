# Running Subsets of the Test Suite

This project’s `runtests.jl` supports running only specific test files via `Pkg.test(; test_args=...)` or by executing test files directly. This is handy when iterating on a focused area (e.g., scenarios/groups).

## Using Pkg.test with test_args

`runtests.jl` reads `ARGS` and, when non-empty, includes only those files and skips the full suite.

Examples
- Run two files:
  ```bash
  julia --project=. -e 'using Pkg; Pkg.test(; test_args=["test/statistical_validation/population_context_bootstrap_validation.jl","test/validation/test_population_scenarios_groups.jl"])'
  ```
- Paths can be provided with or without the leading `test/` prefix; both work:
  ```bash
  julia --project=. -e 'using Pkg; Pkg.test(; test_args=["statistical_validation/population_context_bootstrap_validation.jl","validation/test_population_scenarios_groups.jl"])'
  ```

Notes
- The harness strips whitespace/newlines from each argument and normalizes the `test/` prefix if present.
- If you pass multiple files that internally include the same utilities, you may see duplicate method definition warnings — these are benign in this mode.

## Running a single file directly

You can run a single test file without the Pkg test harness:
```bash
julia --project=. --color=yes --startup-file=no test/statistical_validation/population_context_bootstrap_validation.jl
```

This executes just that file in isolation. Use this for quick checks when the file doesn’t rely on the grouped initialization in `runtests.jl`.

## Tips
- Redirect output to a file for later inspection:
  ```bash
  julia --project=. -e 'using Pkg; Pkg.test(; test_args=["test/validation/test_population_scenarios_groups.jl"])' > test/output.txt 2>&1
  ```
- When invoking from a shell that wraps arguments across lines, ensure your quotes are correct; `runtests.jl` also strips stray newlines just in case.

