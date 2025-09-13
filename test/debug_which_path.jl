# Computational pathway analysis for marginal effects estimation
using Margins, GLM, DataFrames, Tables

@info "Computational Pathway Analysis for Marginal Effects Estimation"

# Create test data like the allocation test
n_rows = 1000
df = DataFrame(
    y = randn(n_rows),
    x = randn(n_rows)
)
model = lm(@formula(y ~ x), df)

@info "Executing population_margins with sample size n = $n_rows"

# Systematic analysis of computational pathway selection during execution
result = population_margins(model, df; type=:effects, vars=[:x])

@info "Population margins computation completed successfully"
@info "Marginal effect estimate: $(result.estimates[1])"
@info "Delta-method standard error: $(result.standard_errors[1])"

# Manual verification of computational pathway selection
data_nt = Tables.columntable(df)
engine = Margins.get_or_build_engine(model, data_nt, [:x], GLM.vcov)

@info "Direct validation of variable-specific AME computation"
var_type = Margins._detect_variable_type(engine.data_nt, :x)
@info "Variable classification: $var_type"

rows = 1:n_rows
scale = :response
backend = :ad

if var_type === :continuous
    @info "Routing to continuous variable AME computation"
    ame_val, gradient = Margins._compute_continuous_ame(engine, :x, rows, scale, backend)
    @info "Direct computation result - AME estimate: $ame_val, Gradient vector norm: $(sum(abs.(gradient)))"
end