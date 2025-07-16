
@testset "large data OLS (with repvals)" begin
    # simulate
    Random.seed!(42)
    n = 2_000_000
    x = randn(n)
    d = rand(n) .> 0.5       # Bool
    z = randn(n)
    # true model: y = β0 + βx x + βd d + βz z + βxd x*d + βxz x*z + βdz d*z + βxdz x*d*z + ε
    β = (β0=1.0, βx=2.0, βd= -1.5, βz=0.5, βxd=0.8, βxz=1.2, βdz=-0.7, βxdz=0.4)
    μ = β.β0 .+ β.βx*x .+ β.βd*(d .== true) .+ β.βz*z .+
        β.βxd*(x .* (d .== true)) .+ β.βxz*(x .* z) .+ β.βdz*((d .== true) .* z) .+
        β.βxdz*(x .* (d .== true) .* z)
    y = μ .+ randn(n)*0.1
    df = DataFrame(y=y, x=x, d=d, z=z)

    m = lm(@formula(y ~ x * d * z), df)
    # AME of x at d=false, z= mean(z)
    zv = mean(z)
    # closed-form derivative: ∂μ/∂x = βx + βxd*d + βxz*z + βxdz*d*z
    # at d=false => d=0: deriv = βx + βxz*zv
    ame_closed = β.βx + β.βxz*zv
    # SE via delta method: g = ∂β/∂x vector; compute var(g'β̂)
    cn = coefnames(m); coefs = coef(m); V = vcov(m)
    # build contrast c
    c = zeros(length(coefs))
    # intercept excluded
    iβx  = findfirst(isequal("x"), cn)
    iβxz = findfirst(isequal("x & z"), cn)
    c[iβx]  = 1
    c[iβxz] = zv
    se_closed = sqrt(c' * V * c)

    @time ame = margins(m, :x, df; repvals=Dict(:d => [false], :z => [zv]))

    # fails at high tolerance (above 1e-2)
    # @test isapprox(ame.effects[:x][(false, zv)], ame_closed; atol=1e-6)

    # instead use estimated coefficients
    β̂ = coef(m)
    ame_closed_est = β̂[iβx] + β̂[iβxz] * zv
    @test isapprox(ame.effects[:x][(false, zv)], ame_closed_est; atol=1e-6)

    @test isapprox(ame.ses[:x][(false, zv)], se_closed; atol=1e-6)
end

using Profile

@time ame = margins(m, :x, df);
@profile ame = margins(m, :x, df)
open("profile_output.txt", "w") do io
    Profile.print(io)
end

@time ct = columntable(df);
@time ct[:y];


@time ame = margins(m, :x, df; repvals=Dict(:d => [false], :z => [zv]))

@profile ame = margins(m, :x, df; repvals=Dict(:d => [false], :z => [zv]))


##
vars = :x
repvals = Dict{Symbol,Vector{Float64}}();
pairs = :allpairs;
type  = :dydx;

type  ∈ (:dydx, :predict)  || throw(ArgumentError("`type` must be :dydx or :predict, got `$type`"))
pairs ∈ (:allpairs, :baseline) || throw(ArgumentError("`pairs` must be :allpairs or :baseline, got `$pairs`"))

varlist = isa(vars, Symbol) ? [vars] : collect(vars)
for v in varlist
    v ∈ Symbol.(names(df)) || throw(ArgumentError("Variable $v not found in data"))
end

model = m;

using Margins: cholesky, get_invlink, get_link_derivatives, is_continuous_variable

n   = nrow(df)
β   = coef(model)
@time Σβ  = vcov(model)
@time cholΣβ = cholesky(Σβ)
invlink, dinvlink, d2invlink = get_invlink(model), get_link_derivatives(model)...

@time data_nt = Tables.columntable(df);

# Classify variables by type
iscts(v) = is_continuous_variable(v, data_nt)
cts_vars = filter(iscts, varlist)
cat_vars = setdiff(varlist, cts_vars)

using Margins: LeanAMEWorkspace

@time ws = LeanAMEWorkspace(model, df);
result_map = Dict{Symbol,Any}()
se_map = Dict{Symbol,Any}()
grad_map = Dict{Symbol,Any}()

# compute_standard_ames_lean!
overrides = Dict{Symbol,Any}()
# ames, ses, grads = compute_continuous_ames_lean!(cts_vars, model, ws, β, cholΣβ,
# dinvlink, d2invlink; overrides=overrides)

variables = cts_vars

k = length(variables)
ames = Vector{Float64}(undef, k)
ses = Vector{Float64}(undef, k)
grads = Vector{Vector{Float64}}(undef, k)

#     ame, se, grad = compute_continuous_ame_lean!(
#     variable, model, ws, β, cholΣβ, dinvlink, d2invlink; overrides=overrides
# )

n = length(first(ws.data))
p = length(β)

# Initialize accumulators
sum_ame = 0.0
@time fill!(ws.grad_work, 0.0)

using Margins: modelrow!, fixed_effects_form

model = m
rhs = fixed_effects_form(model).rhs;
expected_width = width(rhs);
row_vec = ws.row_vec;
@assert expected_width == length(row_vec) "Pre-allocated X has wrong number of columns ($(length(row_vec))), expected $expected_width)"

data = columntable(df);

# check access situation under different types
@time data[:y];
@time df[!, :y];

uu = deepcopy(df[!, :y]);
@time uu;
uuu = Dict(:y => uu);
@time uuu;

using FormulaCompiler
using Profile
row_vec .= 0.0

@time _cols!(row_vec, rhs, data)
@profile [_cols!(row_vec, rhs, data) for _ in 1:10]

using BenchmarkTools
@btime _cols!(row_vec, rhs, data)

@time rhs_c = collect(rhs.terms);
row_vec .= 0.0

@time _cols!(row_vec, rhs_c, data);

@time _cols!(row_vec, rhs.terms, data);

@code_warntype _cols!(row_vec, rhs_c, data);

@profile _cols!(row_vec, rhs_c, data);
open("profile_output.txt", "w") do io
    Profile.print(io)
end

@btime _cols!(row_vec, rhs_c, data);

##

import FormulaCompiler._cols!
# Or: Recursive unrolling (more elegant)
@inline function _cols!(row_vec, terms::Tuple, data, pos=1)
    return _cols_unroll!(row_vec, terms, data, pos)
end

@inline _cols_unroll!(row_vec, ::Tuple{}, data, pos) = pos

@inline function _cols_unroll!(row_vec, terms::Tuple, data, pos)
    new_pos = _cols!(row_vec, first(terms), data, pos)
    return _cols_unroll!(row_vec, Base.tail(terms), data, new_pos)
end

rhs_c = collect(rhs.terms);
row_vec .= 0.0
@btime _cols!(row_vec, rhs_c, data);

##

@time modelrow!(rhs, expected_width, ws.row_vec, model, ws.data, i; overrides=overrides);
@profile modelrow!(ws.row_vec, model, ws.data, i; overrides=overrides);
open("profile_output.txt", "w") do io
    Profile.print(io)
end

##
i=1
for i in 1:n
    # Fill model matrix row for observation i with overrides applied (reuses ws.row_vec)
    @time modelrow!(ws.row_vec, model, ws.data, i; overrides=overrides)
    η_i = dot(ws.row_vec, β)
    
    # Fill analytical derivative row for observation i w.r.t. variable with overrides (reuses ws.deriv_vec)
    modelrow_derivative!(ws.deriv_vec, model, ws.data, i, variable; overrides=overrides)
    dη_dx_i = dot(ws.deriv_vec, β)
    
    # Compute marginal effect for this observation
    if isfinite(η_i) && isfinite(dη_dx_i)
        μp_i = dinvlink(η_i)
        marginal_effect_i = μp_i * dη_dx_i
        
        if isfinite(marginal_effect_i) && abs(marginal_effect_i) < 1e10
            sum_ame += marginal_effect_i
            
            # Gradient computation for this observation
            # ∂AME/∂β = (1/n) * Σᵢ [∇(marginal_effect_i)]
            # where ∇(marginal_effect_i) = ∇(μ'(ηᵢ) * dη_dx_i)
            μpp_i = d2invlink(η_i)
            
            @inbounds for j in 1:p
                # Using product rule: ∂(μ'(ηᵢ) * dη_dx_i)/∂βⱼ
                # = μ''(ηᵢ) * ∂ηᵢ/∂βⱼ * dη_dx_i + μ'(ηᵢ) * ∂(dη_dx_i)/∂βⱼ
                # = μ''(ηᵢ) * xᵢⱼ * dη_dx_i + μ'(ηᵢ) * deriv_vec[j]
                
                term1 = μpp_i * ws.row_vec[j] * dη_dx_i  # Second-order term
                term2 = μp_i * ws.deriv_vec[j]           # First-order term
                ws.grad_work[j] += (term1 + term2) / n
            end
        end
    end
end
