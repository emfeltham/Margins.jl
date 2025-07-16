# three-way debug.jl
# Add this debug code to your test to investigate the issue

@testset "3-way interaction DEBUG" begin
    Random.seed!(42)  # Same seed for consistency
    n = 500
    x = randn(n)
    d = rand(n) .> 0.5       
    z = randn(n)
    β = (β0=1.0, βx=2.0, βd= -1.5, βz=0.5, βxd=0.8, βxz=1.2, βdz=-0.7, βxdz=0.4)
    μ = β.β0 .+ β.βx*x .+ β.βd*(d .== true) .+ β.βz*z .+
        β.βxd*(x .* (d .== true)) .+ β.βxz*(x .* z) .+ β.βdz*((d .== true) .* z) .+
        β.βxdz*(x .* (d .== true) .* z)
    y = μ .+ randn(n)*0.1
    df = DataFrame(y=y, x=x, d=CategoricalArray(d), z=z)

    m = lm(@formula(y ~ x * d * z), df)
    zv = mean(z)
    
    # Debug: Check model coefficients
    println("Model coefficients:")
    for (i, name) in enumerate(coefnames(m))
        println("  $name: $(coef(m)[i])")
    end
    
    # Debug: Check expected vs actual derivative
    cn = coefnames(m); coefs = coef(m)
    iβx  = findfirst(isequal("x"), cn)
    iβxz = findfirst(isequal("x & z"), cn)
    
    expected_ame = coefs[iβx] + coefs[iβxz] * zv
    println("Expected AME: $expected_ame")
    
    # Compute AME
    ame = margins(m, :x, df; repvals=Dict(:d => categorical([false]), :z => [zv]))
    actual_ame = ame.effects[:x][(false, zv)]
    actual_se = ame.ses[:x][(false, zv)]
    
    println("Actual AME: $actual_ame")
    println("Actual SE: $actual_se")
    println("Difference: $(abs(actual_ame - expected_ame))")
    
    # Debug: Check if SE computation failed
    if isnan(actual_se)
        println("❌ SE computation returned NaN - gradient or covariance issue")
    end
    
    # Debug: Try with regular Bool instead of CategoricalArray
    df2 = DataFrame(y=y, x=x, d=d, z=z)  # Regular Bool
    m2 = lm(@formula(y ~ x * d * z), df2)
    ame2 = margins(m2, :x, df2; repvals=Dict(:d => [false], :z => [zv]))
    
    println("With regular Bool:")
    println("  AME: $(ame2.effects[:x][(false, zv)])")
    println("  SE: $(ame2.ses[:x][(false, zv)])")
    
    # Check if the analytical derivatives are being computed correctly
    # You could add more debug output here to trace the derivative computation
end

#######

# Debug analytical derivatives computation

function debug_analytical_derivatives_detailed(model, df)
    println("🔬 DEBUGGING ANALYTICAL DERIVATIVES COMPUTATION")
    
    # Setup
    ws = AMEWorkspace(model, df)
    ipm = InplaceModeler(model, nrow(df))
    
    # Create representative values data
    zv = mean(df.z)
    repvals = Dict(:d => categorical([false]), :z => [zv])
    repvars = collect(keys(repvals))
    combo = (false, zv)
    
    n = length(first(ws.base_data))
    repval_data = create_representative_data(ws.base_data, repvars, combo, n)
    
    println("Representative values:")
    println("  d: $(repval_data[:d][1:3]) (all should be false)")
    println("  z: $(repval_data[:z][1:3]) (all should be $(zv))")
    println("  x: $(repval_data[:x][1:3]) (original values)")
    
    # Update workspace to representative values
    ws.base_data = repval_data
    original_base_matrix = copy(ws.base_matrix)
    modelmatrix_with_base!(ipm, repval_data, ws.work_matrix, original_base_matrix, repvars, ws.mapping)
    
    println("\nModel matrix at representative values (first 3 rows):")
    for i in 1:3
        println("  Row $i: $(ws.work_matrix[i, :])")
    end
    
    # Show what the interaction terms should be
    println("\nExpected interaction values at representative values:")
    println("  x values: $(repval_data[:x][1:3])")
    println("  d values (false): $(Float64.(repval_data[:d][1:3] .== true))")  # Should be [0,0,0]
    println("  z values: $(repval_data[:z][1:3])")
    println("  x*d (should be 0): $(repval_data[:x][1:3] .* Float64.(repval_data[:d][1:3] .== true))")
    println("  x*z: $(repval_data[:x][1:3] .* repval_data[:z][1:3])")
    println("  d*z (should be 0): $(Float64.(repval_data[:d][1:3] .== true) .* repval_data[:z][1:3])")
    println("  x*d*z (should be 0): $(repval_data[:x][1:3] .* Float64.(repval_data[:d][1:3] .== true) .* repval_data[:z][1:3])")
    
    # Now compute analytical derivatives
    println("\n🧮 Computing analytical derivatives for :x")
    prepare_analytical_derivatives!(ws, :x, 0.0, ipm)
    
    println("Derivative matrix (first 3 rows):")
    for i in 1:3
        println("  Row $i: $(ws.derivative_matrix[i, :])")
    end
    
    # Check which columns should be non-zero for x derivatives
    println("\nColumn analysis:")
    cn = coefnames(model)
    for (j, name) in enumerate(cn)
        println("  Column $j ($name): derivative = $(ws.derivative_matrix[1, j])")
    end
    
    # Expected derivatives for ∂μ/∂x at d=false, z=zv:
    # ∂μ/∂x = βx + βxd*d + βxz*z + βxdz*d*z
    # At d=false: ∂μ/∂x = βx + βxz*z = βx + βxz*zv
    println("\nExpected derivatives:")
    iβx = findfirst(isequal("x"), cn)
    iβxz = findfirst(isequal("x & z"), cn)
    println("  Column $iβx (x): should be 1.0, actual = $(ws.derivative_matrix[1, iβx])")
    println("  Column $iβxz (x & z): should be $zv, actual = $(ws.derivative_matrix[1, iβxz])")
    
    # All other x-related columns should be 0 at d=false
    iβxd = findfirst(isequal("x & d: true"), cn)
    iβxdz = findfirst(isequal("x & d: true & z"), cn)
    if iβxd !== nothing
        println("  Column $iβxd (x & d): should be 0.0, actual = $(ws.derivative_matrix[1, iβxd])")
    end
    if iβxdz !== nothing
        println("  Column $iβxdz (x & d & z): should be 0.0, actual = $(ws.derivative_matrix[1, iβxdz])")
    end
    
    # Compute the expected AME manually
    expected_derivative = 1.0 + zv  # βx coefficient is 1, βxz coefficient is 1, times zv
    manual_ame = expected_derivative  # For linear model, AME = derivative
    println("\nManual calculation:")
    println("  Expected derivative per observation: $expected_derivative")
    println("  Expected AME: $manual_ame")
    
    # Compare with what _ame_continuous_selective_fixed! would compute
    β = coef(model)
    cholΣβ = cholesky(vcov(model))
    invlink, dinvlink, d2invlink = link_functions(model)
    
    ame, se, grad = _ame_continuous_selective_fixed!(
        β, cholΣβ, ws.work_matrix, ws.derivative_matrix, dinvlink, d2invlink, ws
    )
    
    println("\n_ame_continuous_selective_fixed! results:")
    println("  AME: $ame")
    println("  SE: $se")
    println("  Expected AME: $(β[iβx] + β[iβxz] * zv)")
    
    return ws, ame, se
end

using Margins: AMEWorkspace,InplaceModeler,create_representative_data,modelmatrix_with_base!
using Margins: prepare_analytical_derivatives!, cholesky, link_functions, _ame_continuous_selective_fixed!

# Run the debug
ws, ame, se = debug_analytical_derivatives_detailed(m, df)