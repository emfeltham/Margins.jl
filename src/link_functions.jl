# link_functions.jl - Clean implementation of link function utilities

###############################################################################
# Link Function Extraction and Utilities
###############################################################################

function get_inverse_link_function(workspace::MarginalEffectsWorkspace)
    # For now return identity - in a full implementation this would be extracted from the model
    # and stored in the workspace during creation
    return identity
end

"""
    extract_link_functions(model::StatisticalModel) -> (inverse_link, first_derivative, second_derivative)

Extract inverse link function and its first and second derivatives from a statistical model.

# Returns
- `inverse_link::Function`: μ(η) - inverse link function
- `first_derivative::Function`: dμ/dη - first derivative of inverse link  
- `second_derivative::Function`: d²μ/dη² - second derivative of inverse link

# Example
```julia
model = glm(@formula(y ~ x), df, Binomial(), LogitLink())
invlink, dinvlink, d2invlink = extract_link_functions(model)

η = 0.5
μ = invlink(η)        # Predicted value
dμ = dinvlink(η)      # Rate of change
d2μ = d2invlink(η)    # Curvature
```
"""
function extract_link_functions(model::StatisticalModel)
    model_family = family(model)
    link_object = model_family.link
    
    inverse_link = η -> linkinv(link_object, η)
    first_derivative = η -> mueta(link_object, η)
    second_derivative = η -> mueta2(link_object, η)
    
    return inverse_link, first_derivative, second_derivative
end

"""
    get_inverse_link_function(model::StatisticalModel) -> Function

Get just the inverse link function from a statistical model.
Convenience function when only μ(η) is needed.
"""
function get_inverse_link_function(model::StatisticalModel)
    link_object = family(model).link
    return η -> linkinv(link_object, η)
end

###############################################################################
# Link Function Validation and Testing
###############################################################################

"""
    validate_link_derivatives(link::Link, test_points::AbstractVector = [-2.0, -1.0, 0.0, 1.0, 2.0]) -> Bool

Validate that analytical derivatives match numerical derivatives for a link function.
Useful for testing custom link implementations.

# Arguments
- `link::Link`: Link function to validate
- `test_points::AbstractVector`: Points at which to test derivatives

# Returns
`true` if all derivatives match within tolerance, `false` otherwise
"""
function validate_link_derivatives(link::Link, test_points::AbstractVector = [-2.0, -1.0, 0.0, 1.0, 2.0])
    tolerance = 1e-8
    all_valid = true
    
    for η in test_points
        try
            # Get analytical derivatives
            μ = linkinv(link, η)
            dμ_analytical = mueta(link, η)
            d2μ_analytical = mueta2(link, η)
            
            # Compute numerical derivatives
            dμ_numerical = ForwardDiff.derivative(η_val -> linkinv(link, η_val), η)
            d2μ_numerical = ForwardDiff.derivative(η_val -> mueta(link, η_val), η)
            
            # Check first derivative
            if abs(dμ_analytical - dμ_numerical) > tolerance
                @warn "First derivative mismatch at η=$η: analytical=$dμ_analytical, numerical=$dμ_numerical"
                all_valid = false
            end
            
            # Check second derivative  
            if abs(d2μ_analytical - d2μ_numerical) > tolerance
                @warn "Second derivative mismatch at η=$η: analytical=$d2μ_analytical, numerical=$d2μ_numerical"
                all_valid = false
            end
            
        catch e
            @warn "Derivative validation failed at η=$η: $e"
            all_valid = false
        end
    end
    
    return all_valid
end

"""
    benchmark_link_derivatives(link::Link, η_values::AbstractVector; samples::Int = 1000) -> NamedTuple

Benchmark performance of link function derivative computations.

# Arguments
- `link::Link`: Link function to benchmark
- `η_values::AbstractVector`: Linear predictor values to test
- `samples::Int`: Number of benchmark samples

# Returns
NamedTuple with timing statistics for analytical vs numerical derivatives
"""
function benchmark_link_derivatives(link::Link, η_values::AbstractVector; samples::Int = 1000)
    n_points = length(η_values)
    
    # Benchmark analytical derivatives
    analytical_first_times = Float64[]
    analytical_second_times = Float64[]
    
    for _ in 1:samples
        # First derivative
        t1 = @elapsed for η in η_values
            mueta(link, η)
        end
        push!(analytical_first_times, t1)
        
        # Second derivative
        t2 = @elapsed for η in η_values
            mueta2(link, η)
        end
        push!(analytical_second_times, t2)
    end
    
    # Benchmark numerical derivatives  
    numerical_first_times = Float64[]
    numerical_second_times = Float64[]
    
    for _ in 1:samples
        # First derivative
        t1 = @elapsed for η in η_values
            ForwardDiff.derivative(η_val -> linkinv(link, η_val), η)
        end
        push!(numerical_first_times, t1)
        
        # Second derivative
        t2 = @elapsed for η in η_values
            ForwardDiff.derivative(η_val -> mueta(link, η_val), η)
        end
        push!(numerical_second_times, t2)
    end
    
    return (
        link_type = typeof(link),
        test_points = n_points,
        benchmark_samples = samples,
        
        # Analytical timing
        analytical_first_mean_ns = mean(analytical_first_times) * 1e9 / n_points,
        analytical_second_mean_ns = mean(analytical_second_times) * 1e9 / n_points,
        
        # Numerical timing
        numerical_first_mean_ns = mean(numerical_first_times) * 1e9 / n_points,
        numerical_second_mean_ns = mean(numerical_second_times) * 1e9 / n_points,
        
        # Speedup ratios
        first_derivative_speedup = mean(numerical_first_times) / mean(analytical_first_times),
        second_derivative_speedup = mean(numerical_second_times) / mean(analytical_second_times)
    )
end

###############################################################################
# Link Function Diagnostics
###############################################################################

"""
    LinkFunctionDiagnostics

Diagnostic information for a link function's behavior and numerical properties.
"""
struct LinkFunctionDiagnostics
    link_type::Type
    domain_issues::Vector{String}
    range_issues::Vector{String}
    derivative_issues::Vector{String}
    numerical_stability::Dict{String,Any}
    performance_info::Union{NamedTuple,Nothing}
end

"""
    diagnose_link_function(link::Link; 
                          test_range::Tuple{Real,Real} = (-10.0, 10.0),
                          n_test_points::Int = 1000) -> LinkFunctionDiagnostics

Comprehensive diagnostic analysis of a link function's properties.

# Arguments
- `link::Link`: Link function to diagnose
- `test_range::Tuple{Real,Real}`: Range of η values to test
- `n_test_points::Int`: Number of test points to evaluate

# Returns
`LinkFunctionDiagnostics` containing detailed analysis of the link function
"""
function diagnose_link_function(link::Link; 
                               test_range::Tuple{Real,Real} = (-10.0, 10.0),
                               n_test_points::Int = 1000)
    
    η_values = range(test_range[1], test_range[2], length=n_test_points)
    
    domain_issues = String[]
    range_issues = String[]
    derivative_issues = String[]
    
    # Test inverse link function
    μ_values = Float64[]
    dμ_values = Float64[]
    d2μ_values = Float64[]
    
    for η in η_values
        try
            μ = linkinv(link, η)
            dμ = mueta(link, η)
            d2μ = mueta2(link, η)
            
            push!(μ_values, μ)
            push!(dμ_values, dμ)
            push!(d2μ_values, d2μ)
            
            # Check for domain issues
            if !isfinite(μ)
                push!(domain_issues, "Non-finite μ at η=$η")
            end
            
            if !isfinite(dμ)
                push!(derivative_issues, "Non-finite dμ/dη at η=$η")
            end
            
            if !isfinite(d2μ)
                push!(derivative_issues, "Non-finite d²μ/dη² at η=$η")
            end
            
            # Check for invalid derivative signs
            if dμ <= 0
                push!(derivative_issues, "Non-positive dμ/dη=$dμ at η=$η (should be positive for valid link)")
            end
            
        catch e
            push!(domain_issues, "Evaluation failed at η=$η: $e")
        end
    end
    
    # Analyze numerical stability
    numerical_stability = Dict{String,Any}()
    
    if !isempty(μ_values)
        numerical_stability["μ_range"] = (minimum(μ_values), maximum(μ_values))
        numerical_stability["μ_finite_fraction"] = count(isfinite, μ_values) / length(μ_values)
    end
    
    if !isempty(dμ_values)
        finite_dμ = filter(isfinite, dμ_values)
        if !isempty(finite_dμ)
            numerical_stability["dμ_range"] = (minimum(finite_dμ), maximum(finite_dμ))
            numerical_stability["dμ_condition_number"] = maximum(finite_dμ) / minimum(finite_dμ)
        end
    end
    
    if !isempty(d2μ_values)
        finite_d2μ = filter(isfinite, d2μ_values)
        if !isempty(finite_d2μ)
            numerical_stability["d2μ_range"] = (minimum(finite_d2μ), maximum(finite_d2μ))
        end
    end
    
    # Validate derivatives if no major issues
    derivative_validation_passed = false
    if length(domain_issues) < 10 && length(derivative_issues) < 10
        try
            derivative_validation_passed = validate_link_derivatives(link, [-2.0, -1.0, 0.0, 1.0, 2.0])
        catch e
            push!(derivative_issues, "Derivative validation failed: $e")
        end
    end
    
    numerical_stability["derivative_validation_passed"] = derivative_validation_passed
    
    # Performance benchmarking (optional, for non-problematic links)
    performance_info = nothing
    if length(domain_issues) == 0 && length(derivative_issues) == 0
        try
            test_points = range(-2.0, 2.0, length=100)
            performance_info = benchmark_link_derivatives(link, test_points; samples=10)
        catch e
            @warn "Performance benchmarking failed: $e"
        end
    end
    
    return LinkFunctionDiagnostics(
        typeof(link),
        domain_issues,
        range_issues,
        derivative_issues,
        numerical_stability,
        performance_info
    )
end

function Base.show(io::IO, diag::LinkFunctionDiagnostics)
    println(io, "LinkFunctionDiagnostics for $(diag.link_type)")
    println(io, "━" ^ 50)
    
    if isempty(diag.domain_issues) && isempty(diag.range_issues) && isempty(diag.derivative_issues)
        println(io, "✅ No issues detected")
    else
        if !isempty(diag.domain_issues)
            println(io, "⚠ Domain issues ($(length(diag.domain_issues))):")
            for issue in diag.domain_issues[1:min(5, end)]
                println(io, "  • $issue")
            end
            if length(diag.domain_issues) > 5
                println(io, "  ... and $(length(diag.domain_issues) - 5) more")
            end
        end
        
        if !isempty(diag.derivative_issues)
            println(io, "⚠ Derivative issues ($(length(diag.derivative_issues))):")
            for issue in diag.derivative_issues[1:min(5, end)]
                println(io, "  • $issue")
            end
            if length(diag.derivative_issues) > 5
                println(io, "  ... and $(length(diag.derivative_issues) - 5) more")
            end
        end
    end
    
    # Show numerical stability info
    if haskey(diag.numerical_stability, "derivative_validation_passed")
        status = diag.numerical_stability["derivative_validation_passed"] ? "✅ PASSED" : "❌ FAILED"
        println(io, "Derivative validation: $status")
    end
    
    if haskey(diag.numerical_stability, "dμ_condition_number")
        cond_num = diag.numerical_stability["dμ_condition_number"]
        println(io, "Derivative condition number: $(round(cond_num, digits=2))")
    end
    
    # Show performance info if available
    if diag.performance_info !== nothing
        perf = diag.performance_info
        println(io, "Performance (analytical vs numerical):")
        println(io, "  First derivative speedup: $(round(perf.first_derivative_speedup, digits=1))x")
        println(io, "  Second derivative speedup: $(round(perf.second_derivative_speedup, digits=1))x")
    end
end

###############################################################################
# Standard Link Function Tests
###############################################################################

"""
    test_all_standard_links() -> Dict{Type,Bool}

Test all standard GLM link functions for correctness and performance.
Returns a dictionary mapping link types to test results.
"""
function test_all_standard_links()
    standard_links = [
        IdentityLink(),
        LogLink(),
        LogitLink(),
        ProbitLink(),
        CloglogLink(),
        InverseLink(),
        InverseSquareLink(),
        SqrtLink(),
        PowerLink(2.0),
        PowerLink(0.5),
        PowerLink(-1.0)
    ]
    
    results = Dict{Type,Bool}()
    
    for link in standard_links
        try
            # Test derivative validation
            validation_passed = validate_link_derivatives(link)
            results[typeof(link)] = validation_passed
            
            if validation_passed
                println("✅ $(typeof(link)): All tests passed")
            else
                println("❌ $(typeof(link)): Derivative validation failed")
            end
            
        catch e
            println("💥 $(typeof(link)): Testing failed with error: $e")
            results[typeof(link)] = false
        end
    end
    
    return results
end

###############################################################################
# Export Statements
###############################################################################

export extract_link_functions, get_inverse_link_function
export mueta2
export validate_link_derivatives, benchmark_link_derivatives
export diagnose_link_function, LinkFunctionDiagnostics
export test_all_standard_links
