# julia --project="." test/test_performance_regression.jl > test/test_performance_regression.txt 2>&1

using Test, BenchmarkTools, Margins, GLM, DataFrames, CategoricalArrays

@testset "Performance Regression Tests" begin
    n = 1000
    df = DataFrame(
        y = randn(n),
        x_continuous = randn(n),
        x_boolean = rand([true, false], n),
        x_categorical = categorical(rand(["A", "B", "C"], n))
    )
    
    model = lm(@formula(y ~ x_continuous * x_boolean * x_categorical), df)
    
    @testset "No Performance Regression" begin
        # Continuous variables should be fast (using optimal FormulaCompiler)
        result = @benchmark population_margins($model, $df; type=:effects, vars=[:x_continuous])
        @test median(result.times) < 5e6  # Under 5ms
        @debug "Continuous variable performance: $(median(result.times)/1e6) ms"
        
        # Boolean variables should be very fast (DataScenario optimization)
        result = @benchmark population_margins($model, $df; type=:effects, vars=[:x_boolean])
        @test median(result.times) < 1e6  # Under 1ms
        @debug "Boolean variable performance: $(median(result.times)/1e6) ms"
        
        # Categorical variables should be very fast (DataScenario optimization)
        result = @benchmark population_margins($model, $df; type=:effects, vars=[:x_categorical])
        @test median(result.times) < 5e6  # Under 5ms (may be slightly slower due to contrasts)
        @debug "Categorical variable performance: $(median(result.times)/1e6) ms"
        
        # Mixed variables should scale reasonably
        result = @benchmark population_margins($model, $df; type=:effects, vars=[:x_continuous, :x_boolean, :x_categorical])
        @test median(result.times) < 10e6  # Under 10ms
        @debug "Mixed variables performance: $(median(result.times)/1e6) ms"
    end
    
    @testset "Backend Performance Comparison" begin
        # Compare AD vs FD performance for continuous variables
        result_ad = @benchmark population_margins($model, $df; type=:effects, vars=[:x_continuous], backend=:ad)
        result_fd = @benchmark population_margins($model, $df; type=:effects, vars=[:x_continuous], backend=:fd)
        
        @debug "Continuous AD performance: $(median(result_ad.times)/1e6) ms"
        @debug "Continuous FD performance: $(median(result_fd.times)/1e6) ms"
        
        # Both should be reasonably fast
        @test median(result_ad.times) < 10e6  # Under 10ms
        @test median(result_fd.times) < 10e6  # Under 10ms
        
        # Neither should be dramatically slower (within 2x of each other)
        faster = min(median(result_ad.times), median(result_fd.times))
        slower = max(median(result_ad.times), median(result_fd.times))
        @test slower / faster < 2.0  # Less than 2x difference
    end
    
    @testset "Memory Allocation Tests" begin
        # Test that the unified system doesn't have excessive allocations
        
        # Continuous variables (baseline - some allocations expected)
        result = @benchmark population_margins($model, $df; type=:effects, vars=[:x_continuous]) samples=10
        cont_allocs = result.memory
        @debug "Continuous variable allocations: $(cont_allocs) bytes"
        
        # Boolean variables (should be very efficient with DataScenario)
        result = @benchmark population_margins($model, $df; type=:effects, vars=[:x_boolean]) samples=10
        bool_allocs = result.memory
        @debug "Boolean variable allocations: $(bool_allocs) bytes"
        @test bool_allocs < 1e6  # Under 1MB allocations
        
        # Categorical variables (should be efficient with DataScenario)
        result = @benchmark population_margins($model, $df; type=:effects, vars=[:x_categorical]) samples=10
        cat_allocs = result.memory
        @debug "Categorical variable allocations: $(cat_allocs) bytes"
        @test cat_allocs < 2e6  # Under 2MB allocations (slightly higher due to contrasts)
        
        # Mixed variables (should scale reasonably)
        result = @benchmark population_margins($model, $df; type=:effects, vars=[:x_continuous, :x_boolean, :x_categorical]) samples=10
        mixed_allocs = result.memory
        @debug "Mixed variables allocations: $(mixed_allocs) bytes"
        
        # Mixed variable allocation should not exceed sum of components significantly
        # Allow some overhead but should not exceed 2x the sum
        component_sum = cont_allocs + bool_allocs + cat_allocs
        @test mixed_allocs < 2 * component_sum
    end
    
    @testset "Scaling Performance Tests" begin
        # Test performance scales reasonably with data size
        small_n = 100
        large_n = 5000
        
        small_df = DataFrame(
            y = randn(small_n),
            x_continuous = randn(small_n),
            x_boolean = rand([true, false], small_n),
            x_categorical = categorical(rand(["A", "B", "C"], small_n))
        )
        
        large_df = DataFrame(
            y = randn(large_n),
            x_continuous = randn(large_n), 
            x_boolean = rand([true, false], large_n),
            x_categorical = categorical(rand(["A", "B", "C"], large_n))
        )
        
        model_small = lm(@formula(y ~ x_continuous * x_boolean * x_categorical), small_df)
        model_large = lm(@formula(y ~ x_continuous * x_boolean * x_categorical), large_df)
        
        # Benchmark both sizes
        result_small = @benchmark population_margins($model_small, $small_df; type=:effects, vars=[:x_continuous]) samples=10
        result_large = @benchmark population_margins($model_large, $large_df; type=:effects, vars=[:x_continuous]) samples=10
        
        small_time = median(result_small.times)
        large_time = median(result_large.times)
        
        @debug "Small dataset (n=$small_n): $(small_time/1e6) ms"
        @debug "Large dataset (n=$large_n): $(large_time/1e6) ms"
        
        # Performance should scale roughly linearly (within reasonable bounds)
        # Large dataset is 50x bigger, allow up to 100x slower
        scale_factor = large_n / small_n  # 50x
        max_allowed_slowdown = scale_factor * 2  # Allow 2x overhead
        
        @test large_time / small_time < max_allowed_slowdown
        
        # Both should complete in reasonable time
        @test small_time < 5e6   # Under 5ms for small
        @test large_time < 100e6 # Under 100ms for large
    end
    
    @testset "Warmup and Stability Tests" begin
        # Test that performance is stable after warmup
        
        # Run several times to check for consistency
        times = Float64[]
        for i in 1:5
            result = @benchmark population_margins($model, $df; type=:effects, vars=[:x_continuous]) samples=3
            push!(times, median(result.times))
        end
        
        @debug "Performance consistency across runs: $(times ./ 1e6) ms"
        
        # Performance should be reasonably consistent (CV < 0.5)
        mean_time = sum(times) / length(times)
        std_time = sqrt(sum((t - mean_time)^2 for t in times) / (length(times) - 1))
        cv = std_time / mean_time
        
        @test cv < 0.5  # Coefficient of variation less than 50%
        @debug "Performance coefficient of variation: $(round(cv, digits=3))"
    end
end