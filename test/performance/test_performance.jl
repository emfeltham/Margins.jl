# julia --project="." test/performance/test_performance.jl > test/test_performance.txt 2>&1

using Test
using Margins, GLM, DataFrames, StatsModels, BenchmarkTools
using Random

@testset "Performance Regression Prevention" begin
    
    # Helper function to create test datasets
    function create_test_data(n::Int)
        Random.seed!(06515)
        DataFrame(
            y = randn(n),
            x1 = randn(n),
            x2 = randn(n)
        )
    end
    
    @testset "O(1) Profile Margins Scaling" begin
        dataset_sizes = [1000, 5000, 10000, 1_000_000]
        profile_times = Float64[]
        
        for n in dataset_sizes
            @debug "Testing performance scaling" dataset_size=n
            data = create_test_data(n)
            model = lm(@formula(y ~ x1 + x2), data)
            
            profile_benchmark = @benchmark begin
                result_profile = profile_margins($model, $data, means_grid($data); type=:effects, vars=[:x1])
                df_profile = DataFrame(result_profile)
                df_profile
            end samples=5 evals=1
            
            result_profile = profile_margins(model, data, means_grid(data); type=:effects, vars=[:x1])
            df_profile = DataFrame(result_profile)
            @test nrow(df_profile) == 1
            @test !isempty(df_profile.estimate)
            @test !isnan(df_profile.estimate[1])
            
            profile_time = minimum(profile_benchmark).time / 1e9
            push!(profile_times, profile_time)
            
            pop_benchmark = @benchmark begin
                result_pop = population_margins($model, $data; type=:effects, vars=[:x1])
                df_pop = DataFrame(result_pop)
                df_pop
            end samples=5 evals=1
            
            result_pop = population_margins(model, data; type=:effects, vars=[:x1])
            df_pop = DataFrame(result_pop)
            @test nrow(df_pop) == 1
            
            pop_time = minimum(pop_benchmark).time / 1e9
            
            @debug "Performance benchmark results" n=n profile_time_sec=profile_time pop_time_sec=pop_time ratio=profile_time/pop_time
            
            if n > 1000
                @test profile_time < pop_time * 2
            end
        end
        
        if length(profile_times) >= 3
            time_ratio = profile_times[end] / profile_times[2]
            @debug "Profile scaling validation" largest_time=profile_times[end] mid_time=profile_times[2] scaling_ratio=time_ratio threshold=5.0 passes_scaling_test=(time_ratio < 5.0)
            @test time_ratio < 5.0
        end
    end
    
    @testset "Performance Characteristics Verification" begin
        data = create_test_data(1000)
        model = lm(@formula(y ~ x1 + x2), data)
        
        profile_benchmark = @benchmark profile_margins($model, $data, means_grid($data); type=:effects, vars=[:x1]) samples=3 evals=1
        profile_time = minimum(profile_benchmark).time / 1e9
        @test profile_time < 1.0
        
        pop_benchmark = @benchmark population_margins($model, $data; type=:effects, vars=[:x1]) samples=3 evals=1  
        pop_time = minimum(pop_benchmark).time / 1e9
        @test pop_time < 5.0
        
        profile_result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:x1])
        pop_result = population_margins(model, data; type=:effects, vars=[:x1])
        
        profile_df = DataFrame(profile_result)
        pop_df = DataFrame(pop_result)
        
        @test nrow(profile_df) == 1
        @test nrow(pop_df) == 1
        @test isfinite(profile_df.estimate[1])
        @test isfinite(pop_df.estimate[1])
    end
end