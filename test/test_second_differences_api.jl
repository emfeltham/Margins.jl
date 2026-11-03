# test_second_differences_api.jl
# Public surface of the second-differences API and the migration hints for the removed functions

using Test
using Margins

@testset "second differences: public API surface" begin
    exported = Set(names(Margins))
    @test :second_differences in exported
    removed = (:second_differences_at, :second_differences_pairwise, :second_differences_all_contrasts,
               :second_differences_table, :second_difference)
    for f in removed
        @test f ∉ exported                       # no longer exported ...
        @test !isdefined(Margins, f)             # ... and no longer defined
    end
    @test Set(keys(Margins.REMOVED_SECOND_DIFFERENCE_FUNCTIONS)) == Set(removed)

    # the two methods of the one public function
    ms = methods(second_differences)
    @test any(m -> Base.unwrap_unionall(m.sig).parameters[2] <: Margins.EffectsResult, ms)
    @test any(m -> Base.unwrap_unionall(m.sig).parameters[2] <: Margins.RegressionModel, ms)
end

@testset "second differences: removed functions explain themselves" begin
    for (f, mention) in ((:second_differences_at, "at="), (:second_differences_pairwise, "contrast="),
                         (:second_differences_all_contrasts, "second_differences(result"),
                         (:second_differences_table, "second_differences(result"),
                         (:second_difference, "include_gradients=true"))
        err = try
            getfield(Margins, f); nothing
        catch e
            e
        end
        @test err isa UndefVarError
        msg = sprint(showerror, err)
        @test occursin("removed in Margins 3.0", msg)
        @test occursin(mention, msg)
    end
    # unrelated undefined names get no hint
    err = try getfield(Margins, :definitely_not_a_function); nothing catch e; e end
    @test !occursin("removed in Margins 3.0", sprint(showerror, err))
end
