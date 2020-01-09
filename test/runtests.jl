using LyceumMuJoCo, LyceumBase
using Test, Random, Shapes, Pkg, LinearAlgebra, BenchmarkTools
using MuJoCo: TESTMODELXML

const LYCEUM_SUITE = [
    (LyceumMuJoCo.PointMass, (), ())
]

const GYM = [
    (LyceumMuJoCo.SwimmerV2, (), ()),
    (LyceumMuJoCo.HopperV2, (), ())
]

function test_group(group)
    @testset "Testing $etype\n    Args: $args.\n    Kwargs: $kwargs" for (etype, args, kwargs) in group
        LyceumBase.test_env(etype, args...; kwargs...)
    end
end

@testset "LyceumMuJoCo.jl" begin

    @testset "MJSim" begin include("mjsim.jl") end

    @testset "Environments" begin
        @testset "Lyceum Suite" begin test_group(LYCEUM_SUITE) end
        @testset "Gym" begin test_group(GYM) end
    end

end