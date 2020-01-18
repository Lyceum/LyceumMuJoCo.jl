using LyceumMuJoCo, LyceumBase
using Test, Random, Shapes, Pkg, LinearAlgebra, BenchmarkTools
using MuJoCo: TESTMODELXML

const LYCEUM_SUITE = [
    (LyceumMuJoCo.PointMass, (), ()),
    (LyceumMuJoCo.ArmHandPickup, (), ()),
]

const GYM = [
    (LyceumMuJoCo.SwimmerV2, (), ()),
    (LyceumMuJoCo.HopperV2, (), ()),
    (LyceumMuJoCo.Walker2DV2, (), ())
]

const DMC = [
    (LyceumMuJoCo.CartpoleSwingup, (), ()),
]

function test_group(group)
    @testset "Testing $etype\n    Args: $args.\n    Kwargs: $kwargs" for (etype, args, kwargs) in group
        LyceumBase.testenv_correctness(etype, args...; kwargs...)
    end
end

@testset "LyceumMuJoCo.jl" begin

    @testset "MJSim" begin include("mjsim.jl") end

    @testset "Environments" begin
        @testset "Lyceum Suite" begin test_group(LYCEUM_SUITE) end
        @testset "Gym" begin test_group(GYM) end
        @testset "DMC" begin test_group(DMC) end
    end

end