using LyceumMuJoCo, LyceumBase
using Test, Random, Shapes, Pkg, LinearAlgebra, BenchmarkTools
using MuJoCo: TESTMODELXML

const ROLLOUT_HORIZON = 50

const SPACE_TO_FUNCS = Dict(
    statespace => (get=getstate, getbang=getstate!),
    actionspace => (get=getaction, getbang=getaction!, setbang=setaction!),
    obsspace => (get=getobs, getbang=getobs!),
    #rewardspace => (get=getreward, ),
    #evalspace => (get=geteval, )
)

const LYCEUM_SUITE = [
    (LyceumMuJoCo.PointMass, (), ())
]

const GYM_SUITE = [
    #(LyceumMuJoCo.HumanoidV2, (), ()),
    #(LyceumMuJoCo.SwimmerV2, (), ()),
    #(LyceumMuJoCo.HopperV2, (), ()),
]

include("util.jl")



@testset "LyceumMuJoCo.jl" begin

    @testset "MJSim" begin include("mjsim.jl") end

    @testset "Environments" begin
        @testset "Lyceum Suite" begin test_group(LYCEUM_SUITE) end
        @testset "Gym Suite" begin test_group(GYM_SUITE) end
    end

end