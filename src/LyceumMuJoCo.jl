module LyceumMuJoCo

using Base: @propagate_inbounds

# Stdlib
using Statistics
using LinearAlgebra
using Random

# 3rd party
using UnsafeArrays
using StaticArrays
using Distributions
using Reexport
using Distances

# Lyceum
using Shapes

using MuJoCo
using MuJoCo.MJCore: mjtNum

@reexport using LyceumBase
using LyceumBase: RealVec, @mustimplement


import LyceumBase: statespace,
                   getstate!,
                   setstate!,
                   getstate,

                   obsspace,
                   getobs!,
                   getobs,

                   actionspace,
                   getaction!,
                   getaction,
                   setaction!,

                   rewardspace,
                   getreward,

                   evalspace,
                   geteval,

                   reset!,
                   randreset!,
                   step!,
                   isdone,
                   timestep,

                   tconstruct

using LyceumBase.Tools: perturb!, SPoint3D

export # AbstractMuJoCoEnvironment interface (an addition to AbstractEnvironment's interface)
       AbstractMuJoCoEnvironment,
       getsim,

       # MJSim interface
       MJSim,
       setstate!,
       sensorspace,
       getsensor!,
       getsensor,
       zeroctrl!,
       zerofullctrl!,
       forward!



include("mjsim.jl")


####
#### AbstractMuJoCoEnvironment Interface
####

abstract type AbstractMuJoCoEnvironment <: AbstractEnvironment end




@propagate_inbounds statespace(env::AbstractMuJoCoEnvironment) = statespace(getsim(env))

@propagate_inbounds function getstate!(state, env::AbstractMuJoCoEnvironment)
    getstate!(state, getsim(env))
    state
end

@propagate_inbounds function setstate!(env::AbstractMuJoCoEnvironment, state)
    setstate!(getsim(env), state)
    env
end




@propagate_inbounds obsspace(env::AbstractMuJoCoEnvironment) = sensorspace(getsim(env))

@propagate_inbounds getobs!(obs, env::AbstractMuJoCoEnvironment) = getsensor!(obs, getsim(env))




@propagate_inbounds actionspace(env::AbstractMuJoCoEnvironment) = actionspace(getsim(env))

@propagate_inbounds getaction!(a, env::AbstractMuJoCoEnvironment) = getaction!(a, getsim(env))

@propagate_inbounds function setaction!(env::AbstractMuJoCoEnvironment, a)
    setaction!(getsim(env), a)
    env
end




@propagate_inbounds reset!(env::AbstractMuJoCoEnvironment) = (reset!(getsim(env)); env)



@propagate_inbounds step!(env::AbstractMuJoCoEnvironment) = (step!(getsim(env)); env)



@propagate_inbounds Base.time(env::AbstractMuJoCoEnvironment) = time(getsim(env))

@propagate_inbounds timestep(env::AbstractMuJoCoEnvironment) = timestep(getsim(env))



@mustimplement getsim(env::AbstractMuJoCoEnvironment)


####
#### Environments
####

include("suite/pointmass.jl")
include("suite/armhand/armhandpickup.jl")

include("gym/swimmer-v2.jl")
include("gym/hopper-v2.jl")
include("gym/walker2d-v2.jl")

include("dmc/rewards.jl")
include("dmc/cartpole_swingup.jl")

end # module
