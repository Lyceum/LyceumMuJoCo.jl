module LyceumMuJoCo

using UnsafeArrays, Shapes, StaticArrays, Distributions, Reexport, Random, LinearAlgebra, Distances

using Base: @propagate_inbounds
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

export # AbstractMuJoCoEnv interface (an addition to AbstractEnvironment's interface)
       AbstractMuJoCoEnv,
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
#### AbstractMuJoCoEnv Interface
####

abstract type AbstractMuJoCoEnv <: AbstractEnvironment end




@propagate_inbounds statespace(env::AbstractMuJoCoEnv) = statespace(getsim(env))

@propagate_inbounds function getstate!(state, env::AbstractMuJoCoEnv)
    getstate!(state, getsim(env))
    state
end

@propagate_inbounds function setstate!(env::AbstractMuJoCoEnv, state)
    setstate!(getsim(env), state)
    env
end




@propagate_inbounds obsspace(env::AbstractMuJoCoEnv) = sensorspace(getsim(env))

@propagate_inbounds getobs!(obs, env::AbstractMuJoCoEnv) = getsensor!(obs, getsim(env))




@propagate_inbounds actionspace(env::AbstractMuJoCoEnv) = actionspace(getsim(env))

@propagate_inbounds getaction!(a, env::AbstractMuJoCoEnv) = getaction!(a, getsim(env))

@propagate_inbounds function setaction!(env::AbstractMuJoCoEnv, a)
    setaction!(getsim(env), a)
    env
end




@propagate_inbounds reset!(env::AbstractMuJoCoEnv) = (reset!(getsim(env)); env)



@propagate_inbounds step!(env::AbstractMuJoCoEnv) = (step!(getsim(env)); env)



@propagate_inbounds Base.time(env::AbstractMuJoCoEnv) = time(getsim(env))

@propagate_inbounds timestep(env::AbstractMuJoCoEnv) = timestep(getsim(env))



@mustimplement getsim(env::AbstractMuJoCoEnv)


####
#### Environments
####

include("suite/pointmass.jl")

end # module
