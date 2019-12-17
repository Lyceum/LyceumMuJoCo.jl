module LyceumMuJoCo

using UnsafeArrays, Shapes, StaticArrays, Distributions, Reexport, Random, LinearAlgebra

using Base: @propagate_inbounds
using MuJoCo
using MuJoCo.MJCore: mjtNum

@reexport using LyceumBase
using LyceumBase: RealVec, @mustimplement

import LyceumBase: statespace,
                   getstate!,
                   getstate,

                   observationspace,
                   getobs!,
                   getobs,

                   actionspace,
                   getaction!,
                   getaction,
                   setaction!,

                   rewardspace,
                   getreward,

                   evaluationspace,
                   geteval,

                   reset!,
                   randreset!,
                   step!,
                   isdone,
                   timestep,
                   effective_timestep,

                   thread_construct

export # AbstractMuJoCoEnv interface (an addition to AbstractEnv's interface)
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
       fullreset!,
       forward!



include("mjsim.jl")


####
#### AbstractMuJoCoEnv Interface
####

abstract type AbstractMuJoCoEnv <: AbstractEnv end
@mustimplement getsim(env::AbstractMuJoCoEnv) # TODO document "opt in" behavior
@mustimplement thread_construct(::Type{<:AbstractMuJoCoEnv}, N::Integer, args...; kwargs...)

@inline statespace(env::AbstractMuJoCoEnv) = statespace(getsim(env))
@inline getstate!(s, env::AbstractMuJoCoEnv) = getstate!(s, getsim(env))

@inline observationspace(env::AbstractMuJoCoEnv) = sensorspace(getsim(env))
@inline getobs!(o, env::AbstractMuJoCoEnv) = getsensor!(o, getsim(env))

@inline actionspace(env::AbstractMuJoCoEnv) = actionspace(getsim(env))
@inline getaction!(a, env::AbstractMuJoCoEnv) = getaction!(a, getsim(env))
@inline setaction!(env::AbstractMuJoCoEnv, a) = (setaction!(getsim(env), a); env)

@inline reset!(env::AbstractMuJoCoEnv) = (reset!(getsim(env)); env)
@inline reset!(env::AbstractMuJoCoEnv, s) = (reset!(getsim(env), s); env)
@inline reset!(env::AbstractMuJoCoEnv, s, c) = (reset!(getsim(env), s, c); env)

@inline step!(env::AbstractMuJoCoEnv) = (step!(getsim(env)); env)

@inline timestep(env::AbstractMuJoCoEnv) = timestep(getsim(env))
@inline effective_timestep(env::AbstractMuJoCoEnv) = effective_timestep(getsim(env))
@inline Base.time(env::AbstractMuJoCoEnv) = time(getsim(env))


####
#### Environments
####

include("suite/pointmass.jl")

#include("gym/humanoid-v2.jl")
#include("gym/swimmer-v2.jl")
#include("gym/hopper-v2.jl")

end # module
