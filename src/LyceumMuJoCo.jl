module LyceumMuJoCo

using UnsafeArrays, Shapes, StaticArrays, Distributions, Reexport, Random

using Base: @propagate_inbounds
using MuJoCo
using .MJCore: mjtNum

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
                   sharedmemory_envs,
                   timestep,
                   effective_timestep

export # AbstractMuJoCoEnv interface (an addition to AbstractEnv's interface)
       AbstractMuJoCoEnv,
       getsim,

       # MJSim interface
       MJSim,
       MJSimParameters,
       sharedmemory_mjsims,
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
#### AbstractMuJoCoEnv and suite
####

abstract type AbstractMuJoCoEnv <: AbstractEnv end
@mustimplement getsim(env::AbstractMuJoCoEnv) # TODO document "opt in" behavior
@mustimplement MJSimParameters(::Type{<:AbstractMuJoCoEnv})

function sharedmemory_envs(
    ::Type{E},
    n::Integer,
    args...;
    kwargs...,
) where {E<:AbstractMuJoCoEnv}
    n > 0 || throw(ArgumentError("n must be > 0"))
    defaults = MJSimParameters(E)
    Tuple(E(s, args...; kwargs...) for s in sharedmemory_mjsims(defaults.modelpath, n, defaults.args...; defaults.kwargs...))
end

statespace(env::AbstractMuJoCoEnv) = statespace(getsim(env))
getstate!(s, env::AbstractMuJoCoEnv) = getstate!(s, getsim(env))

observationspace(env::AbstractMuJoCoEnv) = sensorspace(getsim(env))
getobs!(o, env::AbstractMuJoCoEnv) = getsensor!(o, getsim(env))

actionspace(env::AbstractMuJoCoEnv) = actionspace(getsim(env))
getaction!(a, env::AbstractMuJoCoEnv) = getaction!(a, getsim(env))
setaction!(env::AbstractMuJoCoEnv, a) = (setaction!(getsim(env), a); env)

reset!(env::AbstractMuJoCoEnv) = (reset!(getsim(env)); env)
reset!(env::AbstractMuJoCoEnv, s) = (reset!(getsim(env), s); env)
reset!(env::AbstractMuJoCoEnv, s, c) = (reset!(getsim(env), s, c); env)

step!(env::AbstractMuJoCoEnv) = (step!(getsim(env)); env)

timestep(env::AbstractMuJoCoEnv) = timestep(getsim(env))
effective_timestep(env::AbstractMuJoCoEnv) = effective_timestep(getsim(env))
Base.time(env::AbstractMuJoCoEnv) = time(getsim(env))


include("suite/pointmass.jl")

include("gym/humanoid-v2.jl")
include("gym/swimmer-v2.jl")
include("gym/hopper-v2.jl")

end # module
