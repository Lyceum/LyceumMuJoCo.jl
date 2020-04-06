"""
    $(TYPEDEF)

The supertype for all MuJoCo-based environments. Subtypes of `AbstractMuJoCoEnvironment`
provide default functions for state, action, and observation (e.g. setstate!). For more
information, see the documentation for [`MJSim`](@ref).
"""
abstract type AbstractMuJoCoEnvironment <: AbstractEnvironment end


"""
    $(TYPEDSIGNATURES)

Return `env`'s underlying `MJSim` that defines the MuJoCo physics simulation. This is used
for providing default state, action, and observation functions as well as the visualizer
provided by LyceumMuJoCoViz.
"""
@mustimplement getsim(env::AbstractMuJoCoEnvironment)


@propagate_inbounds LyceumBase.statespace(env::AbstractMuJoCoEnvironment) = statespace(getsim(env))

@propagate_inbounds function LyceumBase.getstate!(state, env::AbstractMuJoCoEnvironment)
    getstate!(state, getsim(env))
    state
end

@propagate_inbounds function LyceumBase.setstate!(env::AbstractMuJoCoEnvironment, state)
    setstate!(getsim(env), state)
    env
end


@propagate_inbounds LyceumBase.observationspace(env::AbstractMuJoCoEnvironment) = observationspace(getsim(env))

@propagate_inbounds LyceumBase.getobservation!(obs, env::AbstractMuJoCoEnvironment) = getobservation!(obs, getsim(env))


@propagate_inbounds LyceumBase.actionspace(env::AbstractMuJoCoEnvironment) = actionspace(getsim(env))

@propagate_inbounds LyceumBase.getaction!(a, env::AbstractMuJoCoEnvironment) = getaction!(a, getsim(env))

@propagate_inbounds function LyceumBase.setaction!(env::AbstractMuJoCoEnvironment, a)
    setaction!(getsim(env), a)
    env
end


@propagate_inbounds LyceumBase.reset!(env::AbstractMuJoCoEnvironment) = (reset!(getsim(env)); env)

@propagate_inbounds LyceumBase.step!(env::AbstractMuJoCoEnvironment) = (step!(getsim(env)); env)


@propagate_inbounds Base.time(env::AbstractMuJoCoEnvironment) = time(getsim(env))

@propagate_inbounds LyceumBase.timestep(env::AbstractMuJoCoEnvironment) = timestep(getsim(env))
