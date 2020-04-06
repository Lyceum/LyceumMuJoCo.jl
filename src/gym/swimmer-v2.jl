"""
    $(TYPEDEF)

This task involves a 3-link swimming robot in a viscous fluid, where the goal is to make
it swim forward as fast as possible, by actuating the two joints. The origins of task can
be traced back to Remi Coulom's thesis: ["Reinforcement Learning Using Neural Networks,
with Applications to Motor Control"] (https://tel.archives-ouvertes.fr/tel-00003985/file/tel-00003985.pdf)

* **State: (17, )**
* **Action: (2, )**
* **Observation: (8, )**
"""
mutable struct SwimmerV2{Sim, SSpace, OSpace} <: AbstractMuJoCoEnvironment
    sim::Sim
    statespace::SSpace
    observationspace::OSpace
    last_torso_x::Float64
    randreset_distribution::Uniform{Float64}

    function SwimmerV2(sim::MJSim)
        sspace = MultiShape(simstate=statespace(sim), last_torso_x=ScalarShape(Float64))
        ospace = MultiShape(
            qpos_cropped = VectorShape(Float64, sim.m.nq - 2),
            qvel=statespace(sim).qvel
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(
            sim,
            sspace,
            ospace,
            0,
            Uniform(-0.1, 0.1)
        )
        reset!(env)
    end
end

SwimmerV2() = first(tconstruct(SwimmerV2, 1))

function LyceumBase.tconstruct(::Type{SwimmerV2}, n::Integer)
    modelpath = joinpath(@__DIR__, "swimmer.xml")
    Tuple(SwimmerV2(s) for s in tconstruct(MJSim, n, modelpath, skip=4))
end


@inline getsim(env::SwimmerV2) = env.sim


@inline LyceumBase.statespace(env::SwimmerV2) = env.statespace

function LyceumBase.getstate!(state, env::SwimmerV2)
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        getstate!(shaped.simstate, env.sim)
        shaped.last_torso_x = env.last_torso_x
    end
    state
end

function LyceumBase.setstate!(env::SwimmerV2, state)
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        setstate!(env.sim, shaped.simstate)
        env.last_torso_x = shaped.last_torso_x
    end
    env
end


@inline LyceumBase.observationspace(env::SwimmerV2) = env.observationspace

function LyceumBase.getobservation!(obs, env::SwimmerV2)
    checkaxes(observationspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews obs qpos begin
        shaped = observationspace(env)(obs)
        copyto!(shaped.qpos_cropped, qpos[3:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
    end
    obs
end


function LyceumBase.getreward(state, action, ::Any, env::SwimmerV2)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        reward_fwd = (_torso_x(shapedstate, env) - shapedstate.last_torso_x) / timestep(env)
        reward_ctrl = -1e-4 * sum(x->x^2, action)
        reward_fwd + reward_ctrl
    end
end


function LyceumBase.reset!(env::SwimmerV2)
    reset!(env.sim)
    env.last_torso_x = _torso_x(env)
    env
end

function LyceumBase.randreset!(rng::AbstractRNG, env::SwimmerV2)
    reset_nofwd!(env.sim)
    perturb!(rng, env.randreset_distribution, env.sim.d.qpos)
    perturb!(rng, env.randreset_distribution, env.sim.d.qvel)
    forward!(env.sim)
    env.last_torso_x = _torso_x(env)
    env
end

function LyceumBase.step!(env::SwimmerV2)
    env.last_torso_x = _torso_x(env)
    step!(env.sim)
    env
end

@inline _torso_x(shapedstate::ShapedView, ::SwimmerV2) = shapedstate.simstate.qpos[1]
@inline _torso_x(env::SwimmerV2) = env.sim.d.qpos[1]