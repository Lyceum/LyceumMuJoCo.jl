"""
    $(TYPEDEF)

Make a two-dimensional, one-legged robot walk forward as fast as possible. The robot model is
based on the work T. Erez, Y. Tassa, and E. Todorov: ["Infinite Horizon Model Predictive
Control for Nonlinear Periodic Tasks"](https://homes.cs.washington.edu/~todorov/papers/ErezRSS11.pdf), 2011.

# Spaces

* **State: (20, )**
* **Action: (3, )**
* **Observation: (11, )**
"""
mutable struct HopperV2{Sim, SSpace, OSpace} <: AbstractMuJoCoEnvironment
    sim::Sim
    statespace::SSpace
    observationspace::OSpace
    last_torso_x::Float64
    randreset_distribution::Uniform{Float64}
    function HopperV2(sim::MJSim)
        sspace = MultiShape(
            simstate=statespace(sim),
            last_torso_x=ScalarShape(Float64)
        )
        ospace = MultiShape(
            cropped_qpos = VectorShape(Float64, sim.m.nq - 1),
            qvel = VectorShape(Float64, sim.m.nv)
        )
        env = new{typeof(sim), typeof(sspace), typeof(ospace)}(
            sim, sspace, ospace, 0, Uniform(-0.005, 0.005)
        )
        reset!(env)
    end
end

HopperV2() = first(tconstruct(HopperV2, 1))

function LyceumBase.tconstruct(::Type{HopperV2}, n::Integer)
    modelpath = joinpath(@__DIR__, "hopper.xml")
    [HopperV2(s) for s in tconstruct(MJSim, n, modelpath, skip=4)]
end


@inline getsim(env::HopperV2) = env.sim


@inline LyceumBase.statespace(env::HopperV2) = env.statespace

function LyceumBase.getstate!(state, env::HopperV2)
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        getstate!(shaped.simstate, env.sim)
        shaped.last_torso_x = env.last_torso_x
    end
    state
end

function LyceumBase.setstate!(env::HopperV2, state)
    checkaxes(statespace(env), state)
    @uviews state begin
        shaped = statespace(env)(state)
        setstate!(env.sim, shaped.simstate)
        env.last_torso_x = shaped.last_torso_x
    end
    env
end


@inline LyceumBase.observationspace(env::HopperV2) = env.observationspace

function LyceumBase.getobservation!(obs, env::HopperV2)
    checkaxes(observationspace(env), obs)
    qpos = env.sim.d.qpos
    @views @uviews qpos obs begin
        shaped = observationspace(env)(obs)
        copyto!(shaped.cropped_qpos, qpos[2:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
    end
    obs
end


function LyceumBase.getreward(state, action, ::Any, env::HopperV2)
    checkaxes(statespace(env), state)
    checkaxes(actionspace(env), action)
    @uviews state begin
        shapedstate = statespace(env)(state)
        alive_bonus = 1.0
        reward = (_torso_x(shapedstate, env) - shapedstate.last_torso_x) / timestep(env)
        reward += alive_bonus
        reward -= 1e-3 * sum(x->x^2, action)
        reward
    end
end


function LyceumBase.reset!(env::HopperV2)
    reset!(env.sim)
    env.last_torso_x = _torso_x(env)
    env
end

function LyceumBase.randreset!(rng::AbstractRNG, env::HopperV2)
    reset_nofwd!(env.sim)
    perturb!(rng, env.randreset_distribution, env.sim.d.qpos)
    perturb!(rng, env.randreset_distribution, env.sim.d.qvel)
    forward!(env.sim)
    env.last_torso_x = _torso_x(env)
    env
end


function LyceumBase.step!(env::HopperV2)
    env.last_torso_x = _torso_x(env)
    step!(env.sim)
    env
end

function LyceumBase.isdone(state, ::Any, env::HopperV2)
    checkaxes(statespace(env), state)
    @uviews state begin
        shapedstate = statespace(env)(state)
        torso_x = _torso_x(shapedstate, env)
        height = _torso_height(shapedstate, env)
        torso_ang = _torso_ang(shapedstate, env)
        qpos = shapedstate.simstate.qpos
        qvel = shapedstate.simstate.qvel

        done = !(
            all(isfinite, state)
            && all(x->abs(x) < 100, uview(qpos, 3:length(qpos)))
            && all(x->abs(x) < 100, uview(qvel))
            && height > 0.7
            && abs(torso_ang) < 0.2
        )
        done
    end
end

@inline _torso_x(shapedstate::ShapedView, ::HopperV2) = shapedstate.simstate.qpos[1]
@inline _torso_x(env::HopperV2) = env.sim.d.qpos[1]
@inline _torso_height(shapedstate::ShapedView, ::HopperV2) = shapedstate.simstate.qpos[2]
@inline _torso_ang(shapedstate::ShapedView, ::HopperV2) = shapedstate.simstate.qpos[3]