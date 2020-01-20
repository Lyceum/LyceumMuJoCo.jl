struct HumanoidGetup{S<:MJSim} <: AbstractMuJoCoEnvironment
    sim::S
end

HumanoidGetup() = first(tconstruct(HumanoidGetup, 1))

function tconstruct(::Type{HumanoidGetup}, n::Integer)
    modelpath = joinpath(@__DIR__, "humanoid.xml")
    return Tuple(HumanoidGetup(s) for s in tconstruct(MJSim, n, modelpath, skip = 1))
end


@inline getsim(env::HumanoidGetup) = env.sim


function reset!(env::HumanoidGetup)
    fastreset_nofwd!(env.sim)
    key_qpos = env.sim.m.key_qpos
    @uviews key_qpos @inbounds env.sim.d.qpos .= view(key_qpos, :, 2)
    forward!(env.sim)
    env
end


function randreset!(rng::Random.AbstractRNG, env::HumanoidGetup)
    # reset to default state
    fastreset_nofwd!(env.sim)

    # apply a random, fixed control vector 200 times to find a valid random state
    rand!(rng, Uniform(-0.4, 0.4), env.sim.d.ctrl)
    for _ = 1:100
        step!(env.sim)
    end
    zeroctrl!(env.sim)
    for _ = 1:100
        step!(env.sim)
    end
    # should be on the ground now.

    # reset velocities and compute forward dynamics
    zeroctrl!(env.sim)
    env.sim.d.qvel .= 0.0
    env.sim.d.time = 0.0
    forward!(env.sim)

    env
end


@inline function getreward(state, action, obs, env::HumanoidGetup)
    @boundscheck checkaxes(statespace(env), state)

    height = _getheight(statespace(env)(state), env)
    target = 1.25
    reward = 1.0
    if height < target
        reward -= 2.0 * abs(target - height)
    end
    reward -= 1e-3 * norm(action)^2

    reward
end

@inline function geteval(state, action, obs, env::HumanoidGetup)
    @boundscheck checkaxes(statespace(env), state)

    _getheight(statespace(env)(state), env)
end


@inline _getheight(shapedstate::ShapedView, ::HumanoidGetup) = shapedstate.qpos[3]
