struct HumanoidGetup{S<:MJSim} <: AbstractMuJoCoEnvironment
    sim::S
end

function tconstruct(::Type{HumanoidGetup}, n::Integer)
    modelpath = joinpath(@__DIR__, "humanoid.xml")
    return Tuple(HumanoidGetup(s) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip = 1))
end
HumanoidGetup() = first(tconstruct(HumanoidGetup, 1))

@inline _getheight(shapedstate::ShapedView, ::HumanoidGetup) = shapedstate.qpos[3]

@inline getsim(env::HumanoidGetup) = env.sim

@propagate_inbounds function reset!(env::HumanoidGetup)
    reset_nofwd!(env.sim)
    #@inbounds env.sim.d.qpos .= env.sim.mn.key_qpos[:,:laying]
    # noalloc:
    key_qpos = env.sim.m.key_qpos
    @uviews key_qpos @inbounds env.sim.d.qpos .= view(key_qpos,:,2) # noalloc
    forward!(env.sim)
    return env
end

@propagate_inbounds function randreset!(rng::Random.AbstractRNG, env::HumanoidGetup)
    reset_nofwd!(env.sim)
    rand!(rng, Uniform(-0.4, 0.4), env.sim.d.ctrl)
    forward!(env.sim)
    # apply a random, fixed control vector 200 times
    for _=1:100 step!(env.sim) end
    zeroctrl!(env.sim)
    for _=1:100 step!(env.sim) end
    # should be on the ground now.
    zeroctrl!(env.sim)
    env.sim.d.qvel .= 0.0
    forward!(env.sim)
    env
end

@propagate_inbounds function getreward(state, action, obs, env::HumanoidGetup)
    @boundscheck checkaxes(statespace(env), state)
    height = _getheight(statespace(env)(state), env)
    target = 1.25
    reward = 1.0
    if height < target
        reward -= 2.0 * abs(target - height)
    end
    reward -= 1e-3 * norm(action)^2

    return reward
end

@propagate_inbounds function geteval(state, action, obs, env::HumanoidGetup)
    @boundscheck checkaxes(statespace(env), state)
    return _getheight(statespace(env)(state), env)
end

