struct HumanoidGetup{S<:MJSim, O} <: AbstractMuJoCoEnvironment
    sim::S
    osp::O
    head::Int
    uwaist::Int
    lwaist::Int
    rfoot::Int
    lfoot::Int
    rhand::Int
    lhand::Int
    function HumanoidGetup(sim::MJSim)
        m = sim.m
        osp = MultiShape(z     = VectorShape(Float64, 5),
                         torso = VectorShape(Float64, 3),
                         rleg  = VectorShape(Float64, 3),
                         lleg  = VectorShape(Float64, 3),
                         rarm  = VectorShape(Float64, 3),
                         larm  = VectorShape(Float64, 3),
                         rvel  = VectorShape(Float64, m.nv))

        new{typeof(sim), typeof(osp)}(sim, osp,
                                      jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_GEOM, "head"),
                                      jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_GEOM, "uwaist"),
                                      jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_GEOM, "lwaist"),
                                      jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_GEOM, "right_foot_cap1"),
                                      jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_GEOM, "left_foot_cap1"),
                                      jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_GEOM, "right_hand"),
                                      jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_GEOM, "left_hand")
                                     )

    end
end

HumanoidGetup() = first(tconstruct(HumanoidGetup, 1))

function tconstruct(::Type{HumanoidGetup}, n::Integer)
    modelpath = joinpath(@__DIR__, "humanoid.xml")
    return Tuple(HumanoidGetup(s) for s in tconstruct(MJSim, n, modelpath, skip = 1))
end


@inline getsim(env::HumanoidGetup) = env.sim
@inline obsspace(env::HumanoidGetup) = env.osp


@propagate_inbounds function reset!(env::HumanoidGetup)
    fastreset_nofwd!(env.sim)
    key_qpos = env.sim.m.key_qpos
    @uviews key_qpos @inbounds env.sim.d.qpos .= view(key_qpos, :, 2)
    forward!(env.sim)
    env
end


@propagate_inbounds function randreset!(rng::Random.AbstractRNG, env::HumanoidGetup)
    # reset to default state
    fastreset_nofwd!(env.sim)

    # apply a random, fixed control vector 200 times to find a valid random state
    rand!(rng, Uniform(-0.3, 0.3), env.sim.d.ctrl)
    for _ = 1:20
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

@propagate_inbounds function getobs!(obs, env::HumanoidGetup)
    @boundscheck checkaxes(obsspace(env), obs)

    m, d = env.sim.m, env.sim.d

    qvel = d.qvel
    gx   = d.geom_xpos

    @uviews obs gx qvel @inbounds begin
        o = obsspace(env)(obs)
        o.z[1]   = d.qpos[3]
        o.z[2]   = gx[3, env.rfoot]
        o.z[3]   = gx[3, env.lfoot]
        o.z[4]   = gx[3, env.rhand]
        o.z[5]   = gx[3, env.lhand]
        o.torso .= SPoint3D(gx, env.uwaist) - SPoint3D(gx, env.head)
        o.rleg  .= SPoint3D(gx, env.rfoot) - SPoint3D(gx, env.lwaist)
        o.lleg  .= SPoint3D(gx, env.lfoot) - SPoint3D(gx, env.lwaist)
        o.rarm  .= SPoint3D(gx, env.rhand) - SPoint3D(gx, env.uwaist)
        o.larm  .= SPoint3D(gx, env.lhand) - SPoint3D(gx, env.uwaist)
        o.rvel  .= clamp.(qvel, -20.0, 20.0) ./ 20.0 #view(qvel, 1:6)
    end
    obs
end

@inline function getreward(state, action, obs, env::HumanoidGetup)
    @boundscheck checkaxes(statespace(env), state)

    ss = statespace(env)(state)
    height = _getheight(ss, env)
    target = 1.25
    reward = 1.0
    if height < target
        reward -= 2.0 * abs(target - height)
    else
        if abs(target - height) < 0.1
            reward += 5.0
        end
    end
    #reward -= 1e-3 * norm(ss.qvel)^2

    reward
end

@inline function geteval(state, action, obs, env::HumanoidGetup)
    @boundscheck checkaxes(statespace(env), state)

    _getheight(statespace(env)(state), env)
end

function isdone(state, ::Any, ::Any, env::HumanoidGetup)
    #qvel = env.sim.d.qvel
    #MAX_SPEED = 60.0
    #any(x->abs(x)>MAX_SPEED, qvel) && return true
    return false
end

@inline _getheight(shapedstate::ShapedView, ::HumanoidGetup) = shapedstate.qpos[3]
