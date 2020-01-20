struct PenHand{S <: MJSim, O, P} <: AbstractMuJoCoEnvironment
    sim::S
    osp::O
    ssp::P
    penlength::Float64
    tarlength::Float64
    act_mid::Vector{Float64}
    act_rng::Vector{Float64}

    obj_id::Int
    obj_top::Int
    obj_bot::Int
    trg_top::Int
    trg_bot::Int
    trg_id::Int
    eps_ball::SVector{3, Float64}

    function PenHand(sim::MJSim)
        m  = sim.m
        mn, dn = sim.mn, sim.dn

        osp = MultiShape(qpos     = VectorShape(Float64, m.nq-6),
                         objvel   = VectorShape(Float64, 6),
                         objpos   = VectorShape(Float64, 3),
                         despos   = VectorShape(Float64, 3),
                         objorien = VectorShape(Float64, 3),
                         desorien = VectorShape(Float64, 3))

        ssp = MultiShape(statespace(sim),
                         objquat  = VectorShape(Float64, 4))

        penlength = norm(dn.site_xpos[:, :object_top] - dn.site_xpos[:, :object_bottom])
        tarlength = norm(dn.site_xpos[:, :target_top] - dn.site_xpos[:, :target_bottom])

        act_mid = vec(mean(m.actuator_ctrlrange, dims=1))
        act_rng = 0.5*(m.actuator_ctrlrange[2,:]-m.actuator_ctrlrange[1,:])

        obj_id   = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_BODY, "Object")
        obj_top  = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "object_top")
        obj_bot  = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "object_bottom")
        trg_top  = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "target_top")
        trg_bot  = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "target_bottom")
        trg_id   = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_BODY, "target")

        eps_ball = SVector{3, Float64}(dn.site_xpos[:, :eps_ball])

        new{typeof(sim), typeof(osp), typeof(ssp)}(sim, osp, ssp,
                                                   penlength, tarlength,
                                                   act_mid, act_rng,
                                                   obj_id,
                                                   obj_top, obj_bot,
                                                   trg_top, trg_bot,
                                                   trg_id, eps_ball)
    end
end

function tconstruct(::Type{PenHand}, n::Integer)
    modelpath = joinpath(@__DIR__, "mj_envs/mj_envs/hand_manipulation_suite/assets/DAPG_pen.xml")
    return Tuple(PenHand(s) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip = 5))
end
function ensembleconstruct(::Type{PenHand}, n::Integer)
    modelpath = joinpath(@__DIR__, "mj_envs/mj_envs/hand_manipulation_suite/assets/DAPG_pen.xml")
    return Tuple(PenHand(MJSim(modelpath, skip = 5)) for m=1:n )
end
PenHand() = first(tconstruct(PenHand, 1))

@inline getsim(env::PenHand) = env.sim
@inline obsspace(env::PenHand) = env.osp
@inline statespace(env::PenHand) = env.ssp

@propagate_inbounds function setstate!(env::PenHand, s)
    @uviews s begin
        simstate = view(s, 1:length(statespace(env.sim)))
        setstate!(env.sim, simstate)
    end
    objquat = env.ssp(s).objquat
    env.sim.m.body_quat[1, env.trg_id] = objquat[1]
    env.sim.m.body_quat[2, env.trg_id] = objquat[2]
    env.sim.m.body_quat[3, env.trg_id] = objquat[3]
    env.sim.m.body_quat[4, env.trg_id] = objquat[4]
    env
end
@propagate_inbounds function getstate!(s, env::PenHand)
    @uviews s begin
        simstate = view(s, 1:length(statespace(env.sim)))
        getstate!(simstate, env.sim)
    end
    #env.ssp(s).objquat .= env.sim.m.body_quat[:, env.trg_id] # TODO this allocates
    objquat = env.ssp(s).objquat
    objquat[1] = env.sim.m.body_quat[1, env.trg_id]
    objquat[2] = env.sim.m.body_quat[2, env.trg_id]
    objquat[3] = env.sim.m.body_quat[3, env.trg_id]
    objquat[4] = env.sim.m.body_quat[4, env.trg_id]
    s
end

@propagate_inbounds function getaction!(a, env::PenHand)
    @. a = ( env.sim.d.ctrl - env.act_mid ) / env.act_rng
    a
end
@propagate_inbounds function setaction!(env::PenHand, a)
    env.sim.d.ctrl .= clamp.(a, -1.0, 1.0)
    @. env.sim.d.ctrl = env.act_mid + env.sim.d.ctrl * env.act_rng
    #forwardskip!(env.sim, MuJoCo.MJCore.mjSTAGE_ACC, false)
    env
end

@propagate_inbounds function _euler2quat(euler)
    ai, aj, ak = euler
    si, sj, sk = sin(ai), sin(aj), sin(ak)
    ci, cj, ck = cos(ai), cos(aj), cos(ak)
    cc, cs     = ci * ck, ci * sk
    sc, ss     = si * ck, si * sk
    return (cj*cc+sj*ss, cj*sc-sj*cs, -(cj*ss+sj*cc), cj*cs-sj*sc)
end
@propagate_inbounds function reset!(env::PenHand)
    fastreset_nofwd!(env.sim)
    env.sim.m.body_quat[1, env.trg_id] = 1.0
    env.sim.m.body_quat[2, env.trg_id] = 0.0
    env.sim.m.body_quat[3, env.trg_id] = 0.0
    env.sim.m.body_quat[4, env.trg_id] = 0.0
    forward!(env.sim)
    env
end
@propagate_inbounds function randreset!(rng::Random.AbstractRNG, env::PenHand)
    fastreset_nofwd!(env.sim)
    desorien = SA_F64[rand(rng, Uniform(-1, 1)), rand(rng, Uniform(-1, 1)), 0.0]

    env.sim.m.body_quat[:, env.trg_id] .= _euler2quat(desorien) # TODO change the model

    forward!(env.sim)
    env
end

@propagate_inbounds function getreward(::Any, ::Any, obs, env::PenHand)
    os = obsspace(env)(obs)
    objpos   = siteSA(os.objpos)
    despos   = siteSA(os.despos)
    objorien = siteSA(os.objorien)
    desorien = siteSA(os.desorien)

    dist = norm(objpos - despos)
    reward = -dist
    similarity = dot(objorien, desorien)
    reward += similarity

    # bonus for being close to desired orientation
    if dist < 0.075 && similarity > 0.90
        reward += 10
    end
    if dist < 0.075 && similarity > 0.95
        reward += 50
    end

    if objpos[3] < 0.075 # this is supposed to happen with the is-done calculation
        reward -= 5
    end

    return reward
end
function LyceumBase.isdone(env::PenHand)
    objz = env.sim.d.xpos[3, env.obj_id]
    return objz < 0.075
end

@propagate_inbounds function getobs!(o, env::PenHand)
    m, d = env.sim.m, env.sim.d
    osp  = obsspace(env)
    qpos = d.qpos
    qvel = d.qvel
    sx   = d.site_xpos
    xpos = d.xpos

    @uviews o qpos qvel @inbounds begin
        shaped = osp(o)
        shaped.qpos     .= view(qpos, 1:m.nq-6)
        shaped.objvel   .= view(qvel, (m.nv-5):m.nv)
        shaped.objpos   .= siteSA(xpos, env.obj_id)
        shaped.despos   .= env.eps_ball
        shaped.objorien .= (siteSA(sx, env.obj_top) - siteSA(sx, env.obj_bot)) / env.penlength
        shaped.desorien .= (siteSA(sx, env.trg_top) - siteSA(sx, env.trg_bot)) / env.tarlength
    end
    o
end

@propagate_inbounds function geteval(::Any, ::Any, obs, env::PenHand)
    os = obsspace(env)(obs)
    objorien = siteSA(os.objorien)
    desorien = siteSA(os.desorien)
    similarity = dot(objorien, desorien)
    return similarity
end


