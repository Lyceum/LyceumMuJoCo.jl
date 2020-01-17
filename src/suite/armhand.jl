struct ArmHandPickup{S<:MJSim, O} <: AbstractMuJoCoEnvironment
    sim::S
    osp::O
    ball::Int
    palm::Int
    thumb::Int
    index::Int
    middle::Int
    ring::Int
    pinky::Int
    goal::SVector{3, Float64}

    function ArmHandPickup(sim::MJSim)
        m  = sim.m
        mn, dn = sim.mn, sim.dn

        osp = MultiShape(d_thumb  = ScalarShape(Float64),
                         d_index  = ScalarShape(Float64),
                         d_middle = ScalarShape(Float64),
                         d_ring   = ScalarShape(Float64),
                         d_pinky  = ScalarShape(Float64),
                         a_thumb  = ScalarShape(Float64),
                         a_index  = ScalarShape(Float64),
                         a_middle = ScalarShape(Float64),
                         a_ring   = ScalarShape(Float64),
                         a_pinky  = ScalarShape(Float64),
                         a_close  = ScalarShape(Float64),
                         handball = ScalarShape(Float64),
                         ballgoal = ScalarShape(Float64),
                         ball     = VectorShape(Float64, 3),
                         palm     = VectorShape(Float64, 3))

        ball = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "ball")
        palm = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "palm")

        thumb  = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "thumb_IMU")
        index  = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "index_IMU")
        middle = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "middle_IMU")
        ring   = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "ring_IMU")
        pinky  = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "pinky_IMU")

        goal = SA_F64[0.0, 0.2, 0.5]

        new{typeof(sim), typeof(osp)}(sim, osp,
                                      ball, palm,
                                      thumb, index, middle, ring, pinky,
                                      goal)
    end
end

function tconstruct(::Type{ArmHandPickup}, n::Integer)
    modelpath = joinpath(@__DIR__, "armhand.xml")
    return Tuple(ArmHandPickup(s) for s in LyceumBase.tconstruct(MJSim, n, modelpath, skip = 3))
end
ArmHandPickup() = first(tconstruct(ArmHandPickup, 1))

@inline getsim(env::ArmHandPickup) = env.sim
@inline obsspace(env::ArmHandPickup) = env.osp

@propagate_inbounds function randreset!(rng::Random.AbstractRNG, env::ArmHandPickup)
    reset_nofwd!(env.sim)
    # randomize object starting position
    env.sim.d.qpos[1] = rand(rng, Uniform(-0.15, 0.15))
    env.sim.d.qpos[2] = rand(rng, Uniform(-0.1, 0.1))
    forward!(env.sim)
    env
end

@inline _sitedist(s1, s2, dmin) = min(euclidean(s1, s2), dmin)

@propagate_inbounds function getobs!(obs, env::ArmHandPickup)
    m, d = env.sim.m, env.sim.d
    osp  = obsspace(env)
    sx   = d.site_xpos
    dmin = 0.5

    _ball = siteSA(sx, env.ball)
    _palm = siteSA(sx, env.palm)
    _thumb  = siteSA(sx, env.thumb)
    _index  = siteSA(sx, env.index)
    _middle = siteSA(sx, env.middle)
    _ring   = siteSA(sx, env.ring)
    _pinky  = siteSA(sx, env.pinky)

    _goal = _ball - env.goal

    @uviews obs @inbounds begin
        shaped = osp(obs)

        shaped.ball    .= _ball
        shaped.palm    .= _palm .- _ball

        shaped.handball = _sitedist(_palm, _ball, dmin)
        shaped.ballgoal = _sitedist(_ball, _goal, dmin)

        shaped.d_thumb  = _sitedist(_thumb,  _ball, dmin)
        shaped.d_index  = _sitedist(_index,  _ball, dmin)
        shaped.d_middle = _sitedist(_middle, _ball, dmin)
        shaped.d_ring   = _sitedist(_ring,   _ball, dmin)
        shaped.d_pinky  = _sitedist(_pinky,  _ball, dmin)
        shaped.a_thumb  = cosine_dist(_goal, _thumb)
        shaped.a_index  = cosine_dist(_goal, _index)
        shaped.a_middle = cosine_dist(_goal, _middle)
        shaped.a_ring   = cosine_dist(_goal, _ring)
        shaped.a_pinky  = cosine_dist(_goal, _pinky)
        shaped.a_close  = cosine_dist(_middle, _thumb)
    end
    obs
end

@propagate_inbounds function getreward(state, action, obs, env::ArmHandPickup)
    os = obsspace(env)(obs)
    _handball = os.handball / 0.5
    _ballgoal = os.ballgoal / 0.25

    reward = -_handball
    if _handball < 0.06
        reward = 2.0 - _ballgoal
        reward -= ( os.d_thumb + os.d_index + os.d_middle + os.d_ring + os.d_pinky ) * 0.1
    end
    reward
end

@propagate_inbounds function geteval(state, action, obs, env::ArmHandPickup)
    os = obsspace(env)(obs)
    os.ball[3]
end


