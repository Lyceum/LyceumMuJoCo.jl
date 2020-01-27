"""
    $(TYPEDEF)

Pickup a block using a robot arm modeled after the [Modular Prosthetic Limb]
(https://www.jhuapl.edu/Content/techdigest/pdf/V30-N03/30-3-Johannes.pdf)
developed by the Applied Physics Laboratory, The Johns Hopkins University.

# Spaces

* **State: (106, )**
* **Action: (36, )**
* **Observation: (19, )**
"""
struct ArmHandPickup{S<:MJSim,O<:MultiShape} <: AbstractMuJoCoEnvironment
    sim::S
    obsspace::O
    ball::Int
    palm::Int
    thumb::Int
    index::Int
    middle::Int
    ring::Int
    pinky::Int
    goal::SVector{3,Float64}

    function ArmHandPickup(sim::MJSim)
        m = sim.m
        mn, dn = sim.mn, sim.dn

        obsspace = MultiShape(
            d_thumb = ScalarShape(Float64),
            d_index = ScalarShape(Float64),
            d_middle = ScalarShape(Float64),
            d_ring = ScalarShape(Float64),
            d_pinky = ScalarShape(Float64),
            a_thumb = ScalarShape(Float64),
            a_index = ScalarShape(Float64),
            a_middle = ScalarShape(Float64),
            a_ring = ScalarShape(Float64),
            a_pinky = ScalarShape(Float64),
            a_close = ScalarShape(Float64),
            handball = ScalarShape(Float64),
            ballgoal = ScalarShape(Float64),
            ball = VectorShape(Float64, 3),
            palm = VectorShape(Float64, 3),
        )

        ball = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "ball")
        palm = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "palm")

        thumb = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "thumb_IMU")
        index = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "index_IMU")
        middle = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "middle_IMU")
        ring = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "ring_IMU")
        pinky = jl_name2id(sim.m, MuJoCo.MJCore.mjOBJ_SITE, "pinky_IMU")

        goal = SA_F64[0.0, 0.2, 0.5]

        new{typeof(sim),typeof(obsspace)}(
            sim,
            obsspace,
            ball,
            palm,
            thumb,
            index,
            middle,
            ring,
            pinky,
            goal,
        )
    end
end

ArmHandPickup() = first(tconstruct(ArmHandPickup, 1))

function tconstruct(::Type{ArmHandPickup}, n::Integer)
    modelpath = joinpath(@__DIR__, "armhand.xml")
    Tuple(ArmHandPickup(s) for s in tconstruct(MJSim, n, modelpath, skip = 3))
end


@inline getsim(env::ArmHandPickup) = env.sim


@inline obsspace(env::ArmHandPickup) = env.obsspace

@propagate_inbounds function getobs!(obs, env::ArmHandPickup)
    @boundscheck checkaxes(obsspace(env), obs)

    m, d = env.sim.m, env.sim.d
    sx = d.site_xpos
    dmin = 0.5

    ball = SPoint3D(sx, env.ball)
    palm = SPoint3D(sx, env.palm)
    thumb = SPoint3D(sx, env.thumb)
    index = SPoint3D(sx, env.index)
    middle = SPoint3D(sx, env.middle)
    ring = SPoint3D(sx, env.ring)
    pinky = SPoint3D(sx, env.pinky)
    goal = ball - env.goal

    @uviews obs @inbounds begin
        shaped = obsspace(env)(obs)

        shaped.ball .= ball
        shaped.palm .= palm .- ball

        shaped.handball = _sitedist(palm, ball, dmin)
        shaped.ballgoal = _sitedist(ball, goal, dmin)

        shaped.d_thumb = _sitedist(thumb, ball, dmin)
        shaped.d_index = _sitedist(index, ball, dmin)
        shaped.d_middle = _sitedist(middle, ball, dmin)
        shaped.d_ring = _sitedist(ring, ball, dmin)
        shaped.d_pinky = _sitedist(pinky, ball, dmin)
        shaped.a_thumb = cosine_dist(goal, thumb)
        shaped.a_index = cosine_dist(goal, index)
        shaped.a_middle = cosine_dist(goal, middle)
        shaped.a_ring = cosine_dist(goal, ring)
        shaped.a_pinky = cosine_dist(goal, pinky)
        shaped.a_close = cosine_dist(middle, thumb)
    end

    obs
end


@propagate_inbounds function getreward(state, action, obs, env::ArmHandPickup)
    @boundscheck checkaxes(obsspace(env), obs)

    os = obsspace(env)(obs)
    handball = os.handball / 0.5
    ballgoal = os.ballgoal / 0.5

    reward = -handball
    if handball < 0.06
        reward = 2.0 - 2 * ballgoal
        reward -= 0.1 * (os.d_thumb + os.d_index + os.d_middle + os.d_ring + os.d_pinky)
    end
    reward
end

@propagate_inbounds function geteval(state, action, obs, env::ArmHandPickup)
    @boundscheck checkaxes(obsspace(env), obs)
    obsspace(env)(obs).ball[3]
end


@propagate_inbounds function randreset!(rng::Random.AbstractRNG, env::ArmHandPickup)
    fastreset_nofwd!(env.sim)
    env.sim.d.qpos[1] = rand(rng, Uniform(-0.15, 0.15))
    env.sim.d.qpos[2] = rand(rng, Uniform(-0.1, 0.1))
    forward!(env.sim)
    env
end


@inline _sitedist(s1, s2, dmin) = min(euclidean(s1, s2), dmin)
