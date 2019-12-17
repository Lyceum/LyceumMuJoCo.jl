struct HumanoidV2{SIM, S, O} <: AbstractMuJoCoEnv
    sim::SIM
    statespace::S
    observationspace::O
    last_masscenter::MVector{3, Float64}
    randreset_distribution::Uniform{Float64}
end

function HumanoidV2(s::MJSim)
    m, d = s.m, s.d
    DT = Float64
    sspace = MultiShape(statespace(s), last_masscenter=VectorShape(Float64, 3))
    ospace = MultiShape(
        torsoqpos = VectorShape(DT, m.nq - 2),
        qvel = VectorShape(DT, m.nv),
        cinert = MatrixShape(DT, 10, m.nbody),
        cvel = MatrixShape(DT, 6, m.nbody),
        qfrc_actuator = VectorShape(DT, m.nv),
        cfrc_ext = MatrixShape(DT, 6, m.nbody)
    )
    HumanoidV2(s, sspace, ospace, MVector(masscenter(s)), Uniform(-0.01, 0.01))
end

HumanoidV2() = HumanoidV2(MJSim(MJSimParameters(HumanoidV2)))

function MJSimParameters(::Type{<:HumanoidV2})
    MJSimParameters(modelpath=joinpath(@__DIR__, "humanoid-v2.xml"), skip=5)
end

getsim(env::HumanoidV2) = env.sim


LyceumBase.statespace(env::HumanoidV2) = env.statespace
function LyceumBase.getstate!(s, env::HumanoidV2)
    @uviews s begin
        simstate = view(s, 1:length(statespace(env.sim)))
        getstate!(simstate, env.sim)
        env.statespace(s).last_masscenter .= env.last_masscenter
    end
    s
end


LyceumMuJoCo.observationspace(env::HumanoidV2) = env.observationspace
function LyceumMuJoCo.getobs!(o, env::HumanoidV2)
    d = env.sim.d
    qpos = d.qpos
    osp = observationspace(env)
    @views @uviews o qpos begin
        shaped = osp(o)
        copyto!(shaped.torsoqpos, qpos[3:end])
        copyto!(shaped.qvel, d.qvel)
        copyto!(shaped.cinert, d.cinert)
        copyto!(shaped.cvel, d.cvel)
        copyto!(shaped.qfrc_actuator, d.qfrc_actuator)
        copyto!(shaped.cfrc_ext, d.cfrc_ext)
    end
    o
end


function LyceumBase.getreward(env::HumanoidV2)
    cur_masscenter = masscenter(env.sim)
    d = env.sim.d
    alive_bonus = 5.0
    lin_vel_cost = 1.25 * (cur_masscenter[1] - env.last_masscenter[1]) / effective_timestep(env.sim)
    quad_ctrl_cost = 0.1 * sum(x->x^2, d.ctrl)
    quad_impact_cost = .5e-6 * sum(d.cfrc_ext)
    quad_impact_cost = min(quad_impact_cost, 10)
    lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
end


function LyceumBase.geteval(env::HumanoidV2)
    cur_masscenter = masscenter(env.sim)
    (cur_masscenter[1] - env.last_masscenter[1]) / effective_timestep(env.sim)
end


function LyceumBase.reset!(env::HumanoidV2)
    reset!(env.sim)
    env.last_masscenter .= masscenter(env.sim)
    env
end

function LyceumBase.reset!(env::HumanoidV2, s)
    @uviews s begin
        simstate = view(s, 1:length(statespace(env.sim)))
        reset!(env.sim, simstate)
        env.last_masscenter .= env.statespace(s).last_masscenter
    end
    env
end

function LyceumBase.randreset!(env::HumanoidV2)
    s = reset!(env.sim)

    qpos_noise = similar_type(MVector, statespace(s).qpos)(undef)
    qvel_noise = similar_type(MVector, statespace(s).qvel)(undef)
    rand!(env.randreset_distribution, qpos_noise)
    rand!(env.randreset_distribution, qvel_noise)
    s.d.qpos .+= qpos_noise
    s.d.qvel .+= qvel_noise

    forward!(env.sim)
    copyto!(env.last_masscenter, masscenter(s))
    env
end


function LyceumMuJoCo.step!(env::HumanoidV2)
    env.last_masscenter .= masscenter(env.sim)
    step!(env.sim)
    env
end

LyceumBase.isdone(env::HumanoidV2) = height(env) < 1.0 || height(env) > 2.0


# Humanoid utils
height(env::HumanoidV2) = env.sim.d.qpos[3]