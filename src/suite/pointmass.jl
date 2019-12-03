struct PointMass{S, O} <: AbstractMuJoCoEnv
    sim::S
    osp::O
end

function PointMass(s::MJSim)
    osp = MultiShape(
        agent_xy_pos = VectorShape(mjtNum, 2),
        agent_xy_vel = VectorShape(mjtNum, 2),
        target_xy_pos = VectorShape(mjtNum, 2)
    )
    PointMass(s, osp)
end

PointMass() = first(sharedmemory_envs(PointMass, 1))

function LyceumBase.sharedmemory_envs(::Type{PointMass}, n::Integer)
    model = joinpath(@__DIR__, "pointmass.xml")
    Tuple(PointMass(s) for s in sharedmemory_mjsims(model, n, skip=3))
end


LyceumBase.statespace(env::PointMass) = statespace(env.sim)
LyceumBase.getstate!(s, env::PointMass) = getstate!(s, env.sim)

getsim(env::PointMass) = env.sim

LyceumBase.observationspace(env::PointMass) = env.osp
function LyceumBase.getobs!(o, env::PointMass)
    dn = env.sim.dn
    o .= (
        dn.xpos[:x, :agent], dn.xpos[:y, :agent],
        dn.qvel[:agent_x], dn.qvel[:agent_y],
        dn.xpos[:x, :target], dn.xpos[:y, :target],
    )
    o
end


LyceumBase.actionspace(env::PointMass) = actionspace(env.sim)
LyceumBase.getaction!(a, env::PointMass) = getaction!(a, env.sim)
LyceumBase.setaction!(env::PointMass, a) = (setaction!(env.sim, a); env)


LyceumBase.rewardspace(env::PointMass) = ScalarShape(Float64)
LyceumBase.getreward(env::PointMass) = 1.0 - dist2target(env)


LyceumBase.evaluationspace(env::PointMass) = ScalarShape(Float64)
LyceumBase.geteval(env::PointMass) = dist2target(env)

LyceumBase.reset!(env::PointMass) = (reset!(env.sim); env)
LyceumBase.reset!(env::PointMass, s) = (reset!(env.sim, s); env)
function LyceumBase.randreset!(env::PointMass)
    s = env.sim
    reset!(s)
    s.dn.qpos[:agent_x] = rand() * 2.0 - 1.0
    s.dn.qpos[:agent_y] = rand() * 2.0 - 1.0
    s.dn.qpos[:target_x] = rand() * 2.0 - 1.0
    s.dn.qpos[:target_y] = rand() * 2.0 - 1.0
    forward!(s)
    env
end


LyceumBase.step!(env::PointMass) = (step!(env.sim); env)


Base.time(env::PointMass) = time(env.sim)


function dist2target(env::PointMass)
    dn = env.sim.dn
    dist = (dn.xpos[:x, :agent] - dn.xpos[:x, :target]) ^ 2
    dist += (dn.xpos[:y, :agent] - dn.xpos[:y, :target]) ^ 2
    sqrt(dist)
end


