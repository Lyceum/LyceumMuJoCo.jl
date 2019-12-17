struct PointMass{S <: MJSim, O} <: AbstractMuJoCoEnv
    sim::S
    osp::O
    function PointMass(sim::MJSim)
        osp = MultiShape(
            agent_xy_pos = VectorShape(mjtNum, 2),
            agent_xy_vel = VectorShape(mjtNum, 2),
            target_xy_pos = VectorShape(mjtNum, 2)
        )
        new{typeof(sim), typeof(osp)}(sim, osp)
    end
end

function LyceumBase.thread_construct(::Type{PointMass}, N::Integer)
    modelpath = joinpath(@__DIR__, "pointmass.xml")
    Tuple(PointMass(s) for s in thread_construct(MJSim, N, modelpath, skip=1))
end

PointMass() = first(thread_construct(PointMass, 1))


getsim(env::PointMass) = env.sim

observationspace(env::PointMass) = env.osp
function getobs!(o, env::PointMass)
    dn = env.sim.dn
    @uviews o begin
        shaped = env.osp(o)
        shaped.agent_xy_pos .= dn.xpos[:x, :agent], dn.xpos[:y, :agent]
        shaped.agent_xy_vel .= dn.qvel[:agent_x], dn.qvel[:agent_y]
        shaped.target_xy_pos .= dn.xpos[:x, :target], dn.xpos[:y, :target]
    end
    o
end

function getreward(state, action, obs, env::PointMass)
    shaped = env.osp(obs)
    1.0 - norm(shaped.agent_xy_pos - shaped.target_xy_pos)
end

function geteval(state, action, obs, env::PointMass)
    shaped = env.osp(obs)
    norm(shaped.agent_xy_pos - shaped.target_xy_pos)
end

function isdone(state, action, obs, env::PointMass)
    nsteps = time(env) / effective_timestep(env)
    if Threads.threadid() == 1
        return nsteps > rand(25:50)
    else
        nsteps > rand(100:200)
    end

end

function randreset!(env::PointMass)
    s = env.sim
    reset!(s)
    s.dn.qpos[:agent_x] = rand() * 2.0 - 1.0
    s.dn.qpos[:agent_y] = rand() * 2.0 - 1.0
    s.dn.qpos[:target_x] = rand() * 2.0 - 1.0
    s.dn.qpos[:target_y] = rand() * 2.0 - 1.0
    forward!(s)
    env
end