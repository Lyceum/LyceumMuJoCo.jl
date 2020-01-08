mutable struct WalkerV2{SIM, S, O} <: AbstractMuJoCoEnvironment
    sim::SIM
    statespace::S
    observationspace::O
    lastx::Float64
    randreset_distribution::Uniform{Float64}
    function WalkerV2(sim::MJSim)
        sspace = MultiShape(
            simstate=statespace(sim),
            lastx=ScalarShape(Float64)
        )
        ospace = MultiShape(
            croppedqpos = VectorShape(Float64, sim.m.nq - 1),
            qvel = VectorShape(Float64, sim.m.nv)
        )
        new{typeof(sim), typeof(sspace), typeof(ospace)}(
            sim, sspace, ospace, _getx(sim), Uniform(-0.005, 0.005)
        )
    end
end

WalkerV2() = first(sharedmemory_envs(WalkerV2, 1))

function LyceumBase.sharedmemory_envs(::Type{WalkerV2}, n::Integer)
    model = joinpath(@__DIR__, "walker-v2.xml")
    Tuple(WalkerV2(s) for s in sharedmemory_mjsims(model, n, skip=4))
end

LyceumMuJoCo.getsim(env::WalkerV2) = env.sim


LyceumBase.statespace(env::WalkerV2) = env.statespace
function LyceumBase.getstate!(s, env::WalkerV2)
    @uviews s begin
        shaped = env.statespace(s)
        getstate!(shaped.simstate, env.sim)
        shaped.lastx = env.lastx
    end
    s
end


LyceumMuJoCo.observationspace(env::WalkerV2) = env.observationspace
function LyceumMuJoCo.getobs!(o, env::WalkerV2)
    qpos = env.sim.d.qpos
    @views @uviews qpos o begin
        shaped = env.observationspace(o)
        copyto!(shaped.croppedqpos, qpos[2:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
        clamp!(shaped.qvel, -10, 10)
    end
    o
end


function LyceumMuJoCo.getreward(env::WalkerV2)
    x, height, ang = uview(env.sim.d.qpos, 1:3)
    alive_bonus = 1.0
    reward = (x - env.lastx) / effective_timestep(env)
    reward += alive_bonus
    reward -= 1e-3 * sum(x->x^2, env.sim.d.ctrl)
    reward
end


function LyceumBase.reset!(env::WalkerV2)
    reset!(env.sim)
    env.lastx = _getx(env.sim)
    env
end

function LyceumBase.reset!(env::WalkerV2, s)
    @uviews s begin
        shaped = env.statespace(s)
        reset!(env.sim, shaped.simstate)
        env.lastx = shaped.lastx
    end
    env
end

function LyceumMuJoCo.randreset!(env::WalkerV2)
    reset!(env)
    env.sim.d.qpos .+= rand.(env.randreset_distribution)
    env.sim.d.qvel .+= rand.(env.randreset_distribution)
    forward!(env.sim)
    env.lastx = _getx(env.sim)
    env
end


function LyceumMuJoCo.step!(env::WalkerV2)
    env.lastx = _getx(env.sim)
    step!(env.sim)
    env
end

function isdone(env::WalkerV2)
    qpos = env.sim.d.qpos
    qvel = env.sim.d.qvel
    x, height, ang = uview(qpos, 1:3)
    done = !(height > 0.8
        && height < 2.0
        && abs(ang) < 1.0
    )
    done
end

LyceumMuJoCo.geteval(env::WalkerV2) = _getx(env.sim)
