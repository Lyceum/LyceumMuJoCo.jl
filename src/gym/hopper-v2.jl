mutable struct HopperV2{SIM, S, O} <: AbstractMuJoCoEnv
    sim::SIM
    statespace::S
    observationspace::O
    lastx::Float64
    randreset_distribution::Uniform{Float64}
    function HopperV2(sim::MJSim)
        sspace = MultiShape(statespace(sim), lastx=ScalarShape(Float64))
        ospace = VectorShape{Int(sim.m.nq+sim.m.nv-1), Float64}()
        new{typeof(sim), typeof(sspace), typeof(ospace)}(
            sim, sspace, ospace, _getx(sim), Uniform(-0.005, 0.005)
        )
    end
end

HopperV2() = first(thread_constructor(HopperV2, 1))

function thread_constructor(::Type{<:HopperV2}, N::Integer)
    modelpath = joinpath(@__DIR__, "hopper-v2.xml")
    Tuple(HopperV2(s) for s in thread_constructor(MJSim, N, modelpath, skip=4))
end


getsim(env::HopperV2) = env.sim


statespace(env::HopperV2) = env.statespace
function getstate!(s, env::HopperV2)
    @uviews s begin
        simstate = view(s, 1:length(statespace(env.sim)))
        getstate!(simstate, env.sim)
        env.statespace(s).lastx = env.lastx
    end
    s
end


observationspace(env::HopperV2) = env.observationspace
function getobs!(o, env::HopperV2)
    nq = env.sim.m.nq
    qpos = env.sim.d.qpos
    qvel = env.sim.d.qvel
    @views @uviews qpos qvel o begin
        l = length(qpos[2:end])
        o .= [qpos[2:end]..., clamp!(copy(qvel), -10, 10)...]
    end
    o
end


function getreward(env::HopperV2)
    qpos = env.sim.d.qpos
    qvel = env.sim.d.qvel
    x, height, ang = uview(qpos, 1:3)

    alive_bonus = 1.0
    reward = (x - env.lastx) / effective_timestep(env)
    reward += alive_bonus
    reward -= 1e-3 * sum(x->x^2, env.sim.d.ctrl)
    reward
end


function reset!(env::HopperV2)
    reset!(env.sim)
    env.lastx = _getx(env.sim)
    env
end

function reset!(env::HopperV2, s)
    @uviews s begin
        simstate = view(s, 1:length(statespace(env.sim)))
        reset!(env.sim, simstate)
        env.lastx = _getx(env.sim)
    end
    env
end

function randreset!(env::HopperV2)
    reset!(env)
    env.sim.d.qpos .+= rand.(env.randreset_distribution)
    env.sim.d.qvel .+= rand.(env.randreset_distribution)
    forward!(env.sim)
    env.lastx = _getx(env.sim)
    env
end


function step!(env::HopperV2)
    env.lastx = _getx(env.sim)
    step!(env.sim)
    env
end

function isdone(env::HopperV2)
    qpos = env.sim.d.qpos
    qvel = env.sim.d.qvel
    x, height, ang = uview(qpos, 1:3)
    s = getstate(env.sim)
    done = !(
        any(isinf, s)
        && all(x->abs(x) < 100, uview(s, 3:length(s)))
        && height > 0.7
        && abs(ang) < 0.2
    )
    done
end

geteval(env::HopperV2) = _getx(env.sim)

_getx(sim::MJSim) = sim.d.qpos[1]
_getheight(sim::MJSim) = sim.d.qpos[2]
_getang(sim::MJSim) = sim.d.qpos[3]
@inline wraptopi(theta) = mod((theta + pi), (2 * pi)) - pi