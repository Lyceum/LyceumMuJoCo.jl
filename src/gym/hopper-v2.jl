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


HopperV2() = HopperV2(MJSim(MJSimParameters(HopperV2)))

MJSimParameters(::Type{<:HopperV2}) = MJSimParameters(joinpath(@__DIR__, "hopper-v2.xml"), skip=4)

LyceumMuJoCo.getsim(env::HopperV2) = env.sim


LyceumBase.statespace(env::HopperV2) = env.statespace
function LyceumBase.getstate!(s, env::HopperV2)
    @uviews s begin
        simstate = view(s, 1:length(statespace(env.sim)))
        getstate!(simstate, env.sim)
        env.statespace(s).lastx = env.lastx
    end
    s
end


LyceumMuJoCo.observationspace(env::HopperV2) = env.observationspace
function LyceumMuJoCo.getobs!(o, env::HopperV2)
    nq = env.sim.m.nq
    qpos = env.sim.d.qpos
    qvel = env.sim.d.qvel
    @views @uviews qpos qvel o begin
        l = length(qpos[2:end])
        o .= [qpos[2:end]..., clamp!(copy(qvel), -10, 10)...]
    end
    o
end


function LyceumMuJoCo.getreward(env::HopperV2)
    qpos = env.sim.d.qpos
    qvel = env.sim.d.qvel
    x, height, ang = uview(qpos, 1:3)

    alive_bonus = 1.0
    reward = (x - env.lastx) / effective_timestep(env)
    reward += alive_bonus
    reward -= 1e-3 * sum(x->x^2, env.sim.d.ctrl)
    reward
end


function LyceumBase.reset!(env::HopperV2)
    reset!(env.sim)
    env.lastx = _getx(env.sim)
    env
end

function LyceumBase.reset!(env::HopperV2, s)
    @uviews s begin
        simstate = view(s, 1:length(statespace(env.sim)))
        reset!(env.sim, simstate)
        env.lastx = _getx(env.sim)
    end
    env
end

function LyceumMuJoCo.randreset!(env::HopperV2)
    reset!(env)
    env.sim.d.qpos .+= rand.(env.randreset_distribution)
    env.sim.d.qvel .+= rand.(env.randreset_distribution)
    forward!(env.sim)
    env.lastx = _getx(env.sim)
    env
end


function LyceumMuJoCo.step!(env::HopperV2)
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

LyceumMuJoCo.geteval(env::HopperV2) = _getx(env.sim)

_getx(sim::MJSim) = sim.d.qpos[1]
_getheight(sim::MJSim) = sim.d.qpos[2]
_getang(sim::MJSim) = sim.d.qpos[3]
@inline wraptopi(theta) = mod((theta + pi), (2 * pi)) - pi