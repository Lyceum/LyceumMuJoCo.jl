mutable struct SwimmerV2{SIM <: MJSim, S <: AbstractShape, O <: AbstractShape} <: AbstractMuJoCoEnvironment
    sim::SIM
    statespace::S
    observationspace::O
    lastxp::Float64
    randreset_distribution::Uniform{Float64}

    function SwimmerV2(sim::SIM) where {SIM <: MJSim}
        sspace = MultiShape(simstate=statespace(sim), lastxp=ScalarShape(Float64))
        ospace = MultiShape(
            qpos_cropped = VectorShape(Float64, sim.m.nq - 2),
            qvel=statespace(sim).qvel
        )
        new{SIM, typeof(sspace), typeof(ospace)}(
            sim,
            sspace,
            ospace,
            sim.d.qpos[1],
            Uniform(-0.1, 0.1)
        )
    end
end

SwimmerV2() = first(sharedmemory_envs(SwimmerV2))

function LyceumBase.sharedmemory_envs(::Type{<:SwimmerV2}, n::Integer = 1)
    model = joinpath(@__DIR__, "swimmer-v2.xml")
    Tuple(SwimmerV2(s) for s in sharedmemory_mjsims(model, n, skip=5))
end

getsim(env::SwimmerV2) = env.sim

statespace(env::SwimmerV2) = env.statespace
function getstate!(s, env::SwimmerV2)
    @uviews s begin
        shaped = env.statespace(s)
        getstate!(shaped.simstate, env.sim)
        shaped.lastxp = env.lastxp
    end
    s
end


observationspace(env::SwimmerV2) = env.observationspace
function getobs!(o, env::SwimmerV2)
    qpos = env.sim.d.qpos
    @views @uviews o qpos begin
        shaped = env.observationspace(o)
        copyto!(shaped.qpos_cropped, env.sim.d.qpos[3:end])
        copyto!(shaped.qvel, env.sim.d.qvel)
    end
    o
end


function getreward(env::SwimmerV2)
    reward_fwd = (xp(env) - env.lastxp) / effective_timestep(env)
    reward_ctrl = -1e-4 * sum(x->x^2, env.sim.d.ctrl)
    reward_fwd + reward_ctrl
end

geteval(env::SwimmerV2) = xp(env)


function reset!(env::SwimmerV2)
    reset!(env.sim)
    env.lastxp = xp(env)
    env
end

function reset!(env::SwimmerV2, s)
    @uviews s begin
        shaped = env.statespace(s)
        reset!(env.sim, shaped.simstate)
        env.lastxp = shaped.lastxp
    end
    forward!(env)
    env
end

function randreset!(env::SwimmerV2)
    reset!(env.sim)
    @. env.sim.d.qpos = rand(env.randreset_distribution)
    @. env.sim.d.qvel = rand(env.randreset_distribution)
    forward!(env.sim)
    env.lastxp = xp(env)
    env
end

function step!(env::SwimmerV2)
    env.lastxp = xp(env)
    step!(env.sim)
    env
end

# x-position of Swimmer's torso
xp(env::SwimmerV2) = env.sim.d.qpos[1]