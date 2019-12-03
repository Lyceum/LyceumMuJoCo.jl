struct SwimmerV2{SIM <: MJSim, S <: AbstractShape, O <: AbstractShape} <: AbstractMuJoCoEnv
    sim::SIM
    statespace::S
    observationspace::O
    last_xy_position::MVector{2, Float64}
    randreset_distribution::Uniform{Float64}

    function SwimmerV2(sim::SIM) where {SIM <: MJSim}
        sspace = MultiShape(statespace(sim), last_xy_position=VectorShape(Float64, 2))
        ospace = MultiShape(qpos=statespace(sim).qpos, qvel=statespace(sim).qvel)
        last_xy_position = MVector{2, Float64}(sim.d.qpos[1:2])
        new{SIM, typeof(sspace), typeof(ospace)}(
            sim,
            sspace,
            ospace,
            last_xy_position,
            Uniform(-0.1, 0.1)
        )
    end
end

SwimmerV2() = SwimmerV2(MJSim(MJSimParameters(SwimmerV2)))

MJSimParameters(::Type{<:SwimmerV2}) = MJSimParameters(joinpath(@__DIR__, "swimmer-v2.xml"), skip=4)

getsim(env::SwimmerV2) = env.sim


statespace(env::SwimmerV2) = env.statespace
function getstate!(s, env::SwimmerV2)
    @uviews s begin
        simstate = view(s, 1:length(statespace(env.sim)))
        getstate!(simstate, env.sim)
        copyto!(env.statespace(s).last_xy_position, env.last_xy_position)
    end
    s
end


observationspace(env::SwimmerV2) = env.observationspace
function getobs!(o, env::SwimmerV2)
    @uviews o begin
        shaped = env.observationspace(o)
        copyto!(shaped.qpos, env.sim.d.qpos)
        copyto!(shaped.qvel, env.sim.d.qvel)
    end
    o
end


function getreward(env::SwimmerV2)
    reward_fwd = (xy_position(env)[1] - env.last_xy_position[1]) / effective_timestep(env)
    reward_ctrl = -1e-4 * sum(x->x^2, env.sim.d.ctrl)
    reward_fwd + reward_ctrl
end

geteval(env::SwimmerV2) = xy_position(env)[1]


function reset!(env::SwimmerV2)
    reset!(env.sim)
    copyto!(env.last_xy_position, xy_position(env))
    env
end

function reset!(env::SwimmerV2, s)
    @uviews s begin
        simstate = view(s, 1:length(statespace(env.sim)))
        reset!(env.sim, simstate)
        copyto!(env.last_xy_position, env.statespace(s).last_xy_position)
    end
    env
end

function randreset!(env::SwimmerV2)
    s = env.sim
    reset!(s)

    qpos_noise = similar_type(MVector, statespace(s).qpos)(undef)
    qvel_noise = similar_type(MVector, statespace(s).qvel)(undef)
    rand!(env.randreset_distribution, qpos_noise)
    rand!(env.randreset_distribution, qvel_noise)
    s.d.qpos .+= qpos_noise
    s.d.qvel .+= qvel_noise

    forward!(s)

    copyto!(env.last_xy_position, xy_position(env))

    env
end

#function step!(env::SwimmerV2, a)
#    copyto!(env.last_xy_position, xy_position(env))
#
#    x1 = env.sim.d.qpos[1]
#    setaction!(env.sim, a)
#    step!(env)
#    x2 = env.sim.d.qpos[1]
#    reward_fwd = (x2 - x1) / timestep(env)
#    reward_ctrl = -1e-4 * sum(x->x^2, env.sim.d.ctrl)
#    reward_fwd + reward_ctrl
#end

function step!(env::SwimmerV2)
    copyto!(env.last_xy_position, xy_position(env))
    step!(env.sim)
    env
end

# utils
xy_position(env::SwimmerV2) = SVector{2, Float64}(uview(env.sim.d.qpos, 1:2))