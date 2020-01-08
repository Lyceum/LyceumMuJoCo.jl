struct PointMass{S <: MJSim, O} <: AbstractMuJoCoEnvironment
    sim::S
    obsspace::O
    function PointMass(sim::MJSim)
        obsspace = MultiShape(
            agent_xy_pos = VectorShape(mjtNum, 2),
            agent_xy_vel = VectorShape(mjtNum, 2),
            target_xy_pos = VectorShape(mjtNum, 2)
        )
        new{typeof(sim), typeof(obsspace)}(sim, obsspace)
    end
end

function tconstruct(::Type{PointMass}, N::Integer)
    modelpath = joinpath(@__DIR__, "pointmass.xml")
    Tuple(PointMass(s) for s in tconstruct(MJSim, N, modelpath, skip=1))
end

PointMass() = first(tconstruct(PointMass, 1))


@inline getsim(env::PointMass) = env.sim


@inline obsspace(env::PointMass) = env.obsspace

@propagate_inbounds function getobs!(obs, env::PointMass)
    @boundscheck checkaxes(obsspace(env), obs)
    dn = env.sim.dn
    shaped = obsspace(env)(obs)
    @uviews shaped @inbounds begin
        shaped.agent_xy_pos .= dn.xpos[:x, :agent], dn.xpos[:y, :agent]
        shaped.agent_xy_vel .= dn.qvel[:agent_x], dn.qvel[:agent_y]
        shaped.target_xy_pos .= dn.xpos[:x, :target], dn.xpos[:y, :target]
    end
    obs
end


@propagate_inbounds function getreward(::Any, ::Any, obs, env::PointMass)
    @boundscheck checkaxes(obsspace(env), obs)
    shaped = obsspace(env)(obs)
    @uviews shaped @inbounds begin
        1.0 - euclidean(shaped.agent_xy_pos, shaped.target_xy_pos)
    end
end


@propagate_inbounds function geteval(::Any, ::Any, obs, env::PointMass)
    @boundscheck checkaxes(obsspace(env), obs)
    shaped = env.obsspace(obs)
    @uviews shaped @inbounds begin
        euclidean(shaped.agent_xy_pos, shaped.target_xy_pos)
    end
end


@propagate_inbounds function randreset!(rng::Random.AbstractRNG, env::PointMass)
    reset_nofwd!(env.sim)
    @inbounds begin
        env.sim.dn.qpos[:agent_x] = rand(rng) * 2.0 - 1.0
        env.sim.dn.qpos[:agent_y] = rand(rng) * 2.0 - 1.0
        env.sim.dn.qpos[:target_x] = rand(rng) * 2.0 - 1.0
        env.sim.dn.qpos[:target_y] = rand(rng) * 2.0 - 1.0
    end
    forward!(env.sim)
    env
end