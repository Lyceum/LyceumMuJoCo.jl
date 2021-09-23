"""
    $(TYPEDEF)

`PointMass` is a simple environment useful for trying out and debugging new algorithms. The
task is simply to move a 2D point mass to a target position by applying x and y forces
to the mass.

# Spaces

* **State: (13, )**
* **Action: (2, )**
* **Observation: (6, )**
"""
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

PointMass() = first(tconstruct(PointMass, 1))

function tconstruct(::Type{PointMass}, N::Integer)
    modelpath = joinpath(@__DIR__, "pointmass.xml")
    Tuple(PointMass(s) for s in tconstruct(MJSim, N, modelpath, skip=1))
end

@inline getsim(env::PointMass) = env.sim
@inline obsspace(env::PointMass) = env.obsspace

@propagate_inbounds function getobs!(obs, env::PointMass)
    @boundscheck checkaxes(obsspace(env), obs)
    dn = env.sim.dn
    shaped = obsspace(env)(obs)
    @inbounds begin
        shaped.agent_xy_pos .= dn.xpos[Val(:x), Val(:agent)], dn.xpos[Val(:y), Val(:agent)]
        shaped.agent_xy_vel .= dn.qvel[Val(:agent_x)], dn.qvel[Val(:agent_y)]
        shaped.target_xy_pos .= dn.xpos[Val(:x), Val(:target)], dn.xpos[Val(:y), Val(:target)]
    end
    obs
end

@propagate_inbounds function getreward(::Any, ::Any, obs, env::PointMass)
    @boundscheck checkaxes(obsspace(env), obs)
    shaped = obsspace(env)(obs)
    @inbounds begin
        1.0 - euclidean(shaped.agent_xy_pos, shaped.target_xy_pos)
    end
end

@propagate_inbounds function geteval(::Any, ::Any, obs, env::PointMass)
    @boundscheck checkaxes(obsspace(env), obs)
    shaped = obsspace(env)(obs)
    @inbounds begin
        euclidean(shaped.agent_xy_pos, shaped.target_xy_pos)
    end
end

@propagate_inbounds function randreset!(rng::Random.AbstractRNG, env::PointMass)
    reset_nofwd!(env.sim)
    @inbounds begin
        env.sim.dn.qpos[Val(:agent_x)] = rand(rng) * 2.0 - 1.0
        env.sim.dn.qpos[Val(:agent_y)] = rand(rng) * 2.0 - 1.0
        env.sim.dn.qpos[Val(:target_x)] = rand(rng) * 2.0 - 1.0
        env.sim.dn.qpos[Val(:target_y)] = rand(rng) * 2.0 - 1.0
    end
    forward!(env.sim)
    env
end
