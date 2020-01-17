struct Cartpole{S<:MJSim} <: AbstractMuJoCoEnvironment
    sim::S
end

Cartpole() = first(tconstruct(Cartpole, 1))

function tconstruct(::Type{Cartpole}, n::Integer)
    modelpath = joinpath(@__DIR__, "cartpole.xml")
    return Tuple(Cartpole(s) for s in tconstruct(MJSim, n, modelpath, skip = 2))
end


@inline getsim(env::Cartpole) = env.sim


function reset!(env::Cartpole)
    reset!(env.sim)
    qpos = env.sim.dn.qpos
    @uviews qpos begin
        qpos[:hinge_1] = pi
    end
    forward!(env.sim)
    env
end

function randreset!(rng::Random.AbstractRNG, env::Cartpole)
    reset!(env.sim)
    qpos = env.sim.dn.qpos
    @uviews qpos begin
        qpos[:slider] = 0.01 * randn(rng)
        #qpos[:hinge_1] = pi + 0.01 * randn(rng)
        qpos[:hinge_1] = rand(rng, Uniform(-pi, pi))

        randn!(rng, @view qpos[3:end])
        qpos[3:end] .*= 0.01
    end
    forward!(env.sim)
    env
end

obsspace(env::Cartpole) = VectorShape(Float64, 4)
function getobs!(obs, env::Cartpole)
    obs[1:2] .= env.sim.d.qpos
    obs[3:end] .= env.sim.d.qvel
end

function getreward(state, action, obs, env::Cartpole)
    shapedstate = statespace(env)(state)

    upright = _pole_angle_cosine(shapedstate, env)

    centered = tolerance(_cart_position(shapedstate, env), margin = 2)
    centered = (1 + centered) / 2

    small_control = tolerance(first(action), margin = 1, value_at_margin = 0, sigmoid = quadratic)
    small_control = (4 + small_control) / 5

    small_velocity = min(tolerance(_angular_vel(shapedstate, env), margin = 5.0))
    small_velocity = (1 + small_velocity) / 2

    mean(upright) * small_control * small_velocity * 2centered
    #return mean(upright) * 2*centered * 2small_velocity #small_control * small_velocity * centered
end

function geteval(state, action, obs, env::Cartpole)
    _pole_angle_cosine(statespace(env)(state), env)
end



function _pole_angle_cosine(shapedstate::ShapedView, env::Cartpole)
    (cos(shapedstate.qpos[2]) + 1) / 2 # TODO
end

#_angular_vel(shapedstate::ShapedView, env::Cartpole) = shapedstate.qvel[2:end]
_angular_vel(shapedstate::ShapedView, env::Cartpole) = shapedstate.qvel[2]

_cart_position(shapedstate::ShapedView, env::Cartpole) = shapedstate.qpos[1]



function _sigmoids(x, value_at_1, sigmoid) where {T<:AbstractFloat}

    if sigmoid == "gaussian"
        scale = sqrt(-2 * log(value_at_1))
        return exp.(-0.5 * (x .* scale) .^ 2)

    elseif sigmoid == "hyperbolic"
        scale = acosh(1 / value_at_1)
        return 1 ./ cosh(x .* scale)

    elseif sigmoid == "long_tail"
        scale = sqrt(1 / value_at_1 - 1)
        return 1 ./ ((x .* scale) .^ 2 + 1)

    elseif sigmoid == "cosine"
        scale = acos(2 * value_at_1 - 1) / pi
        scaled_x = x .* scale
        return ifelse.(abs.(scaled_x) .< 1, (1 .+ cos(pi .* scaled_x)) ./ 2, 0.0)

    elseif sigmoid == "linear"
        scale = 1 - value_at_1
        scaled_x = x .* scale
        return ifelse.(abs.(scaled_x) .< 1, 1 .- scaled_x, 0.0)


    elseif sigmoid == "tanh_squared"
        scale = atanh(sqrt(1 - value_at_1))
        return 1 .- tanh(x .* scale) .^ 2

    end

end
