using LinearAlgebra, Random, Statistics
using Plots, UnicodePlots, JLSO
using LyceumBase, LyceumBase.Tools, LyceumAI, LyceumMuJoCo, MuJoCo, UniversalLogger, Shapes
using Distributions

seed_threadrngs!(1)
const _DEFAULT_VALUE_AT_MARGIN = 0.1

struct Cartpole{S<:MJSim} <: AbstractMuJoCoEnvironment
    sim::S
end

# LyceumMuJoCo.getsim(env::Cartpole) = env.sim #src (needs to be here for below example to work)

# modelpath = joinpath(@__DIR__, "cartpole.xml")
# envs = [Cartpole(MJSim(modelpath, skip = 2)) for i = 1:Threads.nthreads()]
# Threads.@threads for i = 1:Threads.nthreads()
#     thread_env = envs[Threads.threadid()]
#     step!(thread_env)
# end

Cartpole() = first(tconstruct(Cartpole, 1))
function LyceumMuJoCo.tconstruct(::Type{Cartpole}, n::Integer)
    modelpath = joinpath(@__DIR__, "cartpole.xml")
    return Tuple(Cartpole(s) for s in tconstruct(MJSim, n, modelpath, skip = 2))
end

# envs = tconstruct(Cartpole, Threads.nthreads())
# Threads.@threads for i = 1:Threads.nthreads()
#     thread_env = envs[Threads.threadid()]
#     step!(thread_env)
# end

_getcosangle(shapedstate::ShapedView, env::Cartpole) = (cos(shapedstate.qpos[2])+1) / 2

LyceumMuJoCo.getsim(env::Cartpole) = env.sim

##########
function _sigmoids(x::Any, value_at_1::T, sigmoid::String) where T<:AbstractFloat

  if sigmoid == "gaussian"
    scale = sqrt(-2 * log(value_at_1))
    return exp.(-0.5 * (x.*scale).^2)

  elseif sigmoid == "hyperbolic"
    scale = acosh(1/value_at_1)
    return 1 ./ cosh(x.*scale)

  elseif sigmoid == "long_tail"
    scale = sqrt(1/value_at_1 - 1)
    return 1 ./ ((x.*scale).^2 + 1)

  elseif sigmoid == "cosine"
    scale = acos(2*value_at_1 - 1) / pi
    scaled_x = x.*scale
    return ifelse.(abs.(scaled_x) .< 1, (1 .+ cos(pi.*scaled_x))./2, 0.0)

  elseif sigmoid == "linear"
    scale = 1-value_at_1
    scaled_x = x.*scale
    return ifelse.(abs.(scaled_x) .< 1, 1 .- scaled_x, 0.0)

  elseif sigmoid == "quadratic"
    scale = sqrt(1-value_at_1)
    scaled_x = x.*scale
    return ifelse.(abs.(scaled_x) .< 1, 1 .- scaled_x .^2, 0.0)

  elseif sigmoid == "tanh_squared"
    scale = atanh(sqrt(1-value_at_1))
    return 1 .- tanh(x.*scale).^2

  end

end

function tolerance(x::Any; bounds=(0.0, 0.0), margin=0.0, sigmoid="gaussian",
                    value_at_margin=_DEFAULT_VALUE_AT_MARGIN)

    lower, upper = bounds
   

    in_bounds = (lower .<= x) .& (x .<= upper)
    if margin == 0
        value = ifelse.(in_bounds, 1.0, 0.0)
    else
        d = ifelse.(x .< lower, lower .- x, x .- upper) ./ margin
        value = ifelse.(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))
    end

    return float(value)

end



##########

# function LyceumMuJoCo.reset!(env::Cartpole)
#     reset!(env.sim)
#     #print(env.sim.d.qpos)
#     env.sim.d.qpos[1] = .01*randn()
#     env.sim.d.qpos[2] = .01*randn()
#     env.sim.d.qpos[3:end] = .1*randn(size(env.sim.d.qpos[3:end]))
#     #env.sim.d.qpos .= LAYING_QPOS
#     forward!(env.sim)
#     return env
# end

function LyceumMuJoCo.randreset!(rng::Random.AbstractRNG, env::Cartpole)
    reset!(env.sim)
    #env.sim.d.qpos[1] = .01*randn()
    env.sim.d.qpos[2] = rand(rng, Uniform(-pi, pi))
    forward!(env.sim)
    return env
end

function LyceumMuJoCo.getreward(state, action, obs, env::Cartpole)
    upright = _getcosangle(statespace(env)(state), env)
    centered = tolerance(statespace(env)(state).qpos[1], margin=2.)
    centered = (1 + centered) / 2
    small_control = tolerance(action, margin=1,
                                      value_at_margin=0.,
                                      sigmoid="quadratic")[1]
    small_control = (4 + small_control) / 5
    small_velocity = min(tolerance(statespace(env)(state).qvel[2], margin=5.))
    small_velocity = (1 + small_velocity) / 2
    return mean(upright) * 2*centered * 2small_velocity #small_control * small_velocity * centered
end

function LyceumMuJoCo.geteval(state, action, obs, env::Cartpole)
    return _getcosangle(statespace(env)(state), env)
end

function cartpole_MPPI(etype = Cartpole; T = 200, H = 64, K = 64)
    env = etype()

    # The following parameters work well for this get-up task, and may work for
    # similar tasks, but are not invariant to the model.
    mppi = MPPI(
        env_tconstructor = n -> tconstruct(etype, n),
        covar0 = Diagonal(0.05^2 * I, size(actionspace(env), 1)),
        lambda = 0.2,
        H = H,
        K = K,
        gamma = 1.0,
    )

    iter = ControllerIterator(mppi, env; T = T, plotiter = div(T, 10), randstart=true)

    # We can time the following loop; if it ends up less than the time the
    # MuJoCo models integrated forward in, then one could conceivably run this
    # MPPI MPC controller interactively...
    elapsed = @elapsed for (t, traj) in iter
        # If desired, one can inspect `traj`, `env`, or `mppi` at each timestep.
    end

    if elapsed < time(env)
        @info "We ran in real time!"
    end

    # Save our experiment results to a file for later review.
    savepath = "/tmp/opt_cartpole.jlso"
    exper = Experiment(savepath, overwrite = true)
    exper[:etype] = etype

    for (k, v) in pairs(iter.trajectory)
        exper[k] = v
    end
    finish!(exper)

    return mppi, env, iter.trajectory
end

function viz_mppi(mppi::MPPI, env::AbstractMuJoCoEnvironment)
    a = allocate(actionspace(env))
    o = allocate(obsspace(env))
    s = allocate(statespace(env))
    ctrlfn = @closure env -> (getstate!(s, env); getaction!(a, s, o, mppi); setaction!(env, a))
    visualize(env, controller=ctrlfn)
end


mppi, env, traj = cartpole_MPPI();
plot(
    [traj.rewards traj.evaluations],
    labels = ["Reward" "Evaluation"],
    title = "Cartpole Swingup",
    legend = :bottomright,
)

data = JLSO.load("/tmp/opt_cartpole.jlso")
plot(
    [data["rewards"] data["evaluations"]],
    labels = ["Reward" "Evaluation"],
    title = "Cartpole Swingup",
    legend = :bottomright,
)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

