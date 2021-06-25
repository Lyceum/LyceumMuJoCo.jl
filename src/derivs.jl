#using ChainRulesCore
import ChainRulesCore.rrule

# statespace, obsspace, actionspace, rewardspace, evalspace

# getstate
# setstate
# getaction
# setaction

# getobs(env) = obs, need dstate
# step, nextstate = step(state, controls), so need dstate, dcontrols
# getreward(state, controls, obs, env) = r, need dstate, dcontrols, dobs
# geteval

# reset!,
# randreset!,
# step!, basically nextstate = step(state, controls), so need dstate, dcontrols
# isdone(state, action, observation, env) --> Bool
# timestep, returns a const wrt state

# tconstruct

mutable struct derivativeholders
    # if current state = previous state, don't finite difference
    currentstate::Vector{mjtNum} # placeholder for calling getstate
    computedstate::Vector{mjtNum}

    dqaccdstate::Matrix{mjtNum}    # nv x state 
    dqaccdaction::Matrix{mjtNum}   # nv x action 

    dobsdstate::Matrix{mjtNum}     # obs x state
    dobsdaction::Matrix{mjtNum}    # obs x action

    drewarddstate::Matrix{mjtNum}  # 1 x state
    drewarddaction::Matrix{mjtNum} # 1 x action
    drewarddobs::Matrix{mjtNum}    # 1 x obs

    # fields to help finite difference
    center::Vector{mjtNum}    # nv
    warmstart::Vector{mjtNum} # nv
    nwarmup::Int # extra solver iterations to improve warmstart
    eps::mjtNum
end

function _getquat!(q, qpos, idx) # idx should be 0-index
    SVector{4, Float64}( @view qpos[idx+1:idx+4] )
end
function _setquat!(qpos, q, idx) # idx should be 0-index
    @view(qpos[idx+1:idx+4]) .= q
    qpos
end


# Goal:
# Lyceum allows higher level access to underlying physics, ie MDP vs dynamics
# Can't autodiff MuJoCo itself due to incompatible function handles.
#
# Allowing LyceumMuJoCo to be autodiff'd could work, but the interaction between an
# 'env' and the underlying 'sim' can get tricky, especially since we want to
# minimize finite-differencing.
#
# The only function that needs to be finite differenced is the abstract notion
# of 'dx = forward(x, a)' -- we defer inverse dynamics for later if needed.
# However, Lyceum has function such as getobs, getreward, step; we would prefer
# not to finite difference for each call if state x has not changed, hence the
# struct above for storing computation.
#
# One use case is as follows (borrowed from CTPG) in not Lyceum specific pseudocode
#
# function f(u, p, t) # handle for diffeq.jl forward function
#    setstate(env, u, t) # set some environments state according to u
#    o = getobs(env)     # get the observations of the env at the new state
#    a = policy(p, o)    # get action from policy with params p for some obs
#    setaction(env, a)   # set the env's actions
#    du = forward(env)   # calculate the forward dynamics and change in state
#    r = getreward(env)  # get the reward for the current state
#    vcat(du, r)         # propagate change in state and accumulate reward
# end
# 
# For something like trajectory optimization:
#
# function f(x, u)
#    setstate(env, x)
#    setaction(env, u)
#    du = forward(env)
#    integrate(env, du)
#    getreward(env)
# end
#
# So we can see that there's a variety of functions that when autodiff'd really
# all hit the same finite difference path, often without much changes.


# needs a skip value
# run forward on state, action
# if only action has changed, FD action
function switchfd(env::AbstractMuJoCoEnvironment)
    D = env.derivativeholders
    sim = env.sim
    m = sim.m
    d = sim.d
    nv = Int(sim.m.nv)

    getstate!(D.currentstate, env)
    if isapprox(D.computedstate, D.currentstate, rtol=1e-9) # TODO approximate comparison? 1e-9 tol?
        # we already did the finite difference, return
        return
    else
        copyto!(D.computedstate, D.currentstate)

        forward!(sim)
        for _=1:D.nwarmup
            forwardskip!(sim, MJCore.mjSTAGE_VEL)
        end

        #copyto!(D.center, sim.d.qacc)
        center = SVector{nv, mjtNum}(sim.d.qacc)
        warmstart = SVector{nv, mjtNum}(sim.d.qacc_warmstart) 
        centerobs = getobs(env) # TODO allocation!

        # iterate over state, action fields,
        #       calling mjforward, logging acc, obs, reward
        # finite-difference over velocity: skip = mjSTAGE_POS
        for i=1:nv
            # perturb velocity
            qvel_i = d.qvel[i]
            d.qvel[i] += eps

            # evaluate dynamics, with center warmstart
            copyto!(d.qacc_warmstart, warmstart)
            forwardskip!(sim, MJCore.mjSTAGE_POS)

            # undo perturbation
            d.qvel[i] = qvel_i

            dobs = @view D.dobsdstate[:, nv+i]
            getobs!(dobs, env)
            @. dobs = (dobs - centerobs) / eps

            dqacc = @view D.dqaccdstate[:, nv+i]
            @. dqacc = (d.qacc - center) / eps
        end

        # finite-difference over position: skip = mjSTAGE_NONE
        for i=1:nv
            # get joint id for this dof
            jid = m.dof_jntid[i] + 1 # +1 for julia

            # get quaternion address and dof position within quaternion (-1: not in quaternion)
            quatadr = -1
            if m.jnt_type[jid] == MJCore.mjJNT_BALL
                quatadr = Int(m.jnt_qposadr[jid])
                dofpos = i - m.jnt_dofadr[jid]
            elseif m.jnt_type[jid] == MJCore.mjJNT_FREE && i>=m.jnt_dofadr[jid]+4
                quatadr = Int(m.jnt_qposadr[jid]) + 3
                dofpos = i - m.jnt_dofadr[jid] - 3
            end

            # apply quaternion or simple perturbation
            if quatadr>=0
                angvel .= 0.0
                angvel[dofpos] = eps # already +1 from for loop's i
                quat = _getquat!(d.qpos, quatadr)
                MJCore.mju_quatIntegrate(quat, angvel, 1.0)
                # TODO mujoco.jl didn't expose mju_quatIntegrate, yet
                _setquat!(d.qpos, quat, quatadr)
            else
                d.qpos[i] += eps
            end

            # evaluate dynamics, with center warmstart
            copyto!(d.qacc_warmstart, warmstart)
            forwardskip!(sim, MJCore.mjSTAGE_NONE)

            # undo perturbation
            copyto!(d.qpos, dmain.qpos) # since quat

            dqacc = @view D.dqaccdstate[:, i]
            @. dqacc = (d.qacc - center) / eps
        end
    end
end

function rrule(::typeof(getstate!), s, sim::MJSim)
    getstate!(s, sim)
    getstate_pullback(Δs) = (NO_FIELDS, Δs, DoesNotExist())
    s, getstate_pullback
end
function rrule(::typeof(getstate), sim::MJSim)
    s = getstate(sim)
    getstate_pullback(Δs) = (NO_FIELDS, Δs)
    s, getstate_pullback
end

# o = getobs(sim.state); returns do/ds
function rrule(::typeof(getobs!), o, sim::MJSim)
    getobs!(o, sim)
    getobs_pullback(Δs) = (NO_FIELDS, sim.dx.dods * Δs', DoesNotExist())
    s, getobs_pullback
end

# maybe the some of the following should be no-grads?
#=
function rrule(::typeof(getstate), env::AbstractMuJoCoEnvironment)
    s = getstate(env)
    getstate_pullback(Δs) = (NO_FIELDS, Δs)
    s, getstate_pullback
end

function rrule(::typeof(getstate!), s, env::AbstractMuJoCoEnvironment)
    getstate!(s, env)
    getstate_pullback(Δs) = (NO_FIELDS, Δs, DoesNotExist())
    s, getstate_pullback
end
=#

# TODO this changes env/mujoco
function rrule(::typeof(setstate!), env::AbstractMuJoCoEnvironment, s)
    setstate!(env, s)
    setstate_pullback(Δs) = (NO_FIELDS, DoesNotExist(), Δs)
    s, setstate_pullback
end

function rrule(::typeof(getaction), env::AbstractMuJoCoEnvironment)
    a = getaction(env)
    getaction_pullback(Δa) = (NO_FIELDS, Δa)
    a, getaction_pullback
end

function rrule(::typeof(getaction!), a, env::AbstractMuJoCoEnvironment)
    a = getaction!(a, env)
    getaction_pullback(Δa) = (NO_FIELDS, Δa, DoesNotExist())
    a, getaction_pullback
end

# TODO this changes env/mujoco, and some models may have additional autodiff-able code?
function rrule(::typeof(setaction!), env::AbstractMuJoCoEnvironment, a)
    setaction!(env, a)
    setaction_env_pullback(Δa) = (NO_FIELDS, DoesNotExist(), Δa)
    env, setaction_env_pullback
end

function rrule(::typeof(setaction!), sim::MJSim, a)
    setaction!(sim, a)
    setaction_sim_pullback(Δa) = (NO_FIELDS, DoesNotExist(), Δa)
    sim, setaction_sim_pullback
end




function rrule(::typeof(getreward), s, a, o, env::AbstractMuJoCoEnvironment)
    r = getreward(s, a, o, env)
    function getreward_pullback(Δr) = (NO_FIELDS, dstate'*Δr, daction'*Δr, dobs'*Δr, DoesNotExist())
    r, getreward_pullback
end
