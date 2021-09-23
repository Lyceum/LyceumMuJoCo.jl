# Copyright 2017 The dm_control Authors.
# Copyright (c) 2019 Colin Summers, The Contributors of LyceumMuJoCo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
    $(TYPEDEF)

Swing up and balance an unactuated pole by applying forces to a cart at its base. The
physical model conforms to [Neuronlike adaptive elements that can solve difficult learning
control problems](https://ieeexplore.ieee.org/document/6313077) (Barto et al., 1983).

# Spaces

* **State: (7, )**
* **Action: (1, )**
* **Observation: (5, )**
"""
struct CartpoleSwingup{S<:MJSim,O<:MultiShape} <: AbstractMuJoCoEnvironment
    sim::S
    obsspace::O
    function CartpoleSwingup(sim::MJSim)
        obsspace = MultiShape(
            pos = MultiShape(
                cart = ScalarShape(Float64),
                pole_zz = ScalarShape(Float64),
                pole_xz = ScalarShape(Float64),
            ),
            vel = MultiShape(
                cart = ScalarShape(Float64),
                pole_ang = ScalarShape(Float64),
            ),
        )
        env = new{typeof(sim), typeof(obsspace)}(sim, obsspace)
        reset!(env)
    end
end

CartpoleSwingup() = first(tconstruct(CartpoleSwingup, 1))

function tconstruct(::Type{CartpoleSwingup}, n::Integer)
    modelpath = joinpath(@__DIR__, "cartpole.xml")
    Tuple(CartpoleSwingup(s) for s in tconstruct(MJSim, n, modelpath))
end


@inline getsim(env::CartpoleSwingup) = env.sim
@inline obsspace(env::CartpoleSwingup) = env.obsspace

@inline function getobs!(obs, env::CartpoleSwingup)
    @boundscheck checkaxes(obsspace(env), obs)

    sobs = obsspace(env)(obs)
    sobs.pos.cart = env.sim.dn.qpos[Val(:slider)]
    sobs.pos.pole_zz = env.sim.dn.xmat[Val(:z), Val(:z), Val(:pole_1)]
    sobs.pos.pole_xz = env.sim.dn.xmat[Val(:x), Val(:z), Val(:pole_1)]
    copyto!(sobs.vel, env.sim.d.qvel)
    obs
end

@inline function getreward(state, action, obs, env::CartpoleSwingup)
    @boundscheck begin
        checkaxes(statespace(env), state)
        checkaxes(actionspace(env), action)
        checkaxes(obsspace(env), obs)
    end

    sobs = obsspace(env)(obs)

    upright = (sobs.pos.pole_zz + 1) / 2

    centered = tolerance(sobs.pos.cart, margin = 2)
    centered = (1 + centered) / 2

    small_control = tolerance(first(action), margin = 1, value_at_margin = 0, sigmoid = quadratic)
    small_control = (4 + small_control) / 5

    small_velocity = min(tolerance(sobs.vel.pole_ang, margin = 5))
    small_velocity = (1 + small_velocity) / 2

    mean(upright) * small_control * small_velocity * centered
end

@inline function geteval(state, action, obs, env::CartpoleSwingup)
    @boundscheck begin
        checkaxes(obsspace(env), obs)
    end
    obsspace(env)(obs).pos.pole_zz
end

function reset!(env::CartpoleSwingup)
    reset_nofwd!(env.sim)

    qpos = env.sim.dn.qpos
    qpos[Val(:hinge_1)] = pi

    forward!(env.sim)

    env
end

function randreset!(rng::Random.AbstractRNG, env::CartpoleSwingup)
    reset_nofwd!(env.sim)

    qpos = env.sim.dn.qpos
    qpos[Val(:slider)] = 0.01 * randn(rng)
    qpos[Val(:hinge_1)] = pi + 0.01*randn(rng)

    randn!(rng, env.sim.d.qvel)
    env.sim.d.qvel .*= 0.01

    forward!(env.sim)

    env
end
