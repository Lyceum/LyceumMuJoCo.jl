module LyceumMuJoCo

using Base: @propagate_inbounds
using Distances
using Distributions
using DocStringExtensions
using LinearAlgebra

using MuJoCo
using MuJoCo.MJCore: mjtNum

using Shapes
using StaticArrays
using Statistics
using Random
using Reexport
using UnsafeArrays

using LyceumBase: @mustimplement, perturb!
@reexport using LyceumBase

const RealVec = AbstractVector{<:Real}


export MJSim, zeroctrl!, zerofullctrl!, forward!
include("mjsim.jl")

export AbstractMuJoCoEnvironment, getsim
include("abstractmujocoenvironment.jl")


####
#### Environments
####

include("suite/pointmass.jl")
include("suite/armhand/armhandpickup.jl")

include("gym/swimmer-v2.jl")
include("gym/hopper-v2.jl")
include("gym/walker2d-v2.jl")

include("dmc/rewards.jl")
include("dmc/cartpole_swingup.jl")

end # module
