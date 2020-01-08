####
#### Core interface
####

const MJSTATE_FIELDS = (:time, :qpos, :qvel, :act, :mocap_pos, :mocap_quat, :userdata, :qacc_warmstart)
const MJACTION_FIELDS = (:ctrl, :qfrc_applied, :xfrc_applied)

const DEFAULT_SKIP = 1
const DEFAULT_CAN_SHARE_MODEL = true

"""
    MJSim

The `MJSim` type couples a `jlModel` and `jlData` from MuJoCo.jl to provide a full simulation.

The following are the official/internal/minimum set of fields from `jlData` for
state, observation, and action in MuJoCo:

- State: `(time, qpos, qvel, act, mocap_pos, mocap_quat, userdata, qacc_warmstart)`
- Observation: `sensordata`
- Action: `(ctrl, qfrc_applied, xfrc_applied)`

MJSim follows this convention.

For more information, see the "State and control" section of www.mujoco.org/book/programming.html

# Constructors
    MJSim(m::jlModel, d::jlData; skip::Integer = 1)
    MJSim(modelpath::String; skip::Integer = 1)

Construct a `MJSim` from either `m` and `d` or the MJCF/MJB model file located
at `modelpath`, with skip `skip`.

See MuJoCo.jl and the MuJoCo documentation (www.mujoco.org) for further
documentation of `jlData`, `jlModel`, as well as the MJCF and MJB file formats.

# Fields
- `m::jlModel`: contains the model description and is expected to remain constant.
- `d::jlData`: contains all dynamic variables and intermediate results.
- `mn::Tuple`: named-access version of `MJSim.m` provided by AxisArrays.jl.
- `dn::Tuple`: named-access version of `MJSim.d` provided by AxisArrays.jl.
- `skip::Int`: the number of times the simulation is integrated, yielding an
  effective simulation timestep of `skip * m.opt.timestep`.
- `initstate::Vector{mjtNum}`: The initial state vector at the time when this MJSim was constructed.
"""
struct MJSim{MN, DN, S, SE, A}
    m::jlModel
    d::jlData
    mn::MN
    dn::DN
    initstate::Vector{mjtNum}
    skip::Int

    # internal
    statespace::S
    sensorspace::SE
    actionspace::A

    function MJSim(m::jlModel, d::jlData; skip::Integer = DEFAULT_SKIP)
        check_skip(skip)

        m_named, d_named = namify(m, d)

        nameshapes = map(MJSTATE_FIELDS) do name
            field = getproperty(d, name)
            name => Shape(eltype(field), size(field)...)
        end
        statespace = MultiShape(nameshapes...)

        if m.nsensor > 0
            nameshapes = map(1:m.nsensor) do id
              name = jl_id2name(m, MJCore.mjOBJ_SENSOR, id)
              name = isnothing(name) ? Symbol("unnamed_$id") : Symbol(name)
              dof = m.sensor_dim[id]
              dof == 1 ? ScalarShape(mjtNum) : VectorShape(mjtNum, dof)
            end
            sensorspace = MultiShape(nameshapes...)
        else
            sensorspace = VectorShape(mjtNum, 0)
        end

        if m.nu > 0
            clampctrl = !jl_disabled(m, MJCore.mjDSBL_CLAMPCTRL)
            nameshapes = map(1:m.nu) do id
                name = jl_id2name(m, MJCore.mjOBJ_ACTUATOR, id)
                name = isnothing(name) ? Symbol("unnamed_$id") : Symbol(name)
                name => ScalarShape(mjtNum) #TODO boundedshape
            end
            actionspace = MultiShape(nameshapes...)
        else
            actionspace = VectorShape(mjtNum, 0)
        end

        initstate = mapreduce(vcat, MJSTATE_FIELDS) do name
            field = getproperty(d, name)
            ndims(field) > 1 ? vec(field) : field
        end

    	sim = new{typeof(m_named), typeof(d_named), typeof(statespace), typeof(sensorspace), typeof(actionspace)}(m, d, m_named, d_named,
            initstate, skip,
            statespace, sensorspace, actionspace)

        forward!(sim)
    end
end

function MJSim(modelpath::String; skip::Integer = DEFAULT_SKIP)
    m = jlModel(modelpath)
    d = jlData(m)
    MJSim(m, d, skip=skip)
end


function LyceumBase.tconstruct(::Type{MJSim}, N::Integer, m::jlModel; skip::Integer = DEFAULT_SKIP, can_share_model::Bool=DEFAULT_CAN_SHARE_MODEL)
    N > 0 || throw(ArgumentError("N must be > 0"))
    if can_share_model
        ntuple(_ -> MJSim(m, jlData(m), skip=skip), N)
    else
        ntuple(N) do _
            mcopy = copy(m)
            MJSim(mcopy, jlData(mcopy), skip=skip)
        end
    end
end

function LyceumBase.tconstruct(::Type{MJSim}, N::Integer, modelpath::AbstractString; kwargs...)
    tconstruct(MJSim, N, jlModel(modelpath); kwargs...)
end


@inline statespace(sim::MJSim) = sim.statespace

@propagate_inbounds function getstate!(state::RealVec, sim::MJSim)
    @boundscheck checkaxes(statespace(sim), state)
    shaped = statespace(sim)(state)
    @uviews shaped begin _copyshaped!(shaped, sim.d) end
    state
end

@propagate_inbounds getstate(sim::MJSim) = getstate!(allocate(statespace(sim)), sim)


# Two things to note about MuJoCo state:
# 1. MJSTATE_FIELDS are theoretically the minimal fields/sufficient statistics, as in
#    all other fields in mjData are a function of MJSTATE_FIELDS, but this is only true for
#    **dynamic** elements. For example, calling rand!(d.xpos) followed by mj_forward(m, d) will
#    overwrite the entries of d.xpos corresponding to dynamically changing elements, but NOT for
#    e.g. the root world body (d.xpos[:, 1]). If the user inadvertantly overwrites these fields,
#    mjData will be left in a corrupt state. Because of this, we call mj_resetData(m, d) followed
#    by mj_forward(m, d) inside the exported setstate!, while providing the ability to
#    skip this using copystate! for users who know what they're doing and want to squeeze some
#    extra performance out.
# 2. qacc_warmstart is part of state in that it will control the final solution (because MuJoCo's
#    optimizer may not find the exact minima). This means that:
#       setstate!(sim, state)
#       @assert getstate(sim) == state
#    will fail if the optimizer terminated early. To avoid this, we first copy over the state
#    variables (MJSTATE_FIELDS, including qacc_warmstart), call mj_forward, and then re-copy
#    over qacc_warmstart so that the above equality holds true. This means that mjData is left
#    in a slightly inconsitent state (qacc_warmstart will be slightly out of sync).

@propagate_inbounds function setstate!(sim::MJSim, state::RealVec)
    mj_resetData(sim.m, sim.d)
    copystate!(sim, state)
    forward!(sim)
    shaped = statespace(sim)(state)
    @uviews shaped begin copyto!(sim.d.qacc_warmstart, shaped.qacc_warmstart) end
    sim
end

@propagate_inbounds function copystate!(sim::MJSim, state::RealVec)
    @boundscheck checkaxes(statespace(sim), state)
    shaped = @inbounds statespace(sim)(state)
    @uviews shaped begin _copyshaped!(sim.d, shaped) end
    sim
end

@inline function _copyshaped!(dst, src)
    @inbounds dst.time = src.time
    @inbounds copyto!(dst.qpos, src.qpos)
    @inbounds copyto!(dst.qvel, src.qvel)
    @inbounds copyto!(dst.act, src.act)
    @inbounds copyto!(dst.mocap_pos, src.mocap_pos)
    @inbounds copyto!(dst.mocap_quat, src.mocap_quat)
    @inbounds copyto!(dst.userdata, src.userdata)
    @inbounds copyto!(dst.qacc_warmstart, src.qacc_warmstart)
    dst
end


@inline sensorspace(sim::MJSim) = sim.sensorspace

@propagate_inbounds function getsensor!(sensordata::RealVec, sim::MJSim)
    @boundscheck checkaxes(sensorspace(sim), sensordata)
    @inbounds copyto!(sensordata, sim.d.sensordata)
end

@propagate_inbounds getsensor(sim::MJSim) = getsensor!(allocate(sim.sensorspace), sim)


@inline actionspace(sim::MJSim) = sim.actionspace

@propagate_inbounds function getaction!(action::RealVec, sim::MJSim)
    @boundscheck checkaxes(actionspace(sim), action)
    @inbounds copyto!(action, sim.d.ctrl)
end

@propagate_inbounds getaction(sim::MJSim, action::RealVec) = getaction!(allocate(sim.actionspace), sim)

@propagate_inbounds function setaction!(sim::MJSim, action::RealVec)
    forwardskip!(setaction_nofwd!(sim, action), MJCore.mjSTAGE_VEL)
end

@propagate_inbounds function setaction_nofwd!(sim::MJSim, action::RealVec)
    @boundscheck checkaxes(actionspace(sim), action)
    @inbounds copyto!(sim.d.ctrl, action)
    sim
end


@inline reset!(sim::MJSim) = forward!(reset_nofwd!(sim))
@inline reset_nofwd!(sim::MJSim) = (mj_resetData(sim.m, sim.d); sim)

# typically 2-3x faster than reset!
@inline fastreset!(sim::MJSim) = forward!(fastreset_nofwd!(sim))
@inline function fastreset_nofwd!(sim::MJSim)
    zerofullctrl_nofwd!(sim)
    @inbounds copystate!(sim, sim.initstate)
end


"""
    step!(sim::MJSim[, skip::Integer])

Step the simulation by `skip` steps, where `skip` defaults to `MJSim.skip`.

State-dependent controls (e.g. the `ctrl`, `xfrc_applied`, `qfrc_applied`
fields of `sim.d`) should be set before calling `step!`.
"""
function step!(sim::MJSim, skip::Integer=sim.skip)
    check_skip(skip)
    # According to MuJoCo docs order must be:
    # 1. mj_step1
    # 2. set_{ctrl, xfrc_applied, qfrc_applied}
    # 3. mj_step2
    # This implies user should call setaction! then step!
    for _=1:skip
        mj_step2(sim.m, sim.d)
        mj_step1(sim.m, sim.d)
    end
    sim
end


"""
    zeroctrl!(sim::MJSim)

Zero out `sim.d.ctrl` and compute new forward dynamics.
"""
@inline zeroctrl!(sim::MJSim) = forwardskip!(zeroctrl_nofwd!(sim), MJCore.mjSTAGE_VEL)
@inline zeroctrl_nofwd!(sim::MJSim) = (fill!(sim.d.ctrl, zero(mjtNum)); sim)

"""
    zerofullctrl!(sim::MJSim)

Zero out all of the fields in `sim.d` that contribute to forward dynamics calculations,
namely `ctrl`, `qfrc_applied`, and `xfrc_applied`, and compute the forward dynamics.
"""
@inline zerofullctrl!(sim::MJSim) = forwardskip!(zerofullctrl_nofwd!(sim), MJCore.mjSTAGE_VEL)
@inline function zerofullctrl_nofwd!(sim::MJSim)
    fill!(sim.d.ctrl, zero(mjtNum))
    fill!(sim.d.qfrc_applied, zero(mjtNum))
    fill!(sim.d.xfrc_applied, zero(mjtNum))
    sim
end

"""
    masscenter(sim::MJSim)

Calculate the center of mass of the entire simulation by computing the mass-weighted
sum of the Cartesian coordinates of each body in the world frame.
"""
function masscenter(sim::MJSim)
    mcntr = zeros(SVector{3, Float64})
    mtotal = 0.0
    body_mass, xipos = sim.m.body_mass, sim.d.xipos
    @uviews xipos for i=1:sim.m.nbody
        bmass = body_mass[i]
        bodycom = SVector{3, Float64}(uview(xipos, :, i))
        mcntr = mcntr + bmass * bodycom
        mtotal += bmass
    end
    mcntr = SVector(mcntr)
    mcntr / mtotal
end

"""
    forward!(sim::MJSim)

Compute the forward dynamics of the simulation and store them in `sim.d`.
Equivalent to `mj_forward(sim.m, sim.d)`
"""
@inline forward!(sim::MJSim) = (mj_forward(sim.m, sim.d); sim)

"""
    forwardskip!(sim::MJSim, skipstage::MJCore.mjtStage=MJCore.mjSTAGE_NONE, skipsensor::Bool=false)

Compute the forward dynamics of the simulation and store them in `sim.d`, optionally
skipping parts of the dynamics calculation. Equivalent to
`mj_forwardSkip(sim.m, sim.d, skipstage, skipsensor)`.

More information can be found at http://mujoco.org/book/programming.html under 'Forward Dynamics'
"""
@inline function forwardskip!(sim::MJSim, skipstage::MJCore.mjtStage=MJCore.mjSTAGE_NONE, skipsensor::Bool=false)
    mj_forwardSkip(sim.m, sim.d, skipstage, skipsensor)
    sim
end

"""
    timestep(sim::MJSim)

Return the effective timestep of `sim`. Equivalent to `sim.skip * sim.m.opt.timestep`.
"""
@inline timestep(sim::MJSim) = sim.m.opt.timestep * sim.skip

"""
    Base.time(sim::MJSim)

Return the current simulation time, in seconds, of `sim`.
Equivalent to sim.d.time.
"""
@inline Base.time(sim::MJSim) = sim.d.time

@inline getsim(sim::MJSim) = sim

# TODO(cxs):: Make more informative. This is a temp fix for those gawdawful error messages from sim.{mn, dn}.
Base.show(io::IO, ::MIME"text/plain", sim::Union{MJSim, Type{<:MJSim}}) = show(io, sim)
Base.show(io::IO, sim::Union{MJSim, Type{<:MJSim}}) = print(io, "MJSim")

@inline check_skip(skip) = skip > 0 || throw(ArgumentError("`skip` must be > 0"))