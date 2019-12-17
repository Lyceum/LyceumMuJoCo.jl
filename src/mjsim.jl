####
#### Core interface
####

const MJSTATE_FIELDS = (:time, :qpos, :qvel, :act, :mocap_pos, :mocap_quat, :userdata, :qacc_warmstart)
const MJACTION_FIELDS = (:ctrl, :qfrc_applied, :xfrc_applied)

const DEFAULT_SKIP = 1
const DEFAULT_CAN_SHARE_MODEL = true

"""
    MJSim

A type that couples a jlModel and jlData from MuJoCo.jl to provide a full simulation

The following are the official/internal/minimum set of fields from `jlData` for
state, observation, and action in MuJoCo:

- State: `(time, qpos, qvel, act, mocap_pos, mocap_quat, userdata, qacc_warmstart)`
- Observation: `sensordata`
- Action: `(ctrl, qfrc_applied, xfrc_applied)`

MJSim follows this convention.
For more information, see the "State and control" section of http://www.mujoco.org/book/programming.html

# Fields
    `m::jlModel`: ccontains the model description and is expected to remain constant.
    `d::jlData`: contains all dynamic variables and intermediate results.
    `mn::Tuple`: named-access version of MJSim.m provided by AxisArrays.jl.
    `dn::Tuple`: named-access version of MJSim.d provided by AxisArrays.jl.
    `initstate::Vector{mjtNum}`: The initial state vector at the time when this MJSim was constructed.
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
end

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

	sim = MJSim(m, d, m_named, d_named,
        initstate, skip,
		statespace, sensorspace, actionspace)

    forward!(sim)
end

function MJSim(modelpath::String; skip::Integer = DEFAULT_SKIP)
    m = jlModel(modelpath)
    d = jlData(m)
    MJSim(m, d, skip=skip)
end

function LyceumBase.thread_construct(::Type{MJSim}, N::Integer, m::jlModel; skip::Integer = DEFAULT_SKIP, can_share_model::Bool=DEFAULT_CAN_SHARE_MODEL)
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

function LyceumBase.thread_construct(::Type{MJSim}, N::Integer, modelpath::AbstractString; kwargs...)
    thread_construct(MJSim, N, jlModel(modelpath); kwargs...)
end



@inline getsim(sim::MJSim) = sim # Used in LyceumViz.jl



@inline statespace(sim::MJSim) = sim.statespace

@propagate_inbounds function getstate!(state::RealVec, sim::MJSim)
    ms = statespace(sim)
    @boundscheck if axes(state) != axes(ms)
        throw(ArgumentError("axes(state) must equal axes(statespace(sim))"))
    end
    shaped = ms(state)
    @uviews shaped @inbounds begin
        shaped.time = sim.d.time
        copyto!(shaped.qpos, sim.d.qpos)
        copyto!(shaped.qvel, sim.d.qvel)
        copyto!(shaped.act, sim.d.act)
        copyto!(shaped.mocap_pos, sim.d.mocap_pos)
        copyto!(shaped.mocap_quat, sim.d.mocap_quat)
        copyto!(shaped.userdata, sim.d.userdata)
        copyto!(shaped.qacc_warmstart, sim.d.qacc_warmstart)
    end
    state
end

@inline getstate(sim::MJSim) = getstate!(allocate(statespace(sim)), sim)

@propagate_inbounds function setstate!(sim::MJSim, state::RealVec)
    ms = statespace(sim)
    @boundscheck if axes(state) != axes(ms)
        throw(ArgumentError("axes(state) must equal axes(statespace(sim))"))
    end

    shaped = ms(state)
    @uviews shaped @inbounds begin
        sim.d.time = shaped.time
        copyto!(sim.d.qpos, shaped.qpos)
        copyto!(sim.d.qvel, shaped.qvel)
        copyto!(sim.d.act, shaped.act)
        copyto!(sim.d.mocap_pos, shaped.mocap_pos)
        copyto!(sim.d.mocap_quat, shaped.mocap_quat)
        copyto!(sim.d.userdata, shaped.userdata)
        copyto!(sim.d.qacc_warmstart, shaped.qacc_warmstart)
    end
    sim
end



@inline sensorspace(sim::MJSim) = sim.sensorspace

@propagate_inbounds function getsensor!(sensordata::RealVec, sim::MJSim)
    @boundscheck if axes(sensordata) != axes(sensorspace(sim))
        throw(ArgumentError("axes(sensordata) must equal axes(sensorspace(sim))"))
    end
    @inbounds copyto!(sensordata, sim.d.sensordata)
end

@inline getsensor(sim::MJSim) = getsensor!(allocate(sim.sensorspace), sim)



@inline actionspace(sim::MJSim) = sim.actionspace

@propagate_inbounds function getaction!(action::RealVec, sim::MJSim)
    @boundscheck if axes(action) != axes(actionspace(sim))
        throw(ArgumentError("axes(action) must equal axes(actionspace(sim))"))
    end
    @inbounds copyto!(action, sim.d.ctrl)
end

getaction(sim::MJSim, action::RealVec) = getaction!(allocate(sim.actionspace), sim)

@propagate_inbounds function setaction!(sim::MJSim, action::RealVec)
    @boundscheck if axes(action) != axes(actionspace(sim))
        throw(ArgumentError("axes(action) must equal axes(actionspace(sim))"))
    end
    @inbounds copyto!(sim.d.ctrl, action)
    sim
end



fullreset!(sim::MJSim) = (mj_resetData(sim.m, sim.d); forward!(sim))

reset!(sim::MJSim) = reset!(sim, sim.initstate)
reset!(sim::MJSim, state::RealVec) = (zerofullctrl!(sim); setstate!(sim, state); forward!(sim))
function reset!(sim::MJSim, state::RealVec, action::RealVec)
    zerofullctrl!(sim)
    setstate!(sim, state)
    setaction!(sim, action)
    forward!(sim)
end



"""
    step!(sim::MJSim[, skip::Integer])

Step the simulation by `skip` steps, where `skip` defaults to `MJSim.skip`.

State-dependent controls (e.g. MJSim.d.{ctrl, xfrc_applied, qfrc_applied})
should be set before calling `step!`.
"""
function step!(sim::MJSim)
    # According to MuJoCo docs order must be:
    # 1. mj_step1
    # 2. set_{ctrl, xfrc_applied, qfrc_applied}
    # 3. mj_step2
    # This implies user should call setaction! then step!
    for _=1:sim.skip
        mj_step2(sim.m, sim.d)
        mj_step1(sim.m, sim.d)
    end
    sim
end

step!(sim::MJSim, a) = (setaction!(sim, a); step!(sim))



@inline zeroctrl!(sim::MJSim) = (fill!(sim.d.ctrl, zero(mjtNum)); sim)

function zerofullctrl!(sim::MJSim)
    fill!(sim.d.ctrl, zero(mjtNum))
    fill!(sim.d.qfrc_applied, zero(mjtNum))
    fill!(sim.d.xfrc_applied, zero(mjtNum))
    sim
end

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


@inline forward!(sim::MJSim) = (mj_forward(sim.m, sim.d); sim)

@inline function forwardskip!(sim::MJSim, skipstage::MJCore.mjtStage=MJCore.mjSTAGE_NONE, skipsensor::Bool=false)
    mj_forwardSkip(sim.m, sim.d, skipstage, skipsensor)
    sim
end


@inline timestep(sim::MJSim) = sim.m.opt.timestep

@inline effective_timestep(sim::MJSim) = timestep(sim) * sim.skip

@inline Base.time(sim::MJSim) = sim.d.time


# TODO(cxs):: Make more informative. This is a temp fix for those gawdawful error messages from sim.{mn, dn}.
Base.show(io::IO, ::MIME"text/plain", sim::Union{MJSim, Type{<:MJSim}}) = show(io, sim)
Base.show(io::IO, sim::Union{MJSim, Type{<:MJSim}}) = print(io, "MJSim")

@inline check_skip(skip) = skip > 0 || throw(ArgumentError("`skip` must be > 0"))