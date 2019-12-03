####
#### Core interface
####

struct MJSimParameters
    modelpath::String
    args::Tuple
    kwargs::NamedTuple

    function MJSimParameters(modelpath::AbstractString, args...; kwargs...)
        new(modelpath, args, values(kwargs))
    end
end

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
"""
struct MJSim{MN, DN, S, SE, A}
    m::jlModel
    d::jlData

    "named-access version of MJSim.m"
    mn::MN
    "named-access version of MJSim.d"
    dn::DN

    # spaces
    statespace::S
    sensorspace::SE
    actionspace::A

    initstate::Vector{mjtNum}
    skip::Int
end

function MJSim(m::jlModel, d::jlData; skip::Integer=1)
    skip > 0 || error("`skip` must be > 0")

    m_named, d_named = namify(m, d)

    statespace = MultiShape(
        time = ScalarShape(mjtNum),
        qpos = VectorShape(mjtNum, m.nq),
        qvel = VectorShape(mjtNum, m.nv),
        act = VectorShape(mjtNum, m.na),
        qacc_warmstart = VectorShape(mjtNum, m.nv),
        mocap_pos = MatrixShape(mjtNum, m.nmocap, 3),
		mocap_quat = MatrixShape(mjtNum, m.nmocap, 4),
        userdata = VectorShape(mjtNum, m.nuserdata)
    )

    if m.nsensor > 0
        names, shapes = [], []
        for id = 1:m.nsensor
          name = jl_id2name(m, MJCore.mjOBJ_SENSOR, id)
          name = isnothing(name) ? Symbol("unnamed_$id") : Symbol(name)
          dof = m.sensor_dim[id]
          shape = dof == 1 ? ScalarShape(mjtNum) : VectorShape(mjtNum, dof)
          push!(names, name)
          push!(shapes, shape)
        end
        sensorspace = MultiShape{NamedTuple{Tuple(names)}(Tuple(shapes))}()
    else
        sensorspace = VectorShape(mjtNum, 0)
    end

    if m.nu > 0
        names, shapes = [], []
        clampctrl = !jl_disabled(m, MJCore.mjDSBL_CLAMPCTRL)
        for id = 1:m.nu
            name = jl_id2name(m, MJCore.mjOBJ_ACTUATOR, id)
            name = isnothing(name) ? Symbol("unnamed_$id") : Symbol(name)
            shape = ScalarShape(mjtNum) #TODO boundedshape
            push!(names, name)
            push!(shapes, shape)
        end
        actionspace = MultiShape(NamedTuple{Tuple(names)}(Tuple(shapes)))
    else
        actionspace = VectorShape(mjtNum, 0)
    end

    initstate = reduce(vcat,
        (d.time, d.qpos, d.qvel, d.act, d.qacc_warmstart, vec(d.mocap_pos), vec(d.mocap_quat), d.userdata))

	sim = MJSim(m, d, m_named, d_named,
		statespace, sensorspace, actionspace,
		initstate, skip)
    forward!(sim)
end

function MJSim(modelpath::String, args...; kwargs...)
    m = jlModel(modelpath)
    d = jlData(m)
    MJSim(m, d, args...; kwargs...)
end

MJSim(params::MJSimParameters) = MJSim(params.modelpath, params.args...; params.kwargs...)

function sharedmemory_mjsims(modelpath::String, n::Integer, args...; kwargs...)
    m = jlModel(modelpath)
    Tuple(MJSim(m, jlData(m), args...; kwargs...) for i = 1:n)
end

function sharedmemory_mjsims(params::MJSimParameters, n::Integer)
    sharedmemory_mjsims(params.modelpath, n, params.args...; params.kwargs...)
end



@inline getsim(sim::MJSim) = sim

# TODO(cxs):: Make more informative. This is a temp fix for those gawdawful error messages from sim.{mn, dn}.
Base.show(io::IO, ::MIME"text/plain", sim::Union{MJSim, Type{<:MJSim}}) = show(io, sim)
Base.show(io::IO, sim::Union{MJSim, Type{<:MJSim}}) = print(io, "MJSim")



@inline statespace(sim::MJSim) = sim.statespace

@propagate_inbounds function getstate!(state::RealVec, sim::MJSim)
    ms = statespace(sim)
    @boundscheck if axes(state) != axes(ms)
        throw(ArgumentError("axes(state) must equal axes(statespace(sim))"))
    end

    @uviews state begin
        shaped = ms(state)
        @inbounds shaped.time = sim.d.time
        @inbounds _maybecopyto!(Length(ms.qpos), shaped.qpos, sim.d.qpos)
        @inbounds _maybecopyto!(Length(ms.qvel), shaped.qvel, sim.d.qvel)
        @inbounds _maybecopyto!(Length(ms.act), shaped.act, sim.d.act)
        @inbounds _maybecopyto!(Length(ms.qacc_warmstart), shaped.qacc_warmstart, sim.d.qacc_warmstart)
        @inbounds _maybecopyto!(Length(ms.mocap_pos), shaped.mocap_pos, sim.d.mocap_pos)
        @inbounds _maybecopyto!(Length(ms.mocap_quat), shaped.mocap_quat, sim.d.mocap_quat)
        @inbounds _maybecopyto!(Length(ms.userdata), shaped.userdata, sim.d.userdata)
    end
    state
end

getstate(sim::MJSim) = (s = allocate(statespace(sim)); getstate!(s, sim); s)

@propagate_inbounds function setstate!(sim::MJSim, state::RealVec)
    ms = statespace(sim)
    @boundscheck if axes(state) != axes(ms)
        throw(ArgumentError("axes(state) must equal axes(statespace(sim))"))
    end

    @uviews state begin
        shaped = ms(state)
        @inbounds sim.d.time = shaped.time
        @inbounds _maybecopyto!(Length(ms.qpos), sim.d.qpos, shaped.qpos)
        @inbounds _maybecopyto!(Length(ms.qvel), sim.d.qvel, shaped.qvel)
        @inbounds _maybecopyto!(Length(ms.act), sim.d.act, shaped.act)
        @inbounds _maybecopyto!(Length(ms.qacc_warmstart), sim.d.qacc_warmstart, shaped.qacc_warmstart)
        @inbounds _maybecopyto!(Length(ms.mocap_pos), sim.d.mocap_pos, shaped.mocap_pos)
        @inbounds _maybecopyto!(Length(ms.mocap_quat), sim.d.mocap_quat, shaped.mocap_quat)
        @inbounds _maybecopyto!(Length(ms.userdata), sim.d.userdata, shaped.userdata)
    end
    sim
end



@inline sensorspace(sim::MJSim) = sim.sensorspace

@inline function getsensor!(sensordata::RealVec, sim::MJSim)
    @boundscheck if axes(sensordata) != axes(sensorspace(sim))
        throw(ArgumentError("axes(sensordata) must equal axes(sensorspace(sim))"))
    end
    @inbounds copyto!(sensordata, sim.d.sensordata)
end

@inline getsensor(sim::MJSim) = getsensor!(allocate(sim.sensorspace), sim)



@inline actionspace(sim::MJSim) = sim.actionspace

@inline function getaction!(action::RealVec, sim::MJSim)
    @boundscheck if axes(action) != axes(actionspace(sim))
        throw(ArgumentError("axes(action) must equal axes(actionspace(sim))"))
    end
    @inbounds copyto!(action, sim.d.ctrl)
end

getaction(sim::MJSim, action::RealVec) = getaction!(allocate(sim.actionspace), sim)

@inline function setaction!(sim::MJSim, action::RealVec)
    @boundscheck if axes(action) != axes(actionspace(sim))
        throw(ArgumentError("axes(action) must equal axes(actionspace(sim))"))
    end
    @inbounds copyto!(sim.d.ctrl, action)
    sim
end



function fullreset!(sim::MJSim)
    mj_resetData(sim.m, sim.d)
    forward!(sim)
    sim
end

reset!(sim::MJSim) = reset!(sim, sim.initstate)

function reset!(sim::MJSim, state::RealVec)
    zerofullctrl!(sim)
    setstate!(sim, state)
    forward!(sim)
    sim
end

function reset!(sim::MJSim, state::RealVec, action::RealVec)
    zerofullctrl!(sim)
    setstate!(sim, state)
    setaction!(sim, action)
    forward!(sim)
    sim
end



"""
    step!(sim::MJSim[, skip::Integer])

Step the simulation by `skip` steps, where `skip` defaults to `MJSim.skip`.

State-dependent controls (e.g. MJSim.d.{ctrl, xfrc_applied, qfrc_applied})
should be set before calling `step!`.
"""
function step!(sim::MJSim, skip::Integer=sim.skip)
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

timestep(sim::MJSim) = sim.m.opt.timestep
effective_timestep(sim::MJSim) = timestep(sim) * sim.skip
Base.time(sim::MJSim) = sim.d.time

_maybecopyto!(::Shapes.Length{0}, src, dest) = src
_maybecopyto!(::Shapes.Length, src, dest) = copyto!(src, dest)