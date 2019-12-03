using LyceumMuJoCo, LyceumBase
using Test, Random, Shapes, Pkg, LinearAlgebra, BenchmarkTools

const ROLLOUT_HORIZON = 50

const LYCEUMENVS = ("Lyceum",
[
    (LyceumMuJoCo.PointMass, (), ())
]
)

const GYMENVS = ("Gym",
[
    (LyceumMuJoCo.HumanoidV2, (), ()),
    (LyceumMuJoCo.SwimmerV2, (), ()),
    (LyceumMuJoCo.HopperV2, (), ()),
]
)

# TODO put in LyceumBase
macro noalloc(expr)
    tmp = gensym()
    ex = quote
        local $tmp = $BenchmarkTools.@benchmark $(expr) evals=1 samples=1
        iszero($(tmp).allocs)
    end
    :($(esc(ex)))
end

function trajectory(e)
    T = ROLLOUT_HORIZON
    (
        states = Array(undef, statespace(e), T),
        obses = Array(undef, observationspace(e), T),
        acts = Array(undef, actionspace(e), T),
        rews = Array(undef, rewardspace(e), T),
        evals = Array(undef, evaluationspace(e), T)
    )
end

function rollout!(e, traj)
    for t=1:ROLLOUT_HORIZON
        setaction!(e, view(traj.acts, :, t))
        getstate!(view(traj.states, :, t), e)
        getobs!(view(traj.obses, :, t), e)
        traj.rews[t] = getreward(e)
        traj.evals[t] = geteval(e)
        step!(e)
    end
    traj
end

@testset "LyceumMuJoCo.jl" begin

    #proj_uuid = Base.UUID("48b9757e-04b8-4dbf-b6ed-75c13d9e4026")
    #ctx = Pkg.Operations.Context()
    #testpkgs = filter(ctx.env.project.deps) do pair
    #    name, uuid = pair
    #    !haskey(ctx.stdlibs, uuid) && uuid != proj_uuid
    #end
    #@testset "No local path" begin
    #    @testset "$name" for (name, uuid) in testpkgs
    #        entry = Pkg.Types.manifest_info(ctx.env, uuid)
    #        @test isnothing(entry.path)
    #    end
    #end

    # Tuples of (EnvTypeSymbol, args, kwargs). If EnvConstruct takes args then pass cl
    @testset "Testing $name" for (name, group) in [LYCEUMENVS, GYMENVS]
        @testset "Testing $etype\n    Args: $eargs.\n    Kwargs: $ekwargs" for (etype, eargs, ekwargs) in group
            @testset "Basic Interface" begin
                env() = first(sharedmemory_envs(etype, 1, eargs...; ekwargs...))
                e = env()

                @test e isa AbstractMuJoCoEnv
                @test isconcretetype(typeof(e))

                @test @inferred(statespace(e)) isa AbstractShape
                @test @inferred(observationspace(e)) isa AbstractShape
                @test @inferred(actionspace(e)) isa AbstractShape
                @test @inferred(rewardspace(e)) isa ScalarShape
                @test @inferred(evaluationspace(e)) isa ScalarShape

                ssp = statespace(e)
                osp = observationspace(e)
                asp = actionspace(e)
                rsp = rewardspace(e)
                esp = evaluationspace(e)

                @test ndims(ssp) == 1
                @test ndims(osp) == 1
                @test ndims(asp) == 1

                @test isdone(e) isa Bool
                @test time(e) isa Float64
                t1 = time(e)
                t2 = time(e)
                @test t1 == t2

                let e = env(), x1 = rand!(zeros(ssp)), x2 = copy(x1)
                    @test x1 === @inferred(getstate!(x1, e))
                    @test x1 != x2
                    getstate!(x2, e)
                    @test x1 == x2
                end

                let e = env(), x1 = rand!(zeros(osp)), x2 = copy(x1)
                    @test x1 === @inferred(getobs!(x1, e))
                    @test x1 != x2
                    getobs!(x2, e)
                    @test x1 == x2
                end

                let e = env(), x1 = rand!(zeros(asp)), x2 = copy(x1)
                    @test x1 === @inferred(getaction!(x1, e))
                    @test x1 != x2
                    getaction!(x2, e)
                    @test x1 == x2
                    rand!(x1)
                    setaction!(e, x1)
                    getaction!(x2, e)
                    @test x1 == x2
                end

                let e1=env(), e2 = env()
                    s1 = zeros(statespace(e1))
                    s2 = zeros(statespace(e1))
                    reset!(e1)
                    reset!(e2)
                    getstate!(s1, e1)
                    getstate!(s2, e2)
                    @test s1 == s2
                    step!(e1)
                    getstate!(s1, e1)
                    @test s1 != s2

                    a = zeros(actionspace(e1))
                    s1 .= 0
                    s2 .= 0
                    rand!(a)
                    reset!(e1)
                    reset!(e2)
                    step!(e1, a)
                    step!(e2, a)
                    getstate!(s1, e1)
                    getstate!(s2, e2)
                    @test s1 == s2
                end

                let e1=env(), e2 = env()
                    randreset!(e1)
                    randreset!(e2)
                    @test getstate(e1) != getstate(e2)
                    @test getobs(e1) != getobs(e2)
                end
            end

            @testset "Determinism" begin
                e = first(sharedmemory_envs(etype, 1, eargs...; ekwargs...))
                reset!(e)

                # execute a random control trajectory and check for repeatability

                t1 = trajectory(e)
                rand!(t1.acts)
                rollout!(e, t1)

                function maketraj()
                    traj = trajectory(e)
                    rand!(traj.states)
                    rand!(traj.obses)
                    traj.acts .= t1.acts
                    rand!(traj.rews)
                    traj
                end

                t2 = maketraj()
                reset!(e)
                rollout!(e, t2)
                @test t1 == t2

                t2 = maketraj()
                reset!(e)
                rollout!(e, t2)
                @test t1 == t2
            end

            #@testset "Allocation" begin
            #    #@Array sharedmemory_envs(etype, 1) #NOTE broken
            #    e = first(sharedmemory_envs(etype, 1))

            #    @test_broken @noalloc geteval($e)
            #    @test_broken @noalloc getreward($e)

            #    @test @noalloc statespace($e)
            #    @test @noalloc observationspace($e)
            #    @test @noalloc actionspace($e)
            #    @test @noalloc rewardspace($e)
            #    @test @noalloc evaluationspace($e)

            #    s = zeros(statespace(e))
            #    @test @noalloc getstate!($s, $e)
            #    @test @noalloc reset!($e, $s)
            #    @test @noalloc reset!($e)

            #    o = zeros(observationspace(e))
            #    @test @noalloc getobs!($o, $e)

            #    a = zeros(actionspace(e))
            #    @test @noalloc getaction!($a, $e)
            #    @test @noalloc setaction!($e, $a)

            #    @test @noalloc step!($e)

            #    @test @noalloc isdone($e)
            #end
        end
    end
end
