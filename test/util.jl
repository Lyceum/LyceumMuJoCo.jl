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
        st = view(traj.states, :, t)
        at = view(traj.acts, :, t)
        ot = view(traj.obses, :, t)

        getstate!(st, e)
        getobs!(ot, e)
        setaction!(e, at)
        traj.rews[t] = getreward(st, at, ot, e)
        traj.evals[t] = geteval(st, at, ot, e)
        step!(e)
    end
    traj
end



function test_group(group)
    @testset "Testing $etype\n    Args: $args.\n    Kwargs: $kwargs" for (etype, args, kwargs) in group
        test_env(etype, args...; kwargs...)
    end
end

function test_env(etype::Type{<:AbstractMuJoCoEnv}, args...; kwargs...)
    makeenv() = etype(args...; kwargs...)
    @testset "Basic Interface" begin
        e = makeenv()

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

        @testset "$(string(nameof(spacefn)))" for (spacefn, fns) in pairs(SPACE_TO_FUNCS)
            space, x = spacefn(e), fns.get(e)
            @test eltype(x) == eltype(space)
            @test axes(x) == axes(space)
        end

        @test ndims(ssp) == 1
        @test ndims(osp) == 1
        @test ndims(asp) == 1

        @test isdone(e) isa Bool
        @test time(e) isa Float64
        t1 = time(e)
        t2 = time(e)
        @test t1 == t2

        let e = makeenv(), x1 = rand!(zeros(ssp)), x2 = copy(x1)
            @test x1 === @inferred(getstate!(x1, e))
            @test x1 != x2
            getstate!(x2, e)
            @test x1 == x2
        end

        let e = makeenv(), x1 = rand!(zeros(osp)), x2 = copy(x1)
            @test x1 === @inferred(getobs!(x1, e))
            @test x1 != x2
            getobs!(x2, e)
            @test x1 == x2
        end

        let e = makeenv(), x1 = rand!(zeros(asp)), x2 = copy(x1)
            @test x1 === @inferred(getaction!(x1, e))
            @test x1 != x2
            getaction!(x2, e)
            @test x1 == x2
            rand!(x1)
            setaction!(e, x1)
            getaction!(x2, e)
            @test x1 == x2
        end

        let e1=makeenv(), e2 = makeenv()
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
            setaction!(e1, a)
            setaction!(e2, a)
            step!(e1)
            step!(e2)
            getstate!(s1, e1)
            getstate!(s2, e2)
            @test s1 == s2
        end

        let e1=makeenv(), e2 = makeenv()
            randreset!(e1)
            randreset!(e2)
            @test getstate(e1) != getstate(e2)
            @test getobs(e1) != getobs(e2)
        end
    end

    @testset "Determinism" begin
        e = makeenv()
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
