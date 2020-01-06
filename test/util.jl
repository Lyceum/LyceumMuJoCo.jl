# TODO put in LyceumBase
#macro noalloc(expr)
#    tmp = gensym()
#    ex = quote
#        local $tmp = $BenchmarkTools.@benchmark $(expr) evals=1 samples=1
#        iszero($(tmp).allocs)
#    end
#    :($(esc(ex)))
#end




function test_group(group)
    @testset "Testing $etype\n    Args: $args.\n    Kwargs: $kwargs" for (etype, args, kwargs) in group
        test_env(etype, args...; kwargs...)
    end
end
