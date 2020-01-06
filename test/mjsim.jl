#let sim = MJSim(TESTMODELXML), sp = statespace(sim), init = sp(sim.initstate), state = sp(getstate(sim))
#    for name in LyceumMuJoCo.MJSTATE_FIELDS
#        @test getproperty(init, name) == getproperty(state, name)
#    end
#end
#
#
#let sim = MJSim(TESTMODELXML), sp = statespace(sim)
#    s1 = sp(rand(sp))
#    setstate!(sim, s1)
#    s2 = sp(getstate(sim))
#    for name in LyceumMuJoCo.MJSTATE_FIELDS
#        @test getproperty(s1, name) == getproperty(s2, name)
#    end
#end