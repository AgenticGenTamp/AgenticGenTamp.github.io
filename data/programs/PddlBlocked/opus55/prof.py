import cProfile, pstats, sys, numpy as np
from env_client import make_env
from envutil import Sim
import planner
env=make_env(); S=Sim(env,env.reset(seed=int(sys.argv[1]))[0])
blk=S.block('blocker'); g0=S.block('green0')
cProfile.run('planner.make_plan(S.base(),S.q(),blk,g0,g0[2])','prof.out')
p=pstats.Stats('prof.out'); p.sort_stats('cumulative').print_stats(18)
