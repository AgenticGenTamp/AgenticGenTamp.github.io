import sys, numpy as np, time
from env_client import make_env
from envutil import Sim
import planner; planner.DEBUG=True
env=make_env(); S=Sim(env,env.reset(seed=int(sys.argv[1]))[0])
blk=S.block('blocker'); g0=S.block('green0'); print('blk',blk[:2].round(3),'g0',g0[:2].round(3))
t=time.time(); r=planner.make_plan(S.base(),S.q(),blk,g0,g0[2]); print('time',time.time()-t, r is not None)
