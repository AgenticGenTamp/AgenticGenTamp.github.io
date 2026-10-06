import sys, time, numpy as np
from test_approach import make
from approach import *
seed=int(sys.argv[1]); count=int(sys.argv[2])
env=make(); s,info=env.reset(seed=seed, options={'object_count':count})
ap=GeneratedApproach(env.action_space, None, {}); ap.reset(s,info)
orig=ap._make_plan
hits=[0]
def wrapped(state):
    r,rects=ap._parse(state); m=ap._robot_model(r); held=ap._held(r,rects,m)
    ok=orig(state)
    if held and held!=['target_block'] and len(ap.plan)<=1:
        q0=ap._q(r)
        obs=ap._obstacles(rects,set(held))
        hs=[]
        for n in held:
            d=rects[n]
            # replicate held spec
        print("STUCK release", held, "q0",np.round(q0,3))
        # replicate checker as in _make_plan
        import inspect
        hits[0]+=1
    return ok
ap._make_plan=wrapped
src=open('approach.py').read()
for t in range(2000):
    a=ap.get_action(s); s,_,term,_,_=env.step(a)
    if term or hits[0]>=1: break
