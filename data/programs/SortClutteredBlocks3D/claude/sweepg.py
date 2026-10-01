import numpy as np, sys
from env_client import make_env
import approach as A
def trial(zg, gopen, gclose, seed=0, steps=220):
    env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':4})
    ap=A.GeneratedApproach(None,None,{}); ap.z_grasp=zg; ap.gopen=gopen; ap.gclose=gclose
    ap.reset(obs,info)
    best=0
    for t in range(steps):
        a=ap.get_action(obs)
        obs,r,te,tr,i=env.step(a)
        best=max(best, float(A.obj_pos(obs,'cube1')[2]))
        if ap.idx>0: break
    env.close(); return round(best,4)
for zg in [0.405,0.412,0.42]:
    for go,gc in [(0.0,1.0),(1.0,0.0)]:
        print('zg',zg,'open/close',go,gc,'cube1 max z',trial(zg,go,gc))
