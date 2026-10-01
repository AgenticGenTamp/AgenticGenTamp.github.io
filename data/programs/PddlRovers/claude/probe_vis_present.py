import numpy as np
from probe_vis_lib import *
for SEED,PIL in [(20,'obstacle7'),(22,'obstacle10'),(23,'obstacle5'),(39,'obstacle9')]:
    env=make_env(); obs,info=env.reset(seed=SEED, options={'object_count':1})
    L=layout(obs); q=np.array([L[PIL]['x'],L[PIL]['y']])
    free,xs,res=build_grid(obs)
    i=0 if q[0]>0 else 1
    # approach from south of pillar
    start=q+np.array([0.0,-0.45])
    obs,ok=nav(env,obs,start[0],start[1],i=i,tol=0.02,free=free,xs=xs,res=res)
    p0=pose(obs,i)[:2]
    if np.linalg.norm(p0-start)>0.06:
        print("seed%d %s: could not stage (at %s target %s)"%(SEED,PIL,np.round(p0,2),np.round(start,2))); env.close(); continue
    # push north with fine steps
    s=0.2
    while s>=0.001:
        pp=pose(obs,i)[:2].copy()
        obs,_,_,_,_=st(env,dy=s,i=i)
        if np.linalg.norm(pose(obs,i)[:2]-pp)<1e-7: s/=2
        if pose(obs,i)[1]>q[1]+0.3: break
    yf=pose(obs,i)[1]
    print("seed%-3d %-11s pillar_y=%.3f  rover stopped/ended at y=%.3f gap=%.3f  %s"%(SEED,PIL,q[1],yf,q[1]-yf,"PASSED THROUGH" if yf>q[1] else "BLOCKED (solid)"))
    env.close()
