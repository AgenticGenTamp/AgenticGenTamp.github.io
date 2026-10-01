import numpy as np, sys
from env_client import make_env
import approach as A
def trial(dx,dy,zg,seed=0,steps=260):
    env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':4})
    ap=A.GeneratedApproach(None,None,{}); ap.z_grasp=zg; ap.off=np.array([dx,dy])
    ap.reset(obs,info)
    c0=A.obj_pos(obs,'cube1').copy(); best=0.0; moved=0.0
    for t in range(steps):
        a=ap.get_action(obs); obs,r,te,tr,i=env.step(a)
        c=A.obj_pos(obs,'cube1'); best=max(best,float(c[2])); moved=max(moved,float(np.linalg.norm(c[:2]-c0[:2])))
        if ap.idx>0: break
    env.close(); return round(best,4), round(moved,3)
dx,dy,zg=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3])
print(f"dx={dx} dy={dy} zg={zg}", trial(dx,dy,zg), flush=True)
