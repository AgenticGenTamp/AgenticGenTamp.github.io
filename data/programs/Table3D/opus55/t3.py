import numpy as np, time
from env_client import make_env
from fastplan import *
env=make_env()
J=[f'joint_{i}' for i in range(1,8)]
for seed in range(6):
    obs,_=env.reset(seed=seed); r=obs.get_object_from_name('robot')
    x0=np.array([obs.get(r,n) for n in ['pos_base_x','pos_base_y','pos_base_rot']+J])
    for name in sorted(obs.get_object_names()):
        if not name.startswith('cube'): continue
        c=obs.get_object_from_name(name)
        p=np.array([obs.get(c,'pose_x'),obs.get(c,'pose_y'),obs.get(c,'pose_z')])
        out=[]
        t=time.time()
        for k in range(4):
            v,s,ce,ci=solve(x0,p,k*np.pi/2)
            out.append((round(s,3),round(ce,5),round(ci,4)))
        print(seed,name,p.round(2),out,round(time.time()-t,2))
env.close()
