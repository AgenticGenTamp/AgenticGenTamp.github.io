import numpy as np
from plib3 import *
from env_client import make_env
env=make_env()
cands={'c0':[0,0,0],'pp':[1/3.,1/3.,0],'mp':[-1/3.,1/3.,0],'mm':[-1/3.,-1/3.,0],'pm':[1/3.,-1/3.,0]}
for seed in range(0,8):
    obs,info=env.reset(seed=seed); base=rb(obs)
    names=sorted(n for n in obs.get_object_names() if n.startswith('part'))
    print("seed",seed,"nparts",len(names),flush=True)
    for n in names:
        o,f=part_info(obs,n)
        istri='side_a' in f
        y=yaw_of(obs,n)
        desc="%s %s yaw%.1f"%(n,'TRI type%d'%f.get('triangle_type',-9) if istri else 'CUB',np.degrees(y))
        if not istri:
            obs,info=env.reset(seed=seed)
            obs,g,m=grasp_part(env,obs,n,Rdown,base)
            print("  ",desc,"c0 ->",int(g),m,flush=True); continue
        a=f['side_a']; b=f['side_b']
        found=None
        for k,c in cands.items():
            obs,info=env.reset(seed=seed)
            cc=np.array([c[0]*a,c[1]*b,0.0])
            obs,g,m=grasp_part(env,obs,n,Rdown,base,c=cc)
            if g: found=k; break
        print("  ",desc,"-> best",found,flush=True)
env.close()
