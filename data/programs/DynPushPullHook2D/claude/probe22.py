from env_client import make_env
from ctl import oget, rget
import numpy as np
env=make_env()
rows=[]
for seed in range(60):
    obs,info=env.reset(seed=seed)
    hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
    u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
    fe=np.array([hx,hy])-1.4*u-0.0533*nv
    b31=fe-0.31*u; b55=fe-0.545*u
    bx=oget(obs,'target_block','x'); by=oget(obs,'target_block','y'); bw=oget(obs,'target_block','width'); bt=oget(obs,'target_block','theta')
    ext=abs(bw/2*np.sin(bt))+abs(bw/2*np.cos(bt))
    rows.append((seed,info['object_count'],round(hx,2),round(hy,2),round(hth,2),round(b31[0],2),round(b31[1],2),round(b55[0],2),round(b55[1],2),round(bx,2),round(by,2),round(by+ext,2)))
ok31=sum(1 for r in rows if r[5]>=0.2775 and 0.244<=r[6]<=1.481)
ok55=sum(1 for r in rows if r[7]>=0.2775 and 0.244<=r[8]<=1.481)
print("feasible with arm retracted:",ok31,"/60; with arm extended:",ok55)
print("block top max", max(r[11] for r in rows), "min", min(r[11] for r in rows))
print("block x min", min(r[9] for r in rows))
print("hook theta range", min(r[4] for r in rows), max(r[4] for r in rows))
print("counts", sorted(set(r[1] for r in rows)))
for r in rows[:20]: print(r)
env.close()
