from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
rewards={}
def base(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,f)) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
# sweep: go to grid points
targets=[]
for x in np.arange(-4,5,2.0):
    for y in np.arange(-4,5,2.0):
        targets.append((x,y))
res=[]
step=0
for tx,ty in targets:
    for _ in range(200):
        b=base(obs)
        d=np.array([tx-b[0],ty-b[1]])
        if np.linalg.norm(d)<0.05: break
        a=np.zeros(11,dtype=np.float32)
        a[0]=np.clip(d[0],-0.1,0.1); a[1]=np.clip(d[1],-0.1,0.1)
        obs,rew,term,trunc,info=env.step(a); step+=1
        if term: print("TERM at",b,rew); break
    b=base(obs)
    res.append((round(tx,1),round(ty,1),round(float(b[0]),2),round(float(b[1]),2),round(rew,3)))
    if term: break
for r in res: print(r)
print("steps",step)
