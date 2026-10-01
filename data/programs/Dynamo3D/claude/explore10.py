import numpy as np
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=1)
n='obstacle_chair'
def cs(o):
    c=o.get_object_from_name(n); return [round(float(o.get(c,f)),3) for f in ['x','y','z','qw','qx','qy','qz','vx','vy','vz','wx','wy','wz']]
def base(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,f)) for f in ['pos_base_x','pos_base_y']])
print("chair init",cs(obs),"base",base(obs))
cp=np.array([1.005,0.181])
for i in range(200):
    b=base(obs); d=cp-b; dist=np.linalg.norm(d)
    a=np.zeros(11,dtype=np.float32); a[:2]=np.clip(d/max(dist,1e-6)*0.02,-0.1,0.1)
    obs,rew,term,trunc,info=env.step(a)
    b2=base(obs)
    print(i, round(rew,4), term, "d=%.3f"%np.linalg.norm(cp-b2), np.round(b2,3), cs(obs))
    if term or trunc: break
