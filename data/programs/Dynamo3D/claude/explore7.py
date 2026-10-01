import numpy as np, sys
from env_client import make_env
seed=int(sys.argv[1]); 
env=make_env(); obs,info=env.reset(seed=seed)
def base(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,f)) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
names=[n for n in sorted(obs.get_object_names()) if n!='robot']
def chairs(o):
    return {n:[round(float(o.get(o.get_object_from_name(n),f)),3) for f in ['x','y','z']] for n in names}
print("init", [round(v,3) for v in base(obs)], chairs(obs))
c=obs.get_object_from_name(names[0]); cp=np.array([float(obs.get(c,'x')),float(obs.get(c,'y'))])
# approach chair from robot side
for i in range(300):
    b=base(obs); d=cp-b[:2]
    a=np.zeros(11,dtype=np.float32); a[0]=np.clip(d[0],-0.1,0.1); a[1]=np.clip(d[1],-0.1,0.1)
    obs,rew,term,trunc,info=env.step(a)
    if abs(rew+1)>1e-9 or term or trunc or i%15==0:
        print(i, round(rew,5), term,trunc,info, [round(v,3) for v in base(obs)], chairs(obs))
    if term or trunc: break
