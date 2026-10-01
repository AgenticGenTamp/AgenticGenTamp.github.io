from env_client import make_env
import numpy as np
def rob(obs):
    r=obs.get_object_from_name("robot"); f=obs.type_features[r.type]; d=obs.data[r]
    return {n:float(d[f.index(n)]) for n in f}
for dim,name in [(0,'base_x'),(1,'base_y'),(2,'base_rot')]:
    env=make_env(); obs,_=env.reset(seed=0); s0=rob(obs)
    key='pos_'+name
    # single pulse
    a=np.zeros(11,np.float32); a[dim]=0.1
    obs,*_=env.step(a); s1=rob(obs)
    a[dim]=0.0
    traj=[]
    for i in range(20):
        obs,*_=env.step(a); traj.append(rob(obs)[key])
    print("%s: pulse1 delta=%.5f then zeros-> cumulative from start:"%(name,s1[key]-s0[key]),
          [round(v-s0[key],5) for v in traj[:6]], "...final",round(traj[-1]-s0[key],5))
    env.close()
# constant hold transient from rest, then release
for dim,name in [(0,'base_x'),(1,'base_y'),(2,'base_rot')]:
    env=make_env(); obs,_=env.reset(seed=0); key='pos_'+name
    a=np.zeros(11,np.float32); a[dim]=0.1
    prev=rob(obs)[key]; d=[]
    for i in range(12):
        obs,*_=env.step(a); c=rob(obs)[key]; d.append(round(c-prev,5)); prev=c
    a[dim]=0.0; dr=[]
    for i in range(8):
        obs,*_=env.step(a); c=rob(obs)[key]; dr.append(round(c-prev,5)); prev=c
    print("%s hold-0.1 per-step:"%name,d,"| after release:",dr)
    env.close()
