from env_client import make_env
import numpy as np
F=['pos_base_x','pos_base_y','pos_base_rot']
def rob(obs):
    r=obs.get_object_from_name("robot"); f=obs.type_features[r.type]; d=obs.data[r]
    return {n:float(d[f.index(n)]) for n in f}
env=make_env(); obs,_=env.reset(seed=0)
a=np.zeros(11,np.float32)
# rotate to ~1.0 rad
a[2]=0.1
for i in range(400):
    obs,*_=env.step(a)
    if rob(obs)['pos_base_rot']>=1.0: break
s=rob(obs); print("after rot: steps",i+1,"rot",round(s['pos_base_rot'],4),"x",round(s['pos_base_x'],4),"y",round(s['pos_base_y'],4))
a[2]=0.0
for _ in range(30): obs,*_=env.step(a)  # settle
s0=rob(obs); print("settled rot",round(s0['pos_base_rot'],4))
a[0]=0.1
hist=[]
for i in range(40):
    obs,*_=env.step(a); s=rob(obs); hist.append((s['pos_base_x'],s['pos_base_y'],s['pos_base_rot']))
dx=hist[-1][0]-s0['pos_base_x']; dy=hist[-1][1]-s0['pos_base_y']
print("40 steps d_base_x=0.1: dx=%.4f dy=%.4f dtheta=%.4f"%(dx,dy,hist[-1][2]-s0['pos_base_rot']))
print("atan2(dy,dx)=%.4f rad ; rot=%.4f"%(np.arctan2(dy,dx), s0['pos_base_rot']))
print("per-step last5 dx", [round(hist[i][0]-hist[i-1][0],5) for i in range(-5,0)])
print("per-step last5 dy", [round(hist[i][1]-hist[i-1][1],5) for i in range(-5,0)])
env.close()
