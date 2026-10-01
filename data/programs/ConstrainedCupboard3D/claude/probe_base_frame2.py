from env_client import make_env
import numpy as np
def rob(obs):
    r=obs.get_object_from_name("robot"); f=obs.type_features[r.type]; d=obs.data[r]
    return {n:float(d[f.index(n)]) for n in f}
env=make_env(); obs,_=env.reset(seed=0)
a=np.zeros(11,np.float32); a[2]=0.1
for i in range(11): obs,*_=env.step(a)
a[2]=0.0
for _ in range(20): obs,*_=env.step(a)
s0=rob(obs); print("rot0=%.4f x0=%.4f y0=%.4f"%(s0['pos_base_rot'],s0['pos_base_x'],s0['pos_base_y']))
a[0]=0.1
prev=s0
for i in range(15):
    obs,*_=env.step(a); s=rob(obs)
    print("i%2d dx=%+.5f dy=%+.5f drot=%+.5f  vx=%+.4f vy=%+.4f | x=%.4f y=%.4f rot=%.4f"%(
      i,s['pos_base_x']-prev['pos_base_x'],s['pos_base_y']-prev['pos_base_y'],s['pos_base_rot']-prev['pos_base_rot'],
      s['vel_base_x'],s['vel_base_y'],s['pos_base_x'],s['pos_base_y'],s['pos_base_rot']))
    prev=s
env.close()
