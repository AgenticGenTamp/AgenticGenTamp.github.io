import sys, numpy as np
from env_client import make_env
s=18
env=make_env(); obs,info=env.reset(seed=s)
def rb(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,'pos_base_x')),float(o.get(r,'pos_base_y')),float(o.get(r,'pos_base_rot'))])
def vel(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,'vel_base_x')),float(o.get(r,'vel_base_y'))])
a0=np.zeros(11,dtype=np.float32)
for i in range(200):
    obs,r,term,trunc,info=env.step(a0)
    if i%25==0: print("zero",i,np.round(rb(obs),3),np.round(vel(obs),3))
print("--- now drive slowly")
for i in range(200):
    b=rb(obs)[:2]; d=np.array([2.55,2.55])-b
    a=np.zeros(11,dtype=np.float32); a[:2]=np.clip(d,-0.02,0.02)
    obs,r,term,trunc,info=env.step(a)
    if i%20==0 or term: print(i,np.round(rb(obs),3),np.round(vel(obs),3),term)
    if term: break
