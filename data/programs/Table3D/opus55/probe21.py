import numpy as np, sys
from env_client import make_env
from approach import *
H=0.3947
J=[f'joint_{i}' for i in range(1,8)]
def getq(o):
    r=o.get_object_from_name('robot'); return np.array([o.get(r,n) for n in J])
def rotx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def roty(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def step_to(env,obs,qt,grip=0.0):
    for _ in range(30):
        q=getq(obs); d=qt-q
        if np.abs(d).max()<1e-6: return obs,True
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(d,-0.4,0.4); a[10]=grip
        obs2,*_=env.step(a)
        if np.abs(getq(obs2)-q).max()<1e-9: return obs2,False
        obs=obs2
    return obs,False
env=make_env()
out=[]
for axis in ['z']:
  for th in [0.1,0.2,0.3,0.45,0.6,0.785]:
    obs,_=env.reset(seed=0); q=getq(obs)
    c=obs.get_object_from_name('cube0'); r=obs.get_object_from_name('robot')
    cp=np.array([obs.get(c,'pose_x')-ARM_OFFSET_X,obs.get(c,'pose_y'),obs.get(c,'pose_z')-H])
    R=down_R(np.pi/2+th)
    ok=True
    for back in [0.12,0.05,0.0]:
        fl=cp-R[:,2]*(0.16+back)
        qt,e=ik(q,fl,R); obs,ok=step_to(env,obs,qt); q=getq(obs)
        if not ok: break
    a=np.zeros(11,dtype=np.float32); a[10]=-1; obs,*_=env.step(a)
    out.append((axis,th,ok,obs.get(r,'grasp_active')))
print(out)
env.close()
