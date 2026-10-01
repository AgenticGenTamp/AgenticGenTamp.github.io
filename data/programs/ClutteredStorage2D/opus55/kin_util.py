import numpy as np
from env_client import make_env
env=make_env()
def R(obs):
    r=obs.get_object_from_name('robot')
    return tuple(round(float(obs.get(r,f)),4) for f in ['x','y','theta','arm_joint','vacuum'])
def A(dx=0,dy=0,dt=0,da=0,v=0): return np.array([dx,dy,dt,da,v],dtype=np.float32)
def blocks(obs):
    out={}
    for n in sorted(obs.get_object_names()):
        if n.startswith('block'):
            o=obs.get_object_from_name(n); out[n]=tuple(round(float(obs.get(o,f)),4) for f in ['x','y','theta'])
    return out
import math
def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi
def set_theta(obs,t):
    for _ in range(40):
        d=wrap(t-R(obs)[2])
        if abs(d)<1e-6: break
        obs,*_=env.step(A(dt=float(np.clip(d,-0.196,0.196))))
    return obs
def set_arm(obs,a):
    for _ in range(20):
        d=a-R(obs)[3]
        if abs(d)<1e-6: break
        obs,*_=env.step(A(da=float(np.clip(d,-0.1,0.1))))
    return obs
def move_to(obs,x,y):
    for _ in range(300):
        r=R(obs); dx=x-r[0]; dy=y-r[1]
        if abs(dx)<1e-6 and abs(dy)<1e-6: break
        o2,*_=env.step(A(float(np.clip(dx,-.05,.05)),float(np.clip(dy,-.05,.05))))
        if R(o2)==r: return o2,False
        obs=o2
    return obs,True
def push_limit(obs,dx,dy):
    # step in direction with shrinking step until blocked at 1e-4 resolution
    s=1.0
    while s>1e-4:
        o2,*_=env.step(A(dx*s,dy*s))
        if R(o2)==R(obs): s/=2
        else: obs=o2
    return obs
