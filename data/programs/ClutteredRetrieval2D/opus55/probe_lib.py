import numpy as np
from env_client import make_env
env=make_env()
def byname(obs,n): return obs.get_object_from_name(n)
def g(obs,n,f): return float(obs.get(byname(obs,n),f))
def A(dx=0,dy=0,dth=0,da=0,vac=0): return np.array([dx,dy,dth,da,vac],dtype=np.float32)
def rob(obs): return tuple(round(g(obs,'robot',f),4) for f in ['x','y','theta','arm_joint','vacuum'])
def obj(obs,n): return tuple(round(g(obs,n,f),4) for f in ['x','y','theta'])
def names(obs): return sorted(o.name for o in obs.data)
def approach(o,dx,dy,steps=(0.05,0.01,0.002,0.0005),vac=0):
    """move in unit dir (dx,dy) until blocked, refining step size"""
    for st in steps:
        for i in range(400):
            prev=rob(o)
            o2,r,t,tr,info=env.step(A(dx*st,dy*st,0,0,vac))
            if rob(o2)[:2]==prev[:2]: o=o2; break
            o=o2
    return o
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def goto(o,x=None,y=None,th=None,aj=None,vac=0,maxit=300):
    for i in range(maxit):
        r=rob(o)
        tx=r[0] if x is None else x; ty=r[1] if y is None else y
        tt=r[2] if th is None else th; ta=r[3] if aj is None else aj
        dx=np.clip(tx-r[0],-.05,.05); dy=np.clip(ty-r[1],-.05,.05)
        dt=np.clip(wrap(tt-r[2]),-.196,.196); da=np.clip(ta-r[3],-.1,.1)
        if max(abs(dx),abs(dy),abs(dt),abs(da))<1e-6: break
        o2,rew,term,tr,info=env.step(A(dx,dy,dt,da,vac))
        if rob(o2)==rob(o) : return o2,False
        o=o2
    return o,True
def bcenter(o,n='target_block'):
    x,y,t,w,h=[g(o,n,f) for f in ['x','y','theta','width','height']]
    return x+np.cos(t)*w/2-np.sin(t)*h/2, y+np.sin(t)*w/2+np.cos(t)*h/2, t
