import numpy as np
from ik import ik
RDOWN=np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)

def local_to_world(base, p):
    x,y,th=base; c,s=np.cos(th),np.sin(th)
    return np.array([x+c*p[0]-s*p[1], y+s*p[0]+c*p[1], p[2]])

def world_to_local(base, p):
    x,y,th=base; c,s=np.cos(th),np.sin(th)
    dx,dy=p[0]-x,p[1]-y
    return np.array([c*dx+s*dy, -s*dx+c*dy, p[2]])

def move(env, obs, p, R=RDOWN, steps=25, grip=None, tol=0.01, base=None):
    q=obs[128:135].copy()
    qd,_=ik(np.array(p),R,q)
    g = obs[135] if grip is None else grip
    for i in range(steps):
        a=np.zeros(11); a[3:10]=np.clip((qd-obs[128:135]),-0.1,0.1); a[10]=g
        if base is not None: a[0:3]=base
        obs,r,t,tr,_=env.step(a)
        if np.abs(qd-obs[128:135]).max()<tol: break
    return obs, np.abs(qd-obs[128:135]).max()
