import numpy as np
from util import *
def step(env,a):
    return env.step(np.asarray(a,dtype=float))
def move_until_blocked(env, obs, k, direction, maxdist=5.0, minstep=2e-4):
    """move DOF k in direction (+1/-1) until blocked; refine step size. returns obs"""
    s = LIM[k]; moved=0
    while s>=minstep and moved<maxdist:
        r0=rob(obs); a=np.zeros(5); a[k]=direction*s
        obs,_,_,_,_=step(env,a)
        r1=rob(obs)
        dd=abs(r1[k]-r0[k])
        if dd < 0.5*s: s/=2
        else: moved+=dd
    return obs
def seq(env,obs,target,order=(3,4,2,1,0),**kw):
    for k in order:
        t=[None]*5; t[k]=target[k]
        if target[k] is not None: obs,_=goto(env,obs,t,**kw)
    return obs
def prep(env,obs):
    """go to room center at y=1.2, arm down, arm 0.24, gap 0.32"""
    obs,_=goto(env,obs,[None,1.2,None,None,None],maxsteps=100)
    obs=seq(env,obs,[None,None,None,0.24,None],order=(3,))
    obs=seq(env,obs,[1.6,None,None,None,None],order=(0,))
    obs=seq(env,obs,[None,None,-np.pi/2,None,None],order=(2,))
    return obs
def rotate_by(env,obs,delta,maxsteps=100):
    """rotate by delta radians (signed), unwrapped, fixed direction"""
    t=rob(obs)[2]+delta
    for i in range(maxsteps):
        d=t-rob(obs)[2]
        if abs(d)<1e-3: break
        obs,_,_,_,_=step(env,[0,0,float(np.clip(d,-0.19,0.19)),0,0])
    return obs
