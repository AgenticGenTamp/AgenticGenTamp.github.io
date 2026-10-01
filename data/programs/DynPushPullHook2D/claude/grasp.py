import numpy as np
from ctl import act, rget, oget
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def grasp_seq(env, obs, d=0.545, lat=0.0, safey=0.26, verbose=False):
    """Returns (obs, nsteps, held)."""
    n=0
    def step(a):
        nonlocal obs,n
        obs,_,_,_,_=env.step(a); n+=1
    hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
    u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
    fe=np.array([hx,hy])-1.4*u+(-0.0533+lat)*nv
    base=fe-d*u
    for _ in range(80):
        if abs(wrap(hth-rget(obs,'theta')))<0.004 and rget(obs,'finger_gap')>0.318 and rget(obs,'arm_joint')<0.245: break
        step(act(dth=wrap(hth-rget(obs,'theta')),dg=0.02,da=-0.1))
    stage=base-0.30*u
    for _ in range(500):
        dx=np.clip(stage[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(safey-rget(obs,'y'),-0.05,0.05)
        if abs(dx)<0.003 and abs(dy)<0.003: break
        step(act(dx=dx,dy=dy))
    for _ in range(500):
        dy=np.clip(stage[1]-rget(obs,'y'),-0.05,0.05)
        if abs(dy)<0.003: break
        step(act(dy=dy))
    for _ in range(500):
        dx=np.clip(base[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(base[1]-rget(obs,'y'),-0.05,0.05)
        if abs(dx)<0.002 and abs(dy)<0.002: break
        step(act(dx=dx,dy=dy))
    for _ in range(30):
        if rget(obs,'arm_joint')>0.4799: break
        step(act(da=0.1))
    for i in range(15):
        step(act(dg=-0.02))
        if oget(obs,'hook','held')>0.5: break
    return obs,n,oget(obs,'hook','held')>0.5
