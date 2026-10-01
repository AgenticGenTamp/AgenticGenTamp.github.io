from env_client import make_env
from ctl import act, rget, oget
import numpy as np
env=make_env()
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def trial(seed,d,arm):
    obs,info=env.reset(seed=seed)
    def step(a):
        nonlocal obs
        obs,_,_,_,_=env.step(a)
    hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
    u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
    fe=np.array([hx,hy])-1.4*u-0.0533*nv
    base=fe-d*u
    for _ in range(60): step(act(dth=wrap(hth-rget(obs,'theta')),dg=0.02,da=-0.1))
    stage=base-0.6*u
    for _ in range(400):
        dx=np.clip(stage[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(0.3-rget(obs,'y'),-0.05,0.05)
        if abs(dx)<0.003 and abs(dy)<0.003: break
        step(act(dx=dx,dy=dy))
    for _ in range(400):
        dy=np.clip(stage[1]-rget(obs,'y'),-0.05,0.05)
        if abs(dy)<0.003: break
        step(act(dy=dy))
    for _ in range(400):
        dx=np.clip(base[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(base[1]-rget(obs,'y'),-0.05,0.05)
        if abs(dx)<0.002 and abs(dy)<0.002: break
        step(act(dx=dx,dy=dy))
    h0=np.array([oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')])
    for _ in range(30):
        if abs(rget(obs,'arm_joint')-arm)<0.004: break
        step(act(da=np.clip(arm-rget(obs,'arm_joint'),-0.1,0.1)))
    m1=np.abs(np.array([oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')])-h0).max()
    held=None
    for i in range(15):
        step(act(dg=-0.02))
        if oget(obs,'hook','held')>0.5: held=round(rget(obs,'finger_gap'),3); break
    m2=np.abs(np.array([oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')])-h0).max()
    return held, round(m1,4), round(m2,4)
for arm in [0.24,0.30,0.36,0.48]:
    for d in [arm+0.03,arm+0.05,arm+0.07,arm+0.09,arm+0.12,arm+0.16,arm+0.2]:
        print("arm",arm,"d",round(d,3),trial(42,round(d,3),arm))
env.close()
