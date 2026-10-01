import numpy as np
from env_client import make_env
from ctl import act, rget, oget
from grasp import grasp_seq, wrap
env=make_env()
def trial(npush, wait):
    obs,info=env.reset(seed=42)
    def step(a):
        nonlocal obs
        obs,_,_,_,_=env.step(a)
    hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
    u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
    stand=np.array([hx,hy])+(-0.107-0.275)*u-0.45*nv
    for _ in range(60): step(act(dth=wrap(np.pi+hth-rget(obs,'theta')),da=-0.1,dg=-0.02))
    tgt=stand-0.3*u
    for _ in range(300):
        dx=np.clip(tgt[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(tgt[1]-rget(obs,'y'),-0.05,0.05)
        if abs(dx)<0.004 and abs(dy)<0.004: break
        step(act(dx=dx,dy=dy))
    h0=oget(obs,'hook','x')
    for _ in range(6+npush): step(act(dx=0.05))
    moved=oget(obs,'hook','x')-h0
    for _ in range(10): step(act(dx=-0.05))
    for _ in range(wait): step(act())
    obs2,n,ok=grasp_seq(env,obs)
    print("npush",npush,"wait",wait,"hookmoved",round(moved,4),"grasp",ok)
trial(0,0); trial(1,0); trial(3,0); trial(6,0); trial(6,200)
env.close()
