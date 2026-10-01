import numpy as np
from env_client import make_env
from ctl import act, rget, oget
from grasp import grasp_seq, wrap
env=make_env()
for push in [False,True]:
    obs,info=env.reset(seed=42)
    def step(a):
        global obs
        obs,_,_,_,_=env.step(a)
    if push:
        # push stub left face rightwards a bit
        hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
        u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
        stand=np.array([hx,hy])+(-0.107-0.275)*u-0.45*nv
        for _ in range(60): step(act(dth=wrap(np.pi+hth-rget(obs,'theta')),da=-0.1,dg=-0.02))
        tgt=stand-0.3*u
        for _ in range(300):
            dx=np.clip(tgt[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(tgt[1]-rget(obs,'y'),-0.05,0.05)
            if abs(dx)<0.004 and abs(dy)<0.004: break
            step(act(dx=dx,dy=dy))
        for _ in range(12): step(act(dx=0.05))
        print("pushed hook to",round(oget(obs,'hook','x'),3),round(oget(obs,'hook','y'),3),round(oget(obs,'hook','theta'),3))
        # back off left
        for _ in range(10): step(act(dx=-0.05))
    obs2,n,ok=grasp_seq(env,obs)
    print("push",push,"grasp",ok)
env.close()
