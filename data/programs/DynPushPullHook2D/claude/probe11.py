from env_client import make_env
from ctl import act, rget, oget
import numpy as np
env=make_env()
res={}
for tx in [round(v,3) for v in np.arange(1.35,3.36,0.05)]:
    obs,info=env.reset(seed=42)
    def H(): return np.array([oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')])
    for _ in range(120):
        d=(np.pi/2-rget(obs,'theta')+np.pi)%(2*np.pi)-np.pi
        if abs(d)<0.01: break
        obs,_,_,_,_=env.step(act(dth=d,da=-0.1,dg=-0.02))
    for _ in range(300):
        dx=np.clip(tx-rget(obs,'x'),-0.05,0.05); dy=np.clip(1.45-rget(obs,'y'),-0.05,0.05)
        if abs(dx)<0.004 and abs(dy)<0.004: break
        obs,_,_,_,_=env.step(act(dx=dx,dy=dy))
    h0=H(); hit=None
    for _ in range(60):
        obs,_,_,_,_=env.step(act(dy=-0.01))
        if np.abs(H()-h0).max()>2e-4:
            hit=rget(obs,'y'); break
    res[tx]= None if hit is None else round(hit-0.24,4)
print(res)
env.close()
