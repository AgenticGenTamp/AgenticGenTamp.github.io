import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
from ctl import rget, oget, act
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
env = make_env(); ap = GeneratedApproach(env.action_space, env.observation_space, {})
def setup():
    obs, info = env.reset(seed=3); ap.reset(obs,info)
    for i in range(3000):
        a=ap.get_action(obs)
        if ap.phase=='grasp_close': return obs
        obs,r,term,tr,info = env.step(a)
for dlong,dlat,darm in [(0,0,0),(0,0.02,0),(0,-0.02,0),(-0.03,0,0),(0.03,0,0),(0,0,0.06),(0,0,-0.06),(-0.05,0,0),(0.05,0,0),(0,0.04,0),(0,-0.04,0)]:
    obs=setup()
    hth=oget(obs,'hook','theta'); u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
    tgt=np.array([rget(obs,'x'),rget(obs,'y')])+dlong*u+dlat*nv
    for _ in range(60):
        dx=np.clip(tgt[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(tgt[1]-rget(obs,'y'),-0.05,0.05)
        if abs(dx)<0.002 and abs(dy)<0.002: break
        obs,_,_,_,_=env.step(act(dx=dx,dy=dy))
    for _ in range(20):
        d=np.clip(rget(obs,'arm_joint')+0 if darm==0 else darm,-0.1,0.1)
        if darm==0: break
        obs,_,_,_,_=env.step(act(da=darm)); break
    held=None
    for k in range(12):
        obs,_,_,_,_=env.step(act(dg=-0.02))
        if oget(obs,'hook','held')>0.5: held=round(rget(obs,'finger_gap'),3); break
    print("dlong",dlong,"dlat",dlat,"darm",darm,"arm",round(rget(obs,'arm_joint'),3),"HELD" if held else "fail",held)
env.close()
