"""Drive arm to target joint config with P control; save obs snapshots."""
import numpy as np, json, sys
from env_client import make_env

def goto(env, qt, steps=60, grip=0.0, kp=2.0):
    obs=None
    for i in range(steps):
        if obs is None:
            a=np.zeros(11,dtype=np.float32); a[10]=grip
            obs,r,te,tr,inf=env.step(a); continue
        q=np.asarray(obs)[96:103]
        a=np.zeros(11,dtype=np.float32)
        a[3:10]=np.clip(kp*(np.asarray(qt)-q),-0.1,0.1)
        a[10]=grip
        obs,r,te,tr,inf=env.step(a)
    return np.asarray(obs)

if __name__=="__main__":
    env=make_env()
    obs,info=env.reset(seed=0)
    home=np.asarray(obs)[96:103].copy()
    print("home",home)
    out={}
    cfgs={
      "c1":[0,0.5,3.14,-2.0,0,-0.9,1.57],
      "c2":[0,0.9,3.14,-1.5,0,-1.2,1.57],
      "c3":[0,0.3,3.14,-2.4,0,-0.3,1.57],
    }
    for name,qt in cfgs.items():
        o=goto(env,qt,steps=80)
        print(name,"q",np.round(o[96:103],3))
        out[name]=o.tolist()
    json.dump(out,open("cfg_obs.json","w"))
    env.close()
