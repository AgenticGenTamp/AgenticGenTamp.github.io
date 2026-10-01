import numpy as np
from probe_util import *
env = make_env()

def drive_base(env,obs,tx,ty):
    for _ in range(60):
        s=rs(obs); dx=np.clip(tx-s[0],-0.2,0.2); dy=np.clip(ty-s[1],-0.2,0.2)
        if abs(dx)<1e-6 and abs(dy)<1e-6: break
        o2,_,_,_,_=env.step(act(bx=dx,by=dy))
        if np.allclose(rs(o2),s): break
        obs=o2
    return obs

def drive_joints(env,obs,target):
    for _ in range(60):
        s=rs(obs)[3:10]; d=np.clip(np.array(target)-s,-0.2,0.2)
        if np.max(np.abs(d))<1e-6: break
        kw={f"j{i+1}":d[i] for i in range(7)}
        o2,_,_,_,_=env.step(act(**kw))
        if np.allclose(rs(o2)[3:10],s):
            # try one joint at a time
            moved=False
            for i in range(7):
                if abs(d[i])<1e-6: continue
                o3,_,_,_,_=env.step(act(**{f"j{i+1}":d[i]}))
                if not np.allclose(rs(o3)[3:10],rs(o2)[3:10]): moved=True
                o2=o3
            obs=o2
            if not moved: break
        else: obs=o2
    return obs

def sweep(env,obs,j,sign):
    """returns achieved extreme and refined boundary"""
    key=f"j{j}"
    cur=rs(obs)
    for _ in range(120):
        o2,_,_,_,_=env.step(act(**{key:sign*0.2}))
        if np.allclose(rs(o2),cur): break
        obs=o2; cur=rs(o2)
    coarse=cur[2+j]
    # refine with smaller steps
    for step in [0.05,0.01,0.002]:
        for _ in range(6):
            o2,_,_,_,_=env.step(act(**{key:sign*step}))
            if np.allclose(rs(o2),cur): break
            obs=o2; cur=rs(o2)
    return obs, coarse, cur[2+j]

HOME=[0,-0.35,-3.1416,-2.5,0,-0.87,1.5708]
UP=[0,0,0,0,0,0,0]

HOMEs=None
for j in range(1,8):
    for sign in (+1,-1):
        obs,_=env.reset(seed=0)
        obs=drive_base(env,obs,-2.0,-2.0)
        obs=drive_joints(env,obs,UP)
        st=rs(obs)[3:10]
        if np.max(np.abs(st))>1e-6: print("WARN not at zeros",fmt(st))
        obs,coarse,fine=sweep(env,obs,j,sign)
        print(f"joint{j} sign{sign:+d}: coarse={coarse:+.4f} refined={fine:+.4f} finalpose={fmt(rs(obs)[3:10],3)}",flush=True)
env.close()
