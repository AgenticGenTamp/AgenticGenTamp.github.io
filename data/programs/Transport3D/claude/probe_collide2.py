import numpy as np
from probe_util import *

def drive(env,obs,tx,ty,order="yx"):
    for ax in order:
        for _ in range(80):
            s=rs(obs)
            d = (tx-s[0]) if ax=="x" else (ty-s[1])
            d=np.clip(d,-0.2,0.2)
            if abs(d)<1e-6: break
            o2,_,_,_,_=env.step(act(**({"bx":d} if ax=="x" else {"by":d})))
            if np.allclose(rs(o2),s): print("   blocked while routing",ax,fmt(s[:2])); break
            obs=o2
    return obs

def approach(env,obs,dirv,label):
    cur=rs(obs)
    for i in range(60):
        o2,_,_,_,_=env.step(act(bx=dirv[0]*0.2,by=dirv[1]*0.2))
        if np.allclose(rs(o2),cur): break
        obs=o2;cur=rs(o2)
    for step in [0.05,0.01,0.002]:
        for _ in range(6):
            o2,_,_,_,_=env.step(act(bx=dirv[0]*step,by=dirv[1]*step))
            if np.allclose(rs(o2),cur): break
            obs=o2;cur=rs(o2)
    print(f"  {label}: stop at base=({cur[0]:+.3f},{cur[1]:+.3f})",flush=True)
    return obs,cur[:2]

def uparm(env,obs):
    for _ in range(30):
        s=rs(obs)[3:10]; d=np.clip(-s,-0.2,0.2)
        if np.max(np.abs(d))<1e-6: break
        o2,_,_,_,_=env.step(act(**{f"j{i+1}":d[i] for i in range(7)}))
        if np.allclose(rs(o2)[3:10],s): break
        obs=o2
    return obs


def rot(env,obs,target):
    for _ in range(30):
        s=rs(obs)[2]; d=np.clip(target-s,-0.2,0.2)
        if abs(d)<1e-6: break
        o2,_,_,_,_=env.step(act(br=d))
        if np.allclose(rs(o2),rs(obs)): break
        obs=o2
    return obs

# A: table from -x with base rotated
for rr in [0.0, 1.5708, 0.7854, 3.1416]:
    env=make_env(); obs,_=env.reset(seed=3); obs=uparm(env,obs)
    obs=rot(env,obs,rr)
    obs=drive(env,obs,-1.0,0.0,order="yx")
    obs,p=approach(env,obs,(1,0),f"rot={rr:.3f} table from -x")
    print("   gap_x:",round(0.4-p[0],3),"rot",round(rs(obs)[2],3),flush=True)
    env.close()

# B: table from -x at various y offsets (rot 0)
for yy in [0.0,0.3,0.39,0.45,0.6]:
    env=make_env(); obs,_=env.reset(seed=3); obs=uparm(env,obs)
    obs=drive(env,obs,-1.0,yy,order="yx")
    obs,p=approach(env,obs,(1,0),f"y={yy} table from -x")
    print("   stop_x:",round(p[0],3),flush=True)
    env.close()

# C: box0 collision (box0 half extents .1,.15,.1 -> top z=0.2)
env=make_env(); obs,_=env.reset(seed=3); O=objs(obs); b=O["box0"]; print("box0",b)
obs=uparm(env,obs)
obs=drive(env,obs,b[0]-1.2,b[1],order="yx")
obs,p=approach(env,obs,(1,0),"box0 from -x")
print("   dx to box0 center:",round(b[0]-p[0],3),flush=True)
env.close()
# D: home-pose arm vs table from -x (does folded arm stick out?)
env=make_env(); obs,_=env.reset(seed=3)
obs=drive(env,obs,-1.0,0.0,order="yx")
obs,p=approach(env,obs,(1,0),"HOME arm table from -x")
print("   gap_x:",round(0.4-p[0],3))
env.close()
