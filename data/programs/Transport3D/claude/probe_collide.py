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

env=make_env(); obs,_=env.reset(seed=3)
O=objs(obs); print("objs",{k:tuple(round(x,3) for x in v) for k,v in O.items()})
cubes=[k for k in O if k.startswith("cube")]
c=O[cubes[0]]; print("target cube",cubes[0],tuple(round(x,3) for x in c))
# home-pose footprint vs cube: approach from -x
obs=drive(env,obs,c[0]-1.2,c[1])
obs,p=approach(env,obs,(1,0),"HOME arm, cube from -x")
print("  dx to cube:",round(c[0]-p[0],3))
# now raise arm and try again
obs=uparm(env,obs); print("  arm now",fmt(rs(obs)[3:10],2))
obs,p2=approach(env,obs,(1,0),"UP arm, cube from -x")
print("  dx to cube:",round(c[0]-p2[0],3))
# approach same cube from +y
obs=drive(env,obs,p2[0]-0.8,c[1],order="xy")
obs=drive(env,obs,c[0],c[1]+1.0,order="yx")
obs,p3=approach(env,obs,(0,-1),"UP arm, cube from +y")
print("  dy to cube:",round(p3[1]-c[1],3))
env.close()

# table approach from -y with UP arm
env=make_env(); obs,_=env.reset(seed=3)
obs=uparm(env,obs)
obs=drive(env,obs,0.6,-1.6,order="yx")
obs,p=approach(env,obs,(0,1),"UP arm, table from -y (face y=-0.4)")
print("  gap:",round(-0.4-p[1],3))
# table approach from -x with UP arm
obs=drive(env,obs,-1.0,p[1],order="xy")
obs=drive(env,obs,-1.0,0.0,order="yx")
obs,p=approach(env,obs,(1,0),"UP arm, table from -x (face x=0.4)")
print("  gap:",round(0.4-p[0],3))
env.close()
