import numpy as np
from probe_util import *
def uparm(env,obs):
    for _ in range(30):
        s=rs(obs)[3:10]; d=np.clip(-s,-0.2,0.2)
        if np.max(np.abs(d))<1e-6: break
        o2,_,_,_,_=env.step(act(**{f"j{i+1}":d[i] for i in range(7)}))
        if np.allclose(rs(o2)[3:10],s): break
        obs=o2
    return obs
def dr(env,obs,tx,ty):
    for ax in "yx":
        for _ in range(80):
            s=rs(obs); d=np.clip((tx-s[0]) if ax=="x" else (ty-s[1]),-0.2,0.2)
            if abs(d)<1e-6: break
            o2,_,_,_,_=env.step(act(**({"bx":d} if ax=="x" else {"by":d})))
            if np.allclose(rs(o2),s): break
            obs=o2
    return obs
def push(env,obs,j,step,n=60):
    cur=rs(obs)
    for i in range(n):
        o2,_,_,_,_=env.step(act(**{f"j{j}":step}))
        if np.allclose(rs(o2),cur): break
        obs=o2;cur=rs(o2)
    return obs,cur[2+j]

# 1. floor collision for arm: j2->1.57 then j4 + to swing forearm down
env=make_env(); obs,_=env.reset(seed=3); obs=uparm(env,obs); obs=dr(env,obs,-2.0,2.0)
for _ in range(8):
    obs,_,_,_,_=env.step(act(j2=0.2))
obs,v=push(env,obs,2,0.05); print("j2 stop (free space, arm fwd):",round(v,3))
obs,v4=push(env,obs,4,0.05); print("  then j4 stop:",round(v4,3),"(limit 2.66)")
obs,v6=push(env,obs,6,0.05); print("  then j6 stop:",round(v6,3),"(limit 2.23)")
print("  pose",fmt(rs(obs)[3:10],2))
env.close()
# 2. same but standing right next to box0 (does box block the arm?)
env=make_env(); obs,_=env.reset(seed=3); O=objs(obs); b=O["box0"]
obs=uparm(env,obs); obs=dr(env,obs,b[0]-0.35,b[1])
print("box0",tuple(round(x,2) for x in b),"base",fmt(rs(obs)[:3]))
for _ in range(8):
    obs,_,_,_,_=env.step(act(j2=0.2))
obs,v=push(env,obs,2,0.05); print("near box: j2 stop",round(v,3))
obs,v4=push(env,obs,4,0.05); print("near box: j4 stop",round(v4,3))
print("  box0 pos now",tuple(round(x,3) for x in objs(obs)["box0"]))
env.close()
# 3. arm vs table top: base in front of table, reach over
env=make_env(); obs,_=env.reset(seed=3); obs=uparm(env,obs); obs=dr(env,obs,0.0,0.0)
obs=dr(env,obs,0.232,0.0)
print("base at table",fmt(rs(obs)[:3]))
obs,v=push(env,obs,2,0.05); print("table front: j2 stop",round(v,3))
obs,v4=push(env,obs,4,0.05); print("table front: j4 stop",round(v4,3))
env.close()
