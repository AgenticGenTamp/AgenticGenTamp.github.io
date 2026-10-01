"""Probe exact vertical pickup of the initially stored shelf blocks."""

import sys
import numpy as np
from env_client import make_env


def g(s, o, k): return float(s.get(o, k))
def clip(x, m): return max(-m, min(m, x))


seed, count, index = map(int, sys.argv[1:4])
side = len(sys.argv) > 4 and sys.argv[4] == "side"
diag = len(sys.argv) > 4 and sys.argv[4] == "diag"
offset = float(sys.argv[4]) if len(sys.argv) > 4 and not (side or diag) else 0.0
env = make_env(); s, _ = env.reset(seed=seed, options={"object_count": count})
r = s.get_object_from_name("robot"); b = s.get_object_from_name("block%d" % index)
phase = "nav"; initial = (g(s,b,"x"),g(s,b,"y"),g(s,b,"theta")); ticks = 0
for step in range(250):
    rx,ry,rt,arm = [g(s,r,k) for k in ("x","y","theta","arm_joint")]
    bx,by,bt = [g(s,b,k) for k in ("x","y","theta")]
    ticks += 1
    if phase == "nav":
        heading = bt + np.pi / 2
        if diag:
            gx,gy=bx+.48,by-.48; heading=np.arctan2(by-gy,bx-gx)
        else:
            gx,gy,heading = ((bx-.54, by, 0.0) if side else
                         (bx-.54*np.cos(heading), by-.54*np.sin(heading), heading))
        gx -= offset*np.sin(heading); gy += offset*np.cos(heading)
        e=(gx-rx,gy-ry, ((heading-rt+np.pi)%(2*np.pi)-np.pi), .2-arm)
        a=np.array([clip(e[0],.05),clip(e[1],.05),clip(e[2],.196),clip(e[3],.1),0],np.float32)
        if max(map(abs,e)) < .01: phase="extend";ticks=0
    elif phase == "extend":
        a=np.array([0,0,0,.05,1],np.float32)
        if ticks>8: phase="grip";ticks=0
    elif phase == "grip":
        a=np.array([0,0,0,0,1],np.float32)
        if ticks>3: phase="lift";ticks=0
    elif phase == "lift":
        # Pull straight back after toggling suction at aligned contact.
        if diag: a=np.array([.03,0,0,0,1],np.float32)
        else: a=np.array([-.03*np.cos(rt),-.03*np.sin(rt),0,0,1],np.float32)
    s,_,term,trunc,_=env.step(a)
    moved=np.hypot(bx-initial[0],by-initial[1])
    if step%5==0 or moved>.002:
        print(step+1,phase,"r",*(round(g(s,r,k),3) for k in ("x","y","theta","arm_joint","vacuum")),
              "b",*(round(g(s,b,k),3) for k in ("x","y","theta")))
    if term or trunc: break
env.close()
