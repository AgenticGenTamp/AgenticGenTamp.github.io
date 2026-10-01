"""Run seed0/count7 to packing, then probe extraction of block0."""
import math, sys
import numpy as np
from approach import GeneratedApproach
from env_client import make_env

mode = sys.argv[1]
strategy = sys.argv[2] if len(sys.argv) > 2 else "pull"
base_x = float(sys.argv[3]) if len(sys.argv) > 3 else .225
env=make_env(); s,info=env.reset(seed=0,options={"object_count":7})
p=GeneratedApproach(env.action_space,env.observation_space,{}); p.reset(s,info)
def g(o,k): return float(s.get(o,k))
def ang(x): return (x+math.pi)%(2*math.pi)-math.pi
term=False
for step in range(860):
    s,_,term,trunc,_=env.step(p.get_action(s))
    if term or trunc: break
r=s.get_object_from_name("robot"); b=s.get_object_from_name("block0")
print("start",step+1,[round(g(r,k),3) for k in ("x","y","theta","arm_joint")],
      [round(g(b,k),3) for k in ("x","y","theta")],term)
phase="nav"; ticks=0; cycles=0
off=float(mode)
for q in range(180):
    rx,ry,rt,arm=(g(r,k) for k in ("x","y","theta","arm_joint"))
    bx,by=(g(b,k) for k in ("x","y"))
    # Robot stays within its left boundary; aim at an offset point on block0.
    if strategy == "diag":
        gx,gy,th=.567,2.199,3*math.pi/4
    else:
        gx,gy=base_x,2.08
        targetx=bx+off
        th=math.atan2(by-gy,targetx-gx)
    ticks+=1
    if phase=="nav":
        e=(gx-rx,gy-ry,ang(th-rt),.2-arm)
        a=np.array([np.clip(e[0],-.05,.05),np.clip(e[1],-.05,.05),
                    np.clip(e[2],-.196,.196),np.clip(e[3],-.1,.1),0],np.float32)
        if max(map(abs,e))<.012: phase="extend";ticks=0
    elif phase=="extend":
        a=np.array([0,0,0,.08,1],np.float32)
        if ticks>10: phase="pull";ticks=0
    elif phase=="pull":
        if strategy == "ccw": a=np.array([0,0,.08,0,1],np.float32)
        elif strategy == "cw": a=np.array([0,0,-.08,0,1],np.float32)
        elif strategy == "diag":
            a=np.array([0,0,-.05,0,1],np.float32)
        elif strategy == "right": a=np.array([.03,0,0,0,1],np.float32)
        else: a=np.array([0,-.03,0,0,1],np.float32)
        if strategy == "diag" and ticks >= 1:
            phase="reset";ticks=0;cycles+=1
    elif phase=="reset":
        if ticks < 2: a=np.array([0,0,0,0,0],np.float32)
        elif ticks < 5: a=np.array([0,-.05,0,0,0],np.float32)
        else: a=np.array([0,0,0,-.1,0],np.float32)
        if ticks > 5 and arm < .22: phase="nav";ticks=0
    s,_,term,trunc,_=env.step(a)
    if q%10==0 or abs(g(b,"x")-bx)>.002 or abs(g(b,"y")-by)>.002:
        print(q+1,phase,[round(g(r,k),3) for k in ("x","y","theta","arm_joint","vacuum")],
              [round(g(b,k),3) for k in ("x","y","theta")],term)
    if term or trunc: break
env.close()
