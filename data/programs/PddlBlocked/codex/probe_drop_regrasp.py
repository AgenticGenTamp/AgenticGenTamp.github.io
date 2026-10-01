"""Try blocker relocation paths followed by green regrasp on shallow-east seeds."""
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))

def move(e,s,p,t,lift=True,q4=0.,grip=1.,steps=20):
    for _ in range(steps):
        a=p.motion(s,t,grip,lift=lift,q4add=q4)
        s,*_=e.step(a)
    return s

def trial(seed,delta,route_kind):
    e=make_env();s,info=e.reset(seed=seed)
    p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
    # Stop with lifted blocker in hand, before the built-in relocation.
    for _ in range(80):
        if p.stage==3: break
        s,*_=e.step(p.get_action(s))
    graspbase=p.robot(s).copy()
    dropbase=graspbase+np.asarray(delta)
    s=move(e,s,p,dropbase,True,steps=20)
    a=np.zeros(11,np.float32);a[10]=1.;s,*_=e.step(a)
    dropped=g(s,'robot','grasp_active')<.5
    # Stay with the exact blocker-grasp tool heading. Route outside the table.
    greenbase=p.target(p.green)
    routes={
      0:[np.array([greenbase[0],max(1.52,p.robot(s)[1])]),greenbase],
      1:[graspbase,greenbase],
      2:[np.array([5.,max(1.52,p.robot(s)[1])]),np.array([greenbase[0],max(1.52,p.robot(s)[1])]),greenbase],
    }[route_kind]
    for t in routes:s=move(e,s,p,t,True,steps=20)
    s=move(e,s,p,greenbase,False,steps=12)
    a=np.zeros(11,np.float32);a[10]=-1.;s,*_=e.step(a)
    held=g(s,'green0','grasp_active')>.5
    out=(dropped,held,p.robot(s).copy(),
         [g(s,'blocker','pose_x'),g(s,'blocker','pose_y')])
    e.close();return out

seeds=[101,191] if len(sys.argv)==1 else [int(sys.argv[1])]
deltas=[(0,.36),(-.15,.36),(-.3,.36),(-.45,.36),(-.3,.55),(-.5,.55),
        (-.2,0),(-.35,0),(-.5,0),(-.2,-.25),(-.4,-.25),(.0,.55)]
for seed in seeds:
  for delta in deltas:
    for route in range(3):
      result=trial(seed,delta,route)
      print(seed,delta,route,result,flush=True)
      if result[1]: raise SystemExit
