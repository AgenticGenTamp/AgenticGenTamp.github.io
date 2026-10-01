"""Search arm/base offsets that grasp green0 after moving the blocker."""
import math
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach


def get(s, n, f):
    return float(s.get(s.get_object_from_name(n), f))


def command(p, s, base, q, grip=1.):
    a = np.zeros(11, np.float32)
    a[:2] = np.clip(base-p.robot(s), -.2, .2)
    a[2] = np.clip(p.w(p.theta-get(s,"robot","base_rot")), -.2, .2)
    for j in range(7):
        d = q[j]-get(s,"robot","joint_"+str(j+1))
        if j in (4,6): d=p.w(d)
        a[3+j] = np.clip(d,-.12,.12)
    a[10]=grip
    return a


def setup(env, seed):
    s,info=env.reset(seed=seed); p=GeneratedApproach(env.action_space,env.observation_space,{})
    p.reset(s,info)
    for _ in range(80):
        s,*_=env.step(p.get_action(s))
        if p.stage==5: break
    # Return along the opening centerline, with the arm above the walls.
    for alpha in (.31,.15):
        t=p.target(p.green+alpha*p.out)
        for _ in range(8): s,*_=env.step(p.motion(s,t,1,lift=True))
    base=p.target(p.blocker)
    for _ in range(8): s,*_=env.step(command(p,s,base,p.Q,1))
    return s,p,base


seed=int(sys.argv[1]) if len(sys.argv)>1 else 23
env=make_env()
try:
  # Each candidate gets an independent post-blocker state.
  candidates=[(.08,.06,.10)]
  for dq4 in np.arange(.06,.221,.02):
    for dx in (.03,.05,.07,.09):
      for dy in (.02,.04,.06,.08,.10): candidates.append((dq4,dx,dy))
  for ci,(dq4,dx,dy) in enumerate(candidates):
    s,p,base=setup(env,seed); q=p.Q.copy();q[3]+=dq4;base=base+np.array([dx,dy])
    for _ in range(8): s,*_=env.step(command(p,s,base,q,1))
    actualbase=p.robot(s);actualq=np.array([get(s,"robot","joint_"+str(j+1)) for j in range(7)])
    s,*_=env.step(command(p,s,base,q,-1))
    if get(s,"green0","grasp_active")>.5:
      print("HIT",seed,ci,"candidate",dq4,dx,dy,"base",actualbase,"q",actualq)
      qlift=q.copy();qlift[1]-=.2
      for _ in range(4): s,*_=env.step(command(p,s,base,qlift,0))
      for alpha in (.20,.40,.62):
        waypoint=base+alpha*p.out
        for _ in range(4): s,*_=env.step(command(p,s,waypoint,qlift,0))
      print("EXTRACT",get(s,"robot","grasp_active"),"base",p.robot(s),
            "green",[get(s,"green0","pose_"+x) for x in "xyz"])
      break
    if ci%20==0: print("try",ci,(dq4,dx,dy),"actual",actualbase,np.round(actualq-p.Q,3))
  else: print("MISS",seed,len(candidates))
finally: env.close()
