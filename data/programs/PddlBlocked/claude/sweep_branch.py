import numpy as np
from env_client import make_env
from approach import (GeneratedApproach, world_fk, world_chain, ik_solutions,
                      grasp_R, dq_wrap)
from helper_remove import servo
env=make_env()
for back in [0.70,0.78,0.86]:
  # collect branches once
  obs,info=env.reset(seed=1)
  ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
  g0=ap._blocks(obs)["green0"]; d=ap.dir; perp=np.array([-d[1],d[0]])
  p=g0[:2]-d[:2]*back-perp*0.188
  base=np.array([p[0],p[1],np.arctan2(d[1],d[0])])
  R=grasp_R(np.arctan2(d[1],d[0]))
  pg=ap.grasp_pose(g0,d)
  p_pre=pg-d*0.20
  r=ap._robot(obs)
  sols=ik_solutions(p_pre,R,base,r["q"],seeds=14,iters=120)
  print(f"--- back={back}: {len(sols)} pregrasp branches")
  for bi,qp in enumerate(sols):
    obs,info=env.reset(seed=1)
    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
    # remove blocker with default plan but forced base
    ap.base_pose=base; ap.queue=[]
    for t in range(60):
        if ap.task_i>=2: break
        a=ap.get_action(obs); obs,_,_,_,_=env.step(a)
    if ap.task_i<2:
        print(f"  branch {bi}: blocker removal failed"); continue
    seq=ap._cart_path(base,qp,[pg-d*o for o in [0.15,0.10,0.06,0.03,0.0]],R,max_jump=0.9)
    if seq is None: print(f"  branch {bi}: cart fail"); continue
    obs,rej=servo(env,ap,obs,base,qp,25)
    if rej: print(f"  branch {bi}: pregrasp rejected"); continue
    bad=False
    for q in seq:
        obs,rj=servo(env,ap,obs,base,q,4)
        if rj: bad=True; break
    q=ap._robot(obs)["q"]
    a=np.zeros(11,dtype=np.float32); a[10]=-1.0; obs,_,_,_,_=env.step(a)
    pts,_=world_chain(qp,base)
    print(f"  branch {bi}: rej={bad} GRASP={ap._robot(obs)['holding']} elbow={np.round(pts[2],3)} wrist={np.round(pts[3],3)} tool={np.round(world_fk(q,base)[0],3)}")
env.close()
