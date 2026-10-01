import numpy as np
from env_client import make_env
from approach import GeneratedApproach, world_fk, ik_solutions, grasp_R, pose_err
env=make_env(); obs,info=env.reset(seed=1)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
for t in range(16):
    a=ap.get_action(obs); obs,_,_,_,_=env.step(a)
r=ap._robot(obs); base=r["base"]; q=r["q"]
g0=ap._blocks(obs)["green0"]; d=ap.dir; R=grasp_R(np.arctan2(d[1],d[0]))
print("base",base,"tool",world_fk(q,base)[0])
pg=np.array([g0[0]-d[0]*0.02,g0[1]-d[1]*0.02,g0[2]+0.03])
for t in [0.20,0.16,0.12,0.08,0.04,0.0]:
    p=pg-d*t
    sols=ik_solutions(p,R,base,q,seeds=6,iters=120)
    if sols:
        e=pose_err(sols[0],base,p,R)
        print(t, "ok nsol",len(sols),"err",np.round(np.linalg.norm(e[:3]),5))
    else:
        from approach import _solve_one
        qq=_solve_one(q,base,p,R,150,0.5)
        e=pose_err(qq,base,p,R)
        print(t,"FAIL best err pos",np.round(np.linalg.norm(e[:3]),4),"rot",np.round(np.linalg.norm(e[3:]),4))
env.close()
