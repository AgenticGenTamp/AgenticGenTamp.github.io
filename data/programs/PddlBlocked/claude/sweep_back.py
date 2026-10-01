import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach, world_fk
env=make_env()
for back in [0.66,0.70,0.74,0.78,0.82,0.86]:
  for seed in [1]:
    obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
    g0=ap._blocks(obs)["green0"]; d=ap.dir
    perp=np.array([-d[1],d[0]])
    p=g0[:2]-d[:2]*back-perp*0.188
    ap.base_pose=np.array([p[0],p[1],np.arctan2(d[1],d[0])])
    ap.queue=[]
    n=0; term=False
    for t in range(120):
        a=ap.get_action(obs); obs,rew,term,trunc,_=env.step(a); n+=1
        if term: break
        if ap.task_i>=2 and ap.tasks[ap.task_i][0]=="place": break
    r=ap._robot(obs)
    print(f"back={back} seed={seed}: steps={n} task={ap.tasks[ap.task_i]} hold={r['holding']} term={term} tool={np.round(world_fk(r['q'],r['base'])[0],3)}")
env.close()
