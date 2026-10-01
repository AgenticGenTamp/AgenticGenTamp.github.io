import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach, world_fk
sd=int(sys.argv[1]) if len(sys.argv)>1 else 1
N=int(sys.argv[2]) if len(sys.argv)>2 else 80
env=make_env(); obs,info=env.reset(seed=sd)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
prev=None
for t in range(N):
    a=ap.get_action(obs)
    r=ap._robot(obs)
    tool,_=world_fk(r["q"],r["base"])
    print(f"t={t} task={ap.tasks[ap.task_i]} qlen={len(ap.queue)} rej={ap.rej_streak} retry={ap.retry} hold={r['holding']} tool={np.round(tool,3)} base={np.round(r['base'],2)} a={np.round(a,2)}")
    obs,rew,term,trunc,_=env.step(a)
    if term or trunc: print("TERM",term,t); break
print("blocks", {k:np.round(v,3).tolist() for k,v in ap._blocks(obs).items()})
env.close()
