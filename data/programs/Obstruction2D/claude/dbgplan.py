import numpy as np
from env_client import make_env
import approach as A
env=make_env(); ap=A.GeneratedApproach(env.action_space, env.observation_space, {})
obs,info=env.reset(seed=97, options={'object_count':4})
ap.reset(obs,info)
robot,rects,target,surface = ap.parse(obs)
cands=[]
for tx in ap.target_x_candidates(target,surface):
    cands.extend(ap._search(robot,rects,target,surface,tx))
cands.sort(key=lambda t:t[0])
print("n candidates", len(cands))
seen=set()
for c,plan in cands[:60]:
    key=tuple((m[0],round(m[1],2)) for m in plan)
    if key in seen: continue
    seen.add(key)
    n=ap.sim_plan(robot,rects,target,plan)
    print(round(c,2), n, [(m[0],round(m[1],3)) for m in plan])
env.close()
