import numpy as np, sys
from env_client import make_env
import approach as A
seed=int(sys.argv[1]); oc=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':oc})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
term=-1
for t in range(1000):
    a=ap.get_action(obs); obs,r,te,tr,i=env.step(a)
    if te: term=t; break
cb={n:A.obj_pos(obs,n) for n in obs.get_object_names() if n.startswith('cube')}
ok=sum(1 for n,p in cb.items() if ap._placed(n,p))
print(f"seed{seed} oc{oc} term={term} placed={ok}/{len(cb)}", flush=True)
env.close()
