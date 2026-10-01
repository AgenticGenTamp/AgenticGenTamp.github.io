import sys, time
from env_client import make_env
import approach
from approach import GeneratedApproach
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {})
t0=time.time(); ap.reset(obs,info); ta=time.time()-t0; te=0; big=[]
for s in range(1,1001):
    t=time.time(); a=ap.get_action(obs); d=time.time()-t; ta+=d
    if d>0.5: big.append((s,round(d,1),round(ap.elapsed(),1)))
    t=time.time(); obs,r,term,trunc,info=env.step(a); te+=time.time()-t
    if term or trunc: break
print(seed,term,s,'agent',round(ta,1),'env',round(te,1)); print(big[:40])
